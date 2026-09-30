import json
import shutil
from pathlib import Path

import pytest
import yaml

from o2a_gen.catalog import collect_catalog, compile_catalog
from o2a_gen.config import GenConfig, load_config
from o2a_gen.generate import generate
from o2a_gen.llm import FakeLLM, complete_json, extract_json
from o2a_gen.navigator import Catalog, navigate
from o2a_gen.plan import build_plan
from o2a_gen.procedure import _parse
from o2a_gen.refs import double_brace_refs, single_brace_refs, sql_params
from o2a_gen.validate import validate_dir

from .fake_tachyon import PROCEDURE_JSON, make_fake

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[2]


def make_cfg(engine: str = "native", **gen) -> GenConfig:
    return GenConfig(
        llm={"provider": "fake"},
        models={r: f"model-{r}" for r in ("extract", "ground", "navigate", "catalog_summary",
                                          "catalog_cards", "llm_agent")},
        embedding={"provider": "llm", "model": "fake-embed"},
        catalog={"engine": engine, "leaf_max": 2, "branching": 3, "min_group": 1,
                 "p": 3, "max_top": 3, "min_cluster_size": 1},
        generation={"prefix": "pmi_ddn", "pipeline_inputs": ["loan_number"],
                    "default_connection_env": "ECRM_DB_CONN", **gen},
    )


@pytest.fixture(scope="module", params=["native", "corpus2skill"])
def compiled_catalog(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"catalog_{request.param}")
    compile_catalog(FIX / "catalog_src", out, make_fake(), make_cfg(request.param))
    return out


# ---------------------------------------------------------------- refs / json

def test_reference_parsing():
    assert double_brace_refs({"a": ["{{ k1[0].x }}", "{{k2}}"]}) == {"k1", "k2"}
    assert single_brace_refs("use {a} and {b[0].c}; not {{d}} or {\"e\": 1}") == {"a", "b"}
    q = "SELECT x::int FROM t WHERE a = :loan_number AND b = '10:30' AND c = :other"
    assert sql_params(q) == {"loan_number", "other"}


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prefix {"a": [1, 2]} suffix') == {"a": [1, 2]}


def test_complete_json_retries_with_error_feedback():
    replies = iter(['{"n": 1}', '{"n": 2}'])
    fake = FakeLLM(default=lambda s, m: next(replies))

    def must_be_two(d):
        if d["n"] != 2:
            raise ValueError("n must be 2")

    assert complete_json(fake, model="m", system="s", prompt="p", validate=must_be_two) == {"n": 2}
    assert "n must be 2" in fake.calls[1]["messages"][-1]["content"]


# ---------------------------------------------------------------- config

def test_example_config_loads():
    cfg = load_config(REPO / "o2a_gen" / "gen_config.example.yaml")
    assert cfg.llm["chat"] == "o2a_gen.providers.tachyon_provider:chat"
    assert cfg.pipeline_inputs == ["loan_number"]


# ---------------------------------------------------------------- validator

def test_gold_fixture_has_no_errors():
    findings = validate_dir(FIX / "gold", ["loan_number"])
    assert not [f for f in findings if f.severity == "ERROR"], findings


def test_validator_catches_seeded_flaws(tmp_path):
    shutil.copytree(FIX / "gold", tmp_path / "p")
    d = tmp_path / "p"
    v = yaml.safe_load((d / "pmi_ddn_verdict_synthesizer.yaml").read_text())
    v["input_keys"] = ["pmi_ddn_icmp_review"]  # N3: instruction still uses msp_loan_data
    (d / "pmi_ddn_verdict_synthesizer.yaml").write_text(yaml.safe_dump(v))
    db = yaml.safe_load((d / "pmi_ddn_fetch_msp_loan_data.yaml").read_text())
    db["default_db_yaml"] = db["default_db_yaml"].replace("${ENV:ECRM_DB_CONN}", "teradata://u:pw@host")
    (d / "pmi_ddn_fetch_msp_loan_data.yaml").write_text(yaml.safe_dump(db))
    (d / "pmi_ddn_approval_gate.yaml").unlink()

    codes = {(f.code, f.agent) for f in validate_dir(d, ["loan_number"]) if f.severity == "ERROR"}
    assert ("undeclared-ref", "pmi_ddn_verdict_synthesizer") in codes
    assert ("db-credential", "pmi_ddn_fetch_msp_loan_data") in codes
    assert ("missing-agent", "pmi_ddn_pipeline") in codes


def test_validator_catches_read_before_write(tmp_path):
    shutil.copytree(FIX / "gold", tmp_path / "p")
    pre = tmp_path / "p" / "pmi_ddn_pre_process.yaml"
    data = yaml.safe_load(pre.read_text())
    data["sub_agents"].reverse()  # flag now runs before the fetch
    pre.write_text(yaml.safe_dump(data))
    errs = [f for f in validate_dir(tmp_path / "p", ["loan_number"]) if f.code == "order"]
    assert errs and errs[0].agent == "pmi_ddn_icmp_required_flag"


# ---------------------------------------------------------------- planning

def test_plan_names_keys_and_shared_branch_key():
    plan = build_plan(_parse(PROCEDURE_JSON), make_cfg())
    by = plan.by_step
    assert by["S1"].name == "pmi_ddn_fetch_msp_loan_data"
    assert by["S1"].output_key == "pmi_ddn_msp_loan_data"
    assert by["S2"].output_key == "pmi_ddn_icmp_required_flag"
    assert by["S3"].agent_class == "decision_router_agent"
    assert by["S3"].name == "pmi_ddn_icmp_decision_router"
    # both branches write the same outcome key, as in the hand-built pipeline
    assert by["S4"].output_key == by["S5"].output_key == "pmi_ddn_icmp_review"
    assert by["S7"].required_keys == ["pmi_ddn_msp_loan_data", "pmi_ddn_icmp_review"]
    top = [c.name for c in plan.root.children]
    assert top == ["pmi_ddn_pre_process", "pmi_ddn_icmp_decision_router",
                   "pmi_ddn_approval_gate", "pmi_ddn_post_process"]


# ---------------------------------------------------------------- catalog

def test_catalog_collect_kinds_and_ids():
    docs = collect_catalog(FIX / "catalog_src")
    kinds = {d.kind for d in docs}
    assert kinds == {"schema", "metadata", "tool", "reference"}
    assert all(len(d.id) <= 16 for d in docs)  # Corpus2Skill truncates IDs at 16 chars
    assert len({d.id for d in docs}) == len(docs)
    names = {d.name for d in docs if d.kind == "metadata"}
    assert {"MSP_LOAN_MASTER7_CS", "MSP_PMI_HISTORY"} <= names
    schema = {d.name for d in docs if d.kind == "schema"}
    assert {"database_agent", "LlmAgent", "decision_router_agent", "agent_gate",
            "slv_transformation_agent", "transformation_agent"} <= schema
    # a heading naming transformation_agent must not be filed under slv_transformation_agent
    slv = next(d for d in docs if d.kind == "schema" and d.name == "slv_transformation_agent")
    assert "Same fields as slv_transformation_agent" not in slv.text


def test_catalog_ignores_existing_agents_folder(tmp_path, capsys):
    shutil.copytree(FIX / "catalog_src", tmp_path / "src")
    shutil.copytree(FIX / "gold", tmp_path / "src" / "agents")
    docs = collect_catalog(tmp_path / "src")
    assert "ignoring" in capsys.readouterr().out
    assert not any("pmi_ddn_icmp_process" in d.text for d in docs)


def test_catalog_requires_agent_syntax(tmp_path):
    shutil.copytree(FIX / "catalog_src", tmp_path / "src")
    shutil.rmtree(tmp_path / "src" / "schema")
    with pytest.raises(ValueError, match="schema"):
        collect_catalog(tmp_path / "src")


def test_native_tree_layout(tmp_path):
    fake = make_fake()
    compile_catalog(FIX / "catalog_src", tmp_path, fake, make_cfg("native"))
    skills = tmp_path / ".claude" / "skills"
    tops = sorted(p.name for p in skills.iterdir())
    assert tops == ["agent-syntax", "data-metadata", "reference-docs", "tools"]
    assert list((skills / "data-metadata").glob("*/INDEX.md"))  # 3 docs > leaf_max 2: split
    ents = json.loads((tmp_path / "entity_index.json").read_text())
    tops_for_table = {p.split("/")[0] for p in ents["MSP_LOAN_MASTER7_CS"]["skill_paths"]}
    assert tops_for_table == {"data-metadata"}
    # cards are cached by content: a rebuild makes no card calls
    card_calls = sum(c["system"].startswith("You index catalog") for c in fake.calls)
    fake2 = make_fake()
    compile_catalog(FIX / "catalog_src", tmp_path, fake2, make_cfg("native"))
    assert card_calls == len(collect_catalog(FIX / "catalog_src"))
    assert not any(c["system"].startswith("You index catalog") for c in fake2.calls)


def test_native_engine_does_not_import_corpus2skill(tmp_path):
    import sys
    before = {m for m in sys.modules if m.startswith("corpus2skill")}
    compile_catalog(FIX / "catalog_src", tmp_path, make_fake(), make_cfg("native"))
    assert {m for m in sys.modules if m.startswith("corpus2skill")} == before


def test_compile_catalog_builds_tree_and_store(compiled_catalog):
    skills = compiled_catalog / ".claude" / "skills"
    assert list(skills.rglob("SKILL.md"))
    store = json.loads((compiled_catalog / "documents.json").read_text())
    index = json.loads((compiled_catalog / "catalog_index.json").read_text())
    assert set(store) == set(index)


def test_navigator_finds_table_and_returns_docs(compiled_catalog):
    cat = Catalog(compiled_catalog)
    res = navigate(make_fake(), cat, "Pull the loan from MSP", model="m", max_turns=4)
    names = {cat.index[d]["name"] for d in res.doc_ids}
    assert "MSP_LOAN_MASTER7_CS" in names
    assert res.turns == 2


def test_navigator_blocks_path_escape(compiled_catalog):
    cat = Catalog(compiled_catalog)
    with pytest.raises(ValueError):
        cat.read("../../catalog_index.json")


# ---------------------------------------------------------------- end to end

GOLD_ONLY_TEXT = "State whether the deletion denial is supported and cite the letter date."


def test_generate_end_to_end_then_compare(compiled_catalog, tmp_path):
    assert GOLD_ONLY_TEXT in (FIX / "gold" / "pmi_ddn_icmp_process.yaml").read_text()
    fake = make_fake()
    out = tmp_path / "pmi_ddn"
    rep = generate(FIX / "procedure_sample.md", out, make_cfg(), fake,
                   catalog_dir=compiled_catalog, workers=2, compare_with=FIX / "gold")

    # existing YAMLs never reached the model: they are read only for the comparison
    for call in fake.calls:
        blob = json.dumps(call)
        assert GOLD_ONLY_TEXT not in blob
        assert "pmi_ddn_no_icmp_path_handler.yaml" not in blob

    # every agent was written with its class's agent-syntax section in the prompt
    ground_calls = [c for c in fake.calls if c["system"].startswith("You write the fields")]
    first_prompts = [c["messages"][0]["content"] for c in ground_calls]
    db_prompt = next(p for p in first_prompts if "(database_agent)" in p)
    assert "O2A agent syntax for database_agent:" in db_prompt
    assert "default_db_yaml` (string, required)" in db_prompt

    assert rep["errors"] == 0, rep["findings"]
    assert not [f for f in rep["findings"] if f["code"] == "unknown-field"], rep["findings"]
    assert all(c["agent"] for c in rep["coverage"])  # every procedure step became an agent

    db = yaml.safe_load((out / "pmi_ddn_fetch_msp_loan_data.yaml").read_text())
    block = yaml.safe_load(db["default_db_yaml"])  # embedded YAML string, as the syntax requires
    assert block["connection"]["url"] == "${ENV:ECRM_DB_CONN}"
    assert ":loan_number" in block["query"] and db["input_keys"] == ["loan_number"]

    verdict = yaml.safe_load((out / "pmi_ddn_verdict_synthesizer.yaml").read_text())
    assert verdict["model"] == "model-llm_agent"
    assert "pmi_ddn_borrower_ssn" not in verdict["instruction"]  # bad first answer was rejected
    assert set(verdict["input_keys"]) == {"pmi_ddn_msp_loan_data", "pmi_ddn_icmp_review"}

    gate = yaml.safe_load((out / "pmi_ddn_approval_gate.yaml").read_text())
    assert gate["resume_event"] == "supervisor_approval"  # required by the agent syntax

    router = yaml.safe_load((out / "pmi_ddn_icmp_decision_router.yaml").read_text())
    assert [r["target_agent"] for r in router["routes"]] == \
        [s["name"] for s in router["sub_agents"]]

    cmp = rep["comparison"]
    assert cmp["class_agreement"] == 1.0
    assert cmp["db_table_agreement"] == 1.0
    assert cmp["coverage"] == 1.0 and not cmp["extra_generated_agents"]
    assert json.loads((tmp_path / "pmi_ddn_report.json").read_text())["comparison"] == cmp
    assert (tmp_path / "pmi_ddn_steps.json").exists()


def test_generate_without_existing_yamls_has_no_comparison(compiled_catalog, tmp_path):
    rep = generate(FIX / "procedure_sample.md", tmp_path / "p", make_cfg(), make_fake(),
                   catalog_dir=compiled_catalog)
    assert "comparison" not in rep and rep["errors"] == 0


def test_generate_refuses_catalog_with_existing_agents(compiled_catalog, tmp_path):
    shutil.copytree(compiled_catalog, tmp_path / "cat")
    idx_path = tmp_path / "cat" / "catalog_index.json"
    idx = json.loads(idx_path.read_text())
    idx["a0000000000000"] = {"id": "a0000000000000", "kind": "agent", "name": "old", "source": "x"}
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(ValueError, match="existing agent"):
        generate(FIX / "procedure_sample.md", tmp_path / "out", make_cfg(), make_fake(),
                 catalog_dir=tmp_path / "cat")


def test_generate_refuses_compare_dir_as_output(tmp_path):
    shutil.copytree(FIX / "gold", tmp_path / "gold")
    with pytest.raises(ValueError, match="output directory"):
        generate(FIX / "procedure_sample.md", tmp_path / "gold", make_cfg(), make_fake(),
                 overwrite=True, compare_with=tmp_path / "gold")
    assert len(list((tmp_path / "gold").glob("*.yaml"))) == 10  # nothing deleted


def test_validator_flags_fields_outside_agent_syntax(tmp_path):
    shutil.copytree(FIX / "gold", tmp_path / "p")
    gate = tmp_path / "p" / "pmi_ddn_approval_gate.yaml"
    gate.write_text(gate.read_text() + "retry_forever: true\n")
    syntax = {"agent_gate": "resume_event, timeout_hours, output_key"}
    codes = [(f.code, f.agent) for f in validate_dir(tmp_path / "p", ["loan_number"], syntax)]
    assert ("unknown-field", "pmi_ddn_approval_gate") in codes


def test_generate_refuses_to_clobber(tmp_path):
    out = tmp_path / "x"
    out.mkdir()
    (out / "old.yaml").write_text("name: old\n")
    with pytest.raises(FileExistsError):
        generate(FIX / "procedure_sample.md", out, make_cfg(), make_fake())


# ---------------------------------------------------------------- procedure tree

def test_procedure_sections_follow_headings_in_order():
    from o2a_gen.proctree import parse_sections, section_docs
    text = (FIX / "procedure_sample.md").read_text()
    root = parse_sections(text, "sample")
    titles = [s.title for s in root.walk()][1:]
    assert titles == ["Preamble", "1. Pre-process", "2. ICMP check", "3. Approval", "4. Post-process"]
    pre = root.children[1]
    lines = text.splitlines()
    assert lines[pre.start - 1] == "## 1. Pre-process"
    assert lines[pre.own_start - 1].startswith("1. Pull the loan from MSP")
    assert all(d.id.startswith("p") for d in section_docs(root))


def test_procedure_tree_links_sections_to_catalog(tmp_path):
    from o2a_gen.proctree import build_procedure_tree
    cat_dir = tmp_path / "cat"
    compile_catalog(FIX / "catalog_src", cat_dir, make_fake(), make_cfg("native"))
    cat = Catalog(cat_dir)
    cfg = make_cfg()
    tree = build_procedure_tree(make_fake(), (FIX / "procedure_sample.md").read_text(), "sample",
                                tmp_path / "proc", models=cfg.models, embedding=cfg.embedding,
                                catalog_root=cat_dir, catalog_index=cat.index)
    skills = tmp_path / "proc" / ".claude" / "skills" / "procedure"
    assert sorted(p.name for p in skills.iterdir() if p.is_dir()) == [
        "01-preamble", "02-1-pre-process", "03-2-icmp-check", "04-3-approval", "05-4-post-process"]
    pre = tree.doc_for_line(8)
    assert pre.section.title == "1. Pre-process"
    # embeddings link the lookup section to the table it needs, never to agent syntax
    top3 = [cat.index[c]["name"] for c, _ in tree.hints[pre.id][:3]]
    assert "MSP_LOAN_MASTER7_CS" in top3
    assert all(cat.index[c]["kind"] != "schema" for c, _ in tree.hints[pre.id])
    assert tree.related[pre.id] and all(r != pre.id for r, _ in tree.related[pre.id])
    index_md = (skills / "02-1-pre-process" / "INDEX.md").read_text()
    assert "## Likely tools & data" in index_md and "MSP_LOAN_MASTER7_CS" in index_md
    ctx = tree.context_for_line(8, cat.index)
    assert "procedure/02-1-pre-process" in ctx and "MSP_LOAN_MASTER7_CS" in ctx


def test_saved_vectors_ignored_for_a_different_embedding_model(tmp_path):
    from o2a_gen.skilltree import load_vectors
    compile_catalog(FIX / "catalog_src", tmp_path, make_fake(), make_cfg("native"))
    assert load_vectors(tmp_path, {"provider": "llm", "model": "fake-embed"}) is not None
    assert load_vectors(tmp_path, {"provider": "llm", "model": "other-model"}) is None


def test_generate_grounds_each_step_in_its_procedure_section(compiled_catalog, tmp_path):
    fake = make_fake()
    rep = generate(FIX / "procedure_sample.md", tmp_path / "p", make_cfg(), fake,
                   catalog_dir=compiled_catalog)
    assert rep["errors"] == 0
    assert rep["procedure_tree"]["sections"] == 5
    assert all(g["procedure_section"].startswith("procedure/") for g in rep["grounding"])
    nav_first = [c["messages"][0]["content"] for c in fake.calls
                 if c["system"].startswith("You are looking up")]
    assert all("procedure/" in m for m in nav_first)          # procedure folder is browsable
    extract = next(c for c in fake.calls if c["system"].startswith("You convert an operations"))
    assert "Section outline" in extract["messages"][0]["content"]
    db = next(c["messages"][0]["content"] for c in fake.calls
              if c["system"].startswith("You write the fields") and "(database_agent)" in
              c["messages"][0]["content"])
    assert "This step is in procedure section `procedure/02-1-pre-process`" in db


def test_generate_can_skip_procedure_compile(compiled_catalog, tmp_path):
    rep = generate(FIX / "procedure_sample.md", tmp_path / "p",
                   make_cfg(compile_procedure=False), make_fake(), catalog_dir=compiled_catalog)
    assert rep["procedure_tree"] is None and rep["errors"] == 0


def test_extract_rejects_citations_outside_the_procedure():
    from o2a_gen.procedure import extract_procedure
    bad = json.loads(json.dumps(PROCEDURE_JSON))
    bad["phases"][3]["steps"][0]["source_lines"] = "22"   # the sample has 20 lines
    replies = iter([bad, PROCEDURE_JSON])
    fake = FakeLLM(default=lambda s, m: next(replies))
    proc = extract_procedure(fake, (FIX / "procedure_sample.md").read_text(), model="m")
    assert proc.all_steps()[-1].source_lines == "20"
    assert "between 1 and 20" in fake.calls[1]["messages"][-1]["content"]


# ---------------------------------------------------------------- embedding options (Nemotron-style)

def test_local_embedder_loads_once_with_remote_code_and_shortens(monkeypatch):
    import sys
    import types

    import numpy as np

    from o2a_gen import skilltree
    loads, encoded = [], []

    class FakeST:
        def __init__(self, name, **kwargs):
            loads.append((name, kwargs))

        def encode(self, texts, batch_size):
            encoded.extend(texts)
            return np.ones((len(texts), 4096), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setattr(skilltree, "_MODELS", {})
    emb = {"provider": "local", "model": "nvidia/Nemotron-3-Embed-8B-BF16",
           "trust_remote_code": True, "dtype": "bfloat16", "dimensions": 1024,
           "query_template": "Instruct: {instruction}\nQuery: {text}",
           "query_instruction": "Find catalog entries this procedure step needs"}
    docs = skilltree.embed_texts(None, ["table A", "tool B"], emb)
    qs = skilltree.embed_texts(None, ["pull loan data"], emb, as_query=True)
    assert len(loads) == 1                                  # 16 GB model loaded once
    assert loads[0][1]["trust_remote_code"] is True
    assert loads[0][1]["model_kwargs"] == {"torch_dtype": "bfloat16"}
    assert docs.shape == (2, 1024) and qs.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert encoded[:2] == ["table A", "tool B"]             # documents: no prefix
    assert encoded[2].startswith("Instruct: Find catalog entries")


def test_query_template_used_only_for_section_to_catalog_search(tmp_path):
    from o2a_gen.proctree import build_procedure_tree
    tmpl = {"query_template": "Instruct: find inputs\nQuery: {text}"}
    cfg = make_cfg()
    cfg.embedding = {**cfg.embedding, **tmpl}
    compile_catalog(FIX / "catalog_src", tmp_path / "cat", make_fake(), cfg)
    fake = make_fake()
    tree = build_procedure_tree(fake, (FIX / "procedure_sample.md").read_text(), "s", tmp_path / "p",
                                models=cfg.models, embedding=cfg.embedding,
                                catalog_root=tmp_path / "cat",
                                catalog_index=Catalog(tmp_path / "cat").index)
    doc_batch, query_batch = fake.embed_calls
    assert not any(t.startswith("Instruct:") for t in doc_batch)    # related sections: symmetric
    assert all(t.startswith("Instruct: find inputs") for t in query_batch)
    assert any(tree.hints.values())


def test_saved_vectors_ignored_when_dimensions_change(tmp_path):
    from o2a_gen.skilltree import load_vectors
    compile_catalog(FIX / "catalog_src", tmp_path, make_fake(), make_cfg("native"))
    base = {"provider": "llm", "model": "fake-embed"}
    assert load_vectors(tmp_path, base) is not None
    assert load_vectors(tmp_path, {**base, "dimensions": 1024}) is None
