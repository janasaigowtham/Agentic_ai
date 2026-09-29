import json
import shutil
from pathlib import Path

import pytest
import yaml

from o2a_gen.catalog import collect_catalog, compile_catalog
from o2a_gen.compare import compare_dirs
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


def make_cfg(**gen) -> GenConfig:
    return GenConfig(
        llm={"provider": "fake"},
        models={r: f"model-{r}" for r in ("extract", "ground", "navigate", "catalog_summary",
                                          "catalog_cards", "llm_agent")},
        embedding={"provider": "llm", "model": "fake-embed"},
        catalog={"p": 3, "max_top": 3, "min_cluster_size": 1},
        generation={"prefix": "pmi_ddn", "pipeline_inputs": ["loan_number"],
                    "default_connection_env": "ECRM_DB_CONN", **gen},
    )


@pytest.fixture(scope="module")
def compiled_catalog(tmp_path_factory):
    out = tmp_path_factory.mktemp("catalog_build")
    compile_catalog(FIX / "catalog_src", out, make_fake(), make_cfg())
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

def test_catalog_collect_ids_fit_corpus2skill():
    docs = collect_catalog(FIX / "catalog_src")
    kinds = {d.kind for d in docs}
    assert kinds == {"agent", "table", "tool", "reference"}
    assert all(len(d.id) <= 16 for d in docs)  # Corpus2Skill truncates IDs at 16 chars
    assert len({d.id for d in docs}) == len(docs)
    names = {d.name for d in docs if d.kind == "table"}
    assert {"MSP_LOAN_MASTER7_CS", "MSP_PMI_HISTORY"} <= names


def test_compile_catalog_runs_corpus2skill_through_our_client(compiled_catalog):
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

def test_generate_end_to_end_matches_gold(compiled_catalog, tmp_path):
    fake = make_fake()
    out = tmp_path / "pmi_ddn"
    rep = generate(FIX / "procedure_sample.md", out, make_cfg(), fake,
                   catalog_dir=compiled_catalog, workers=2)

    assert rep["errors"] == 0, rep["findings"]
    assert all(c["agent"] for c in rep["coverage"])  # every procedure step became an agent

    db = yaml.safe_load((out / "pmi_ddn_fetch_msp_loan_data.yaml").read_text())
    block = yaml.safe_load(db["default_db_yaml"])  # embedded YAML string, like O2A
    assert block["connection"]["url"] == "${ENV:ECRM_DB_CONN}"
    assert ":loan_number" in block["query"] and db["input_keys"] == ["loan_number"]

    verdict = yaml.safe_load((out / "pmi_ddn_verdict_synthesizer.yaml").read_text())
    assert verdict["model"] == "model-llm_agent"
    assert "pmi_ddn_borrower_ssn" not in verdict["instruction"]  # bad first answer was rejected
    assert set(verdict["input_keys"]) == {"pmi_ddn_msp_loan_data", "pmi_ddn_icmp_review"}

    gate = yaml.safe_load((out / "pmi_ddn_approval_gate.yaml").read_text())
    assert gate["resume_event"] == "supervisor_approval"  # runtime field copied via `extra`

    router = yaml.safe_load((out / "pmi_ddn_icmp_decision_router.yaml").read_text())
    assert [r["target_agent"] for r in router["routes"]] == \
        [s["name"] for s in router["sub_agents"]]

    cmp = compare_dirs(FIX / "gold", out)
    assert cmp["class_agreement"] == 1.0
    assert cmp["db_table_agreement"] == 1.0
    assert cmp["coverage"] == 1.0 and not cmp["extra_generated_agents"]

    assert (tmp_path / "pmi_ddn_report.json").exists()
    assert (tmp_path / "pmi_ddn_steps.json").exists()


def test_generate_refuses_to_clobber(tmp_path):
    out = tmp_path / "x"
    out.mkdir()
    (out / "old.yaml").write_text("name: old\n")
    with pytest.raises(FileExistsError):
        generate(FIX / "procedure_sample.md", out, make_cfg(), make_fake())
