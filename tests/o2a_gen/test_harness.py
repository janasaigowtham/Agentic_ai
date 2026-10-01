"""Playbook harness: stages, the code checks between them, and the review/fix loop."""

import json
import re
import shutil

import pytest
import yaml

from o2a_gen.catalog import compile_catalog
from o2a_gen.generate import generate
from o2a_gen.harness import (available_keys, check_datamap, check_flow, check_plan,
                             check_recipe)
from o2a_gen.llm import FakeLLM

from .test_o2a_gen import FIX, make_cfg

PLAYBOOK = """# Workflow playbook (test fixture)

## Roles
- qa_pipeline: resumable_orchestrator, top level.
- pre_process: SequentialAgent: data_fetcher then qa_q<NN>.
- data_fetcher: database_agent, reads the system of record.
- qa_q<NN>: LlmAgent, one evaluator per attribute.
- verdict_synthesizer: LlmAgent, writes the verdict.

## Logic
Each attribute's If/Then rules go into its qa_q<NN> instruction.

## Data mapping
Use the metadata column names exactly. Missing values are NEEDS_HUMAN.
"""

FLOW = {"attributes": [{
    "id": "Q1", "number": "01", "question": "Is the deletion denial supported?",
    "scope": "all loans", "source_lines": "7-20", "outcomes": ["SUPPORTED", "NOT_SUPPORTED"],
    "checks": [
        {"id": "1", "title": "Pull loan", "source_lines": "8",
         "reads": [{"what": "letter effective date", "where": "MSP"}],
         "rules": [{"if": "always", "then": "next"}]},
        {"id": "2", "title": "Letter applies", "source_lines": "9",
         "reads": [], "rules": [{"if": "date present", "then": "next"}]},
        {"id": "3", "title": "Review letter", "source_lines": "12-14",
         "reads": [{"what": "ICMP letter", "where": "ICMP"}],
         "rules": [{"if": "supported", "then": "SUPPORTED"}]},
    ]}], "unclear": []}

BAD_DATAMAP = {"fields": [{"need": "letter effective date", "checks": ["Q1:1"],
                           "status": "available", "tool": "MSP_LOAN_MASTER7_CS",
                           "field": "LETTER_EFFECTIVE_DT"}]}          # wrong column name
DATAMAP = {"fields": [
    {"need": "letter effective date", "checks": ["Q1:1"], "status": "available",
     "tool": "MSP_LOAN_MASTER7_CS", "field": "LETTER_EFFECTIVE_DATE"},
    {"need": "ICMP letter", "checks": ["Q1:3"], "status": "missing"}],
    "external": [{"system": "ICMP", "tools": ["fetch_loan_document"], "status": "available"}]}


RECIPE = {
    "roles": [
        {"role": "qa_pipeline", "agent_class": "resumable_orchestrator", "when": "always",
         "count": "one per workflow"},
        {"role": "pre_process", "agent_class": "SequentialAgent", "when": "always",
         "count": "one per workflow"},
        {"role": "data_fetcher", "agent_class": "LlmAgent or database_agent", "when": "always",
         "count": "one per workflow"},
        {"role": "qa_q<NN>", "agent_class": "LlmAgent", "when": "always",
         "count": "one per attribute", "output_key": "<prefix>_answer_q<NN>"},
        {"role": "verdict_synthesizer", "agent_class": "LlmAgent", "when": "always",
         "count": "one per workflow"}],
    "precedence": ["the procedure wins on business rules"], "shared_keys": [],
    "orchestration": ["pre_process then verdict"], "logic_rules": ["one qa_q per attribute"],
    "data_rules": ["exact column names"], "naming_rules": [], "checks": ["one root"]}


def _agent(name, cls, **kw):
    return {"name": name, "agent_class": cls,
            "role": name.replace("pmi_ddn_", "").replace("qa_q01", "qa_q<NN>"),
            "purpose": f"{name} does its job", "input_keys": [], "output_key": "",
            "sub_agents": [], "routes": [], "covers": [], "sources": [], **kw}


PLAN = {"name": "pmi_ddn", "pipeline_inputs": ["loan_number"], "root": "pmi_ddn_qa_pipeline",
        "agents": [
            _agent("pmi_ddn_qa_pipeline", "resumable_orchestrator",
                   output_key="pmi_ddn_final_output",
                   sub_agents=["pmi_ddn_pre_process", "pmi_ddn_verdict_synthesizer"]),
            _agent("pmi_ddn_pre_process", "SequentialAgent",
                   sub_agents=["pmi_ddn_data_fetcher", "pmi_ddn_qa_q01"]),
            _agent("pmi_ddn_data_fetcher", "database_agent", input_keys=["loan_number"],
                   output_key="pmi_ddn_data", covers=["Q1:1"], sources=["MSP_LOAN_MASTER7_CS"]),
            _agent("pmi_ddn_qa_q01", "LlmAgent", input_keys=["pmi_ddn_data"],
                   output_key="pmi_ddn_answer_q01", covers=["Q1:2", "Q1:3"],
                   procedure_lines="8-14"),
            _agent("pmi_ddn_verdict_synthesizer", "LlmAgent", input_keys=["pmi_ddn_answer_q01"],
                   output_key="pmi_ddn_verdict_json"),
        ]}
UNCOVERED_PLAN = {**PLAN, "agents": [dict(a, covers=[]) if a["name"] == "pmi_ddn_qa_q01" else a
                                     for a in PLAN["agents"]]}


def _writer(system, messages):
    prompt = messages[0]["content"]
    agent = json.loads(re.search(r"Agent \(from the plan\):\n(\{.*?\n\})\n", prompt, re.S).group(1))
    cls = agent["agent_class"]
    if cls == "database_agent":
        db = ("type: teradata\nconnection:\n  url: ${ENV:ECRM_DB_CONN}\nquery: SELECT "
              "LETTER_EFFECTIVE_DATE FROM MSP_LOAN_MASTER7_CS WHERE LN_NO = :loan_number\n")
        return {"fields": {"default_db_yaml": db}, "description": "Pull the loan.", "gaps": []}
    if cls == "LlmAgent":
        key = agent["input_keys"][0]
        fixed = "previous version" in prompt
        return {"fields": {"instruction": f"Apply the rules to {{{key}}}."
                           + (" Answer NEEDS_HUMAN when the ICMP letter is missing." if fixed else ""),
                           "model": ""},
                "gaps": ["model: no model name given\n- line two of the note\nnot: a key"],
                "input_gaps": ["the syntax does not name $fn arguments"]}
    return {"fields": {}, "gaps": []}


class Script:
    """Scripted stage answers; the first datamap and plan answers are wrong on purpose."""

    def __init__(self):
        self.datamap = [BAD_DATAMAP, DATAMAP]
        self.plan = [UNCOVERED_PLAN, PLAN]
        self.reviews = 0
        self.prompts: dict[str, list[str]] = {}

    def __call__(self, system, messages):
        m = re.search(r"TASK: (\w+)", messages[0]["content"])
        if not m:                      # compile-time summaries of the procedure tree
            return {"summary": "section"}
        task = m.group(1)
        self.prompts.setdefault(task, []).append(messages[-1]["content"])
        assert "PLAYBOOK" in system and "qa_q<NN>" in system      # playbook in every call
        if task == "recipe":
            return RECIPE
        if task == "logic":
            return FLOW
        if task == "datamap":
            return self.datamap.pop(0) if len(self.datamap) > 1 else self.datamap[0]
        if task == "orchestrate":
            return self.plan.pop(0) if len(self.plan) > 1 else self.plan[0]
        if task == "write":
            return _writer(system, messages)
        if task == "review":
            self.reviews += 1
            if self.reviews == 1:
                return {"findings": [{"agent": "pmi_ddn_qa_q01", "severity": "blocking",
                                      "level": "agent",
                                      "problem": "missing ICMP letter is not handled",
                                      "evidence": "playbook: Missing values are NEEDS_HUMAN",
                                      "fix": "answer NEEDS_HUMAN when the ICMP letter is missing"},
                                     {"agent": "pmi_ddn_qa_pipeline", "severity": "blocking",
                                      "level": "plan", "problem": "verdict runs too early",
                                      "evidence": "playbook order", "fix": "keep order"}]}
            return {"findings": [{"agent": "*", "severity": "minor", "problem": "names are long",
                                  "evidence": "", "fix": ""}]}
        raise AssertionError(task)


@pytest.fixture
def playbook_catalog(tmp_path):
    src = tmp_path / "catalog_src"
    shutil.copytree(FIX / "catalog_src", src)
    (src / "playbook").mkdir()
    (src / "playbook" / "playbook.md").write_text(PLAYBOOK)
    out = tmp_path / "catalog"
    compile_catalog(src, out, FakeLLM(), make_cfg())
    return out


def test_playbook_run_builds_checks_reviews_and_fixes(playbook_catalog, tmp_path):
    script = Script()
    client = FakeLLM(default=script)
    out = tmp_path / "out" / "workflow"
    rep = generate(FIX / "procedure_sample.md", out, make_cfg(), client,
                   catalog_dir=playbook_catalog)

    assert rep["mode"] == "playbook" and rep["agents"] == 5 and rep["pipeline"] == "pmi_ddn_qa_pipeline"
    assert sorted(p.stem for p in out.glob("*.yaml")) == sorted(a["name"] for a in PLAN["agents"])
    # wrong answers were sent back with the code check's reason, then corrected
    assert len(script.prompts["datamap"]) == 2 and "LETTER_EFFECTIVE_DT" in script.prompts["datamap"][1]
    assert "Q1:2" in script.prompts["orchestrate"][1]          # retry after the coverage check
    # the blocking review finding was sent to that agent's writer, and the fix landed
    q = yaml.safe_load((out / "pmi_ddn_qa_q01.yaml").read_text())
    assert "NEEDS_HUMAN" in q["instruction"] and q["input_keys"] == ["pmi_ddn_data"]
    assert script.reviews == 2
    text = (out / "pmi_ddn_qa_q01.yaml").read_text()
    assert "# checks: Q1:2, Q1:3" in text and "# TODO: model" in text
    db = yaml.safe_load((out / "pmi_ddn_data_fetcher.yaml").read_text())
    assert "${ENV:ECRM_DB_CONN}" in db["default_db_yaml"]
    root = yaml.safe_load((out / "pmi_ddn_qa_pipeline.yaml").read_text())
    assert [s["name"] for s in root["sub_agents"]] == ["pmi_ddn_pre_process",
                                                      "pmi_ddn_verdict_synthesizer"]
    for note in ("recipe.json", "flow.md", "data_map.md", "plan.md", "review.json",
                 "placeholders.md", "report.json"):
        assert (out.parent / f"workflow_{note}").exists(), note
    assert rep["errors"] == 0 and rep["blocking"] == 0
    # plan-level finding re-ran planning with the previous plan and the problem
    assert len(script.prompts["orchestrate"]) == 3 and rep["replans"] == 1
    assert "Your previous plan" in script.prompts["orchestrate"][2]
    assert "verdict runs too early" in script.prompts["orchestrate"][2]
    # multi-line notes stay comments, so every file parses
    for p in out.glob("*.yaml"):
        yaml.safe_load(p.read_text())
    assert "#   - line two of the note" in text
    # input gaps are reported once, not per agent
    assert rep["input_gaps"] == ["the syntax does not name $fn arguments"]
    notes = (out.parent / "workflow_placeholders.md").read_text()
    assert notes.count("the syntax does not name $fn arguments") == 1
    assert {c["check"]: c["agent"] for c in rep["coverage"]}["Q1:3"] == "pmi_ddn_qa_q01"


def test_without_playbook_the_step_generator_still_runs(tmp_path):
    out = tmp_path / "catalog"
    compile_catalog(FIX / "catalog_src", out, FakeLLM(), make_cfg())
    from o2a_gen.harness import has_playbook
    from o2a_gen.navigator import Catalog
    assert not has_playbook(Catalog(out))


def test_stage_checks_catch_common_mistakes():
    classes = {"resumable_orchestrator", "SequentialAgent", "database_agent", "LlmAgent"}
    with pytest.raises(ValueError, match="not in the syntax"):
        check_recipe({"roles": [{"role": "x", "agent_class": "magic_agent"}]}, classes)
    with pytest.raises(ValueError, match="source_lines"):
        check_flow({"attributes": [{"id": "Q", "source_lines": "900", "checks": [
            {"id": "1", "source_lines": "1", "rules": [{"if": "a", "then": "b"}]}]}]}, 20)
    with pytest.raises(ValueError, match="does not appear"):
        check_datamap(BAD_DATAMAP, FLOW, "MSP_LOAN_MASTER7_CS LETTER_EFFECTIVE_DATE")
    check_datamap(DATAMAP, FLOW, "MSP_LOAN_MASTER7_CS LETTER_EFFECTIVE_DATE fetch_loan_document")
    with pytest.raises(ValueError, match="not covered"):
        check_plan(UNCOVERED_PLAN, classes, FLOW, ["loan_number"])
    bad_key = json.loads(json.dumps(PLAN))
    bad_key["agents"][3]["input_keys"] = ["nobody_writes_this"]
    with pytest.raises(ValueError, match="no agent writes"):
        check_plan(bad_key, classes, FLOW, ["loan_number"])
    check_plan(PLAN, classes, FLOW, ["loan_number"])
    assert available_keys(PLAN, "pmi_ddn_qa_q01", ["loan_number"]) == [
        "loan_number", "pmi_ddn_final_output", "pmi_ddn_data"]


def test_webui_takes_a_playbook_and_shows_its_stages_and_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("O2A_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "k")
    import importlib

    from fastapi.testclient import TestClient

    import o2a_gen.webui.server as server
    server = importlib.reload(server)
    monkeypatch.setattr(server.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda s: None})())
    c = TestClient(server.app)
    files = [("procedure", ("proc.md", b"# P\n", "text/markdown")),
             ("syntax", ("syntax.md", b"## a_agent\n", "text/markdown")),
             ("tools", ("tools.yaml", b"tools: []\n", "application/yaml")),
             ("metadata", ("meta.yaml", b"t: 1\n", "application/yaml")),
             ("playbook", ("playbook.md", PLAYBOOK.encode(), "text/markdown"))]
    jid = c.post("/api/jobs", files=files, data={}).json()["id"]
    jd = tmp_path / "runs" / jid
    assert (jd / "catalog_src" / "playbook" / "playbook.md").exists()
    view = c.get(f"/api/jobs/{jid}").json()
    assert [s["id"] for s in view["stages"]][2:] == ["recipe", "logic", "datamap", "plan",
                                                     "write", "review"]
    (jd / "out").mkdir(parents=True)
    (jd / "out" / "workflow_flow.md").write_text("# Q1")
    assert c.get(f"/api/jobs/{jid}").json()["notes"] == ["flow.md"]
    assert c.get(f"/api/jobs/{jid}/files/flow.md").text == "# Q1"


def test_sources_resolve_from_descriptive_plan_text(playbook_catalog):
    from o2a_gen.harness import _find_sources
    from o2a_gen.navigator import Catalog
    cat = Catalog(playbook_catalog)
    text, ids = _find_sources(cat, ["tools[fetch_loan_document] (POST, auth bearer)",
                                    "metadata MSP_LOAN_MASTER7_CS columns"])
    names = {cat.index[d]["name"] for d in ids}
    assert "fetch_loan_document" in names and "MSP_LOAN_MASTER7_CS" in names
    assert "document_type" in text and "LETTER_EFFECTIVE_DATE" in text
    assert _find_sources(cat, ["nothing like this"]) == ("", [])


def test_runtime_values_are_not_session_keys_and_sql_binds_are():
    from o2a_gen.harness import check_fields
    agent = {"agent_class": "rest_api_agent", "output_key": "out", "input_keys": []}
    check_fields(agent, {"api_config_details": "url: ${ENV:ICMP_URL}\nheaders:\n"
                         "  WF-senderMessageId: ${UUID}\n  ts: ${ISO_TIMESTAMP}\n"},
                 "api_config_details", [])
    db = {"agent_class": "database_agent", "output_key": "rows", "input_keys": []}
    with pytest.raises(ValueError, match="loan_number"):
        check_fields(db, {"default_db_yaml": "type: oracle\nconnection:\n  url: ${ENV:DB}\n"
                          "query: SELECT 1 FROM t WHERE ln = :loan_number\n"},
                     "default_db_yaml", [])


def test_input_keys_follow_the_class_syntax():
    from o2a_gen.harness import _key_fields
    a = {"agent_class": "rest_api_agent", "input_keys": ["body", "batch_id"]}
    keys, note = _key_fields(a, {}, "Optional keys\n- input_key\n")
    assert keys == {"input_key": "body"} and "batch_id" in note
    assert _key_fields(a, {}, "- input_keys: list")[0] == {"input_keys": ["body", "batch_id"]}
    assert _key_fields(a, {"input_key": "x"}, "- input_key")[0] == {}
    assert _key_fields({"agent_class": "agent_gate", "input_keys": []}, {}, "")[0] == {}


def test_plan_check_catches_duplicate_roles_and_writers():
    classes = {"resumable_orchestrator", "SequentialAgent", "database_agent", "LlmAgent",
               "decision_router_agent"}
    dup = json.loads(json.dumps(PLAN))
    dup["agents"][1]["sub_agents"].append("pmi_ddn_qa_q01_again")
    dup["agents"].append(_agent("pmi_ddn_qa_q01_again", "LlmAgent", role="data_fetcher",
                                output_key="pmi_ddn_data", covers=["Q1:2"]))
    with pytest.raises(ValueError) as e:
        check_plan(dup, classes, FLOW, ["loan_number"], RECIPE)
    assert "filled 2 times" in str(e.value) and "both write 'pmi_ddn_data'" in str(e.value)
    # a first-pass and a final evaluator may share the answer key the playbook gives both
    shared = json.loads(json.dumps(PLAN))
    recipe = json.loads(json.dumps(RECIPE))
    recipe["roles"].append({"role": "qa_q<NN>_final", "agent_class": "LlmAgent",
                            "count": "one per attribute", "output_key": "<prefix>_answer_q<NN>"})
    shared["agents"][0]["sub_agents"].insert(1, "pmi_ddn_qa_q01_final")
    shared["agents"].append(_agent("pmi_ddn_qa_q01_final", "LlmAgent", role="qa_q<NN>_final",
                                   input_keys=["pmi_ddn_data"], output_key="pmi_ddn_answer_q01"))
    check_plan(shared, classes, FLOW, ["loan_number"], recipe)
    # a container may report its own child's key; unknown roles are rejected
    box = json.loads(json.dumps(PLAN))
    box["agents"][1]["output_key"] = "pmi_ddn_answer_q01"
    check_plan(box, classes, FLOW, ["loan_number"], RECIPE)
    box["agents"][2]["role"] = "made_up_role"
    with pytest.raises(ValueError, match="not a recipe role"):
        check_plan(box, classes, FLOW, ["loan_number"], RECIPE)
    box["agents"][2]["role"] = "tool: escrow fetch step"
    check_plan(box, classes, FLOW, ["loan_number"], RECIPE)
