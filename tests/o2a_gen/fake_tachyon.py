"""A scripted stand-in for Tachyon that answers each o2a_gen prompt type.

It returns what a well-behaved model would, so the tests exercise the real
parsing, validation, retry, navigation, YAML and Corpus2Skill code paths.
"""

from __future__ import annotations

import json
import re

from o2a_gen.llm import FakeLLM
from o2a_gen.navigator import NAV_SYSTEM
from o2a_gen.procedure import EXTRACT_SYSTEM
from o2a_gen.proctree import SECTION_SYSTEM
from o2a_gen.skilltree import CARD_SYSTEM, FOLDER_SYSTEM

PROCEDURE_JSON = {
    "name": "pmi_ddn_review",
    "description": "Decide whether a PMI deletion denial is supported.",
    "inputs": ["loan_number"],
    "orchestrator_class": "resumable_orchestrator",
    "group_class": "SequentialAgent",
    "phases": [
        {"id": "P1", "title": "Pre-process", "steps": [
            {"id": "S1", "title": "Fetch MSP loan data", "agent_class": "database_agent", "source_lines": "8",
             "text": "Pull the loan from MSP: investor class code and the ICMP letter effective date.",
             "produces": "msp loan data", "uses": ["loan_number"]},
            {"id": "S2", "title": "ICMP required flag", "agent_class": "slv_transformation_agent", "source_lines": "9",
             "text": "Y if the letter effective date is present, otherwise N.",
             "produces": "icmp required flag", "uses": ["S1"]},
        ]},
        {"id": "P2", "title": "ICMP check", "steps": [
            {"id": "S3", "title": "ICMP decision", "agent_class": "decision_router_agent", "source_lines": "12-14",
             "text": "Branch on whether an ICMP letter applies.", "depends_on": "S2", "uses": ["S2"],
             "branches": [
                 {"label": "icmp process", "when": "flag is Y", "steps": [
                     {"id": "S4", "title": "ICMP process", "agent_class": "LlmAgent", "source_lines": "12-13",
                      "text": "Review the ICMP letter against investor guidelines.",
                      "produces": "icmp review", "uses": ["S1"]}]},
                 {"label": "no icmp path handler", "when": "flag is N", "steps": [
                     {"id": "S5", "title": "No ICMP path handler", "agent_class": "slv_transformation_agent",
                      "source_lines": "14", "text": "Record that no ICMP review is needed.",
                      "produces": "icmp review", "uses": []}]},
             ]},
        ]},
        {"id": "P3", "title": "Approval", "direct": True, "steps": [
            {"id": "S6", "title": "Approval gate", "agent_class": "agent_gate", "source_lines": "17",
             "text": "Send the case to a supervisor for approval.", "produces": "approval status"}]},
        {"id": "P4", "title": "Post-process", "steps": [
            {"id": "S7", "title": "Verdict synthesizer", "agent_class": "LlmAgent", "source_lines": "20",
             "text": "Write the denial verdict summary citing the loan data and ICMP review.",
             "produces": "verdict", "uses": ["S1", "S4"]}]},
    ],
}

_AGENT = re.compile(r"Agent to write: `(\w+)` \((\w+)\)")
_OUT = re.compile(r"output_key \(set by the plan\): (\S+)")
_BRANCH = re.compile(r"^- (.+?): when .+? -> (\w+)$", re.M)


def _nav(system, messages):
    if len(messages) == 1:
        q = messages[0]["content"]
        term = "MSP_LOAN_MASTER7_CS" if "Pull the loan" in q else "pmi_ddn"
        return {"action": "find", "term": term}
    ids = re.findall(r"doc `(\w+)`", messages[-1]["content"])
    return {"action": "done", "doc_ids": ids[:2], "notes": "found via find"}


_MODEL = re.compile(r'LLM model name for any model field: "([^"]*)"')


def _ground(system, messages):
    """Answers as a model would from the agent syntax section in the prompt."""
    prompt = messages[0]["content"]
    name, cls = _AGENT.search(prompt).groups()
    out_key = _OUT.search(prompt).group(1)
    retrying = len(messages) > 1
    if cls == "database_agent":
        block = ("type: teradata\nconnection:\n  url: ${ENV:ECRM_DB_CONN}\nquery: |\n"
                 "  SELECT M7.LN_NO, TRIM(M7.INV_CLASS_CODE) AS inv_class_code,\n"
                 "         M7.LETTER_EFFECTIVE_DATE AS letter_effective_date\n"
                 "  FROM MSP_LOAN_MASTER7_CS M7\n"
                 "  WHERE M7.LN_NO = LPAD(TRIM(CAST(:loan_number AS VARCHAR(10))), 10, '0')\n")
        return {"fields": {"input_keys": ["loan_number"], "default_db_yaml": block},
                "description": "Fetch MSP loan fields.", "gaps": []}
    if cls == "slv_transformation_agent":
        if "flag" in out_key:
            expr = {"$cond": {"if": {"left": "{{ pmi_ddn_msp_loan_data[0].letter_effective_date }}",
                                     "op": "not_null"}, "then": "Y", "else": "N"}}
            return {"fields": {"transform": {out_key: expr}, "input_keys": ["pmi_ddn_msp_loan_data"],
                               "strict": False}, "description": "Y when an ICMP letter exists."}
        return {"fields": {"transform": {out_key: "NO_ICMP_REVIEW_REQUIRED"}, "input_keys": []},
                "description": "Record that no ICMP review is needed."}
    if cls == "LlmAgent":
        model = _MODEL.search(prompt).group(1)
        if "verdict" in name and not retrying:
            # First answer references a key that does not exist; the check must reject it.
            return {"fields": {"instruction": "Write the verdict using {pmi_ddn_borrower_ssn}.",
                               "input_keys": [], "model": model}}
        keys = ["pmi_ddn_msp_loan_data"] + (["pmi_ddn_icmp_review"] if "verdict" in name else [])
        instr = "Review the loan data {pmi_ddn_msp_loan_data}" + (
            " and the ICMP review {pmi_ddn_icmp_review}" if "verdict" in name else "")
        return {"fields": {"instruction": instr + ". Cite the letter effective date.",
                           "input_keys": keys, "model": model},
                "description": "Review step."}
    if cls == "decision_router_agent":
        routes = []
        for i, (label, target) in enumerate(_BRANCH.findall(prompt)):
            routes.append({"conditions": [{"context_key": "pmi_ddn_icmp_required_flag",
                                           "operator": "eq", "value": "Y" if i == 0 else "N"}],
                           "target_agent": target, "priority": 10 * (i + 1)})
        return {"fields": {"input_keys": ["pmi_ddn_icmp_required_flag"], "routes": routes},
                "description": "Route on the ICMP flag."}
    if cls == "agent_gate":
        # fill required fields only if the agent syntax passed in the prompt defines them
        fields = {"resume_event": "supervisor_approval"} if "resume_event" in prompt else {}
        return {"fields": fields, "description": "Wait for supervisor approval."}
    return {"fields": {}, "description": f"Runs its sub-agents ({cls})."}


def _c2s(system, messages):
    """Corpus2Skill compile-time prompts (cards, summaries, labels, repartition, entities)."""
    text = messages[-1]["content"]
    if "filesystem-safe label" in text:
        return "msp-loan-catalog"
    return json.dumps({"title": "Catalog item", "one_line": "An MSP catalog item.",
                       "phrases": ["MSP"], "changes": [], "named_entities": ["MSP_LOAN_MASTER7_CS"],
                       "doc_types": ["table"]})


def _card(system, messages):
    text = messages[-1]["content"]
    name = re.search(r"\[\w+\] (.+)", text).group(1).strip()
    caps = re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text)
    return {"title": name, "one_line": f"Catalog entry for {name}.",
            "keywords": list(dict.fromkeys([name, *caps]))[:8]}


def _folder(system, messages):
    first = re.search(r"- (?:sub-folder ')?([^:']+)", messages[-1]["content"]).group(1)
    return {"label": f"{first} group", "summary": f"Items related to {first}."}


def make_fake() -> FakeLLM:
    return FakeLLM(rules=[
        (lambda s, u: s == CARD_SYSTEM, _card),
        (lambda s, u: s == SECTION_SYSTEM,
         lambda s, m: {"summary": "Section: " + m[-1]["content"].split("\n", 1)[0][9:]}),
        (lambda s, u: s == FOLDER_SYSTEM, _folder),
        (lambda s, u: s == EXTRACT_SYSTEM, PROCEDURE_JSON),
        (lambda s, u: s == NAV_SYSTEM, _nav),
        (lambda s, u: s.startswith("You write the fields of one agent"), _ground),
    ], default=_c2s)
