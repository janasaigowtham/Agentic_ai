"""Fixed answer key for MockJudgeClient.

Every entry below is a literal, hand-written JSON response. Nothing here
inspects a trajectory's actual content and decides an outcome from it -- that
would be a rule engine wearing a mock's clothes. MockJudgeClient looks a
marker up in this table (or falls back to MOCK_DEFAULT_BY_STAGE) and returns
whatever is there; the marker itself is built by the caller from identifiers
(specialist name, step ids, trajectory id) that are known before any judgment
is made, never from a computed answer.

Entries are keyed to this repo's one demo fixture (fixtures/trace.jsonl,
produced by fixtures/make_trace.py) so mock-mode reproduces the same, real,
seeded findings on every run without needing API keys.
"""
from __future__ import annotations

from trh.judge.markers import specialist_marker, stage_marker

# fixtures/make_trace.py's trace_id -- fixed, not derived from anything.
DEMO_TRACE_ID = "trace-fixture-0000000000000001"

# fixtures/make_trace.py span ids, in build order. Stable across runs: _SEQ
# starts fresh on every import of that module and build_spans() always
# constructs spans in the same order.
_S = {
    "pipeline_root": "s001",
    "pre_process": "s002",
    "fetch_loan_1": "s003",
    "fetch_loan_2": "s004",
    "fetch_investor": "s005",
    "icmp_router": "s010",
    "approval_gate": "s012",
    "qa_q25527": "s014",
    "verdict_synthesizer": "s015",
}

MOCK_DEFAULT_BY_STAGE: dict[str, dict] = {
    "specialist": {"finding": "No issues found for this step.", "severity": "none"},
    "orient": {
        "goal": "Process a PMI DDN loan servicing decision end to end.",
        "orient_notes": [],
    },
    "aggregator": {"recommendations": []},
    "fact_check": {
        "confidence": "high",
        "notes": "Aggregator report is consistent with the trajectory.",
        "routed_back": False,
    },
}

MOCK_ANSWER_KEY: dict[str, dict] = {
    specialist_marker("database", [_S["fetch_loan_2"]]): {
        "finding": (
            "pmi_ddn_fetch_msp_loan_data ran twice with identical input "
            "({'loan_number': '1234567890'}); the second call's result was "
            "never reconciled against the first."
        ),
        "severity": "significant",
    },
    specialist_marker("llm_reasoning", [_S["verdict_synthesizer"]]): {
        "finding": (
            "pmi_ddn_verdict_synthesizer's instruction template reads "
            "{pmi_ddn_msp_loan_data} directly, but the node's declared "
            "input_keys only lists pmi_ddn_answer_q25527 and "
            "pmi_ddn_qa_q25527_result -- the raw loan record (including "
            "borrower SSN) reaches this prompt through an undeclared path."
        ),
        "severity": "critical",
    },
    specialist_marker("gate_escalation", [_S["approval_gate"]]): {
        "finding": (
            "pmi_ddn_approval_gate resolved in ~4ms -- effectively no human "
            "review window before the ICMP-required path proceeded."
        ),
        "severity": "significant",
    },
    specialist_marker("decision_routing", [_S["icmp_router"]]): {
        "finding": (
            "pmi_ddn_icmp_decision_router's routes use only eq conditions on "
            "pmi_ddn_icmp_required_flag ('Y' or 'N'); any other value falls "
            "through with no matching route and no declared default."
        ),
        "severity": "minor",
    },
    specialist_marker("orchestration_plan_fidelity", [_S["pipeline_root"]]): {
        "finding": (
            "Total trajectory duration is 101s against this pipeline's usual "
            "~60s budget, driven by pmi_ddn_icmp_process (30s) and "
            "pmi_ddn_verdict_synthesizer (38s)."
        ),
        "severity": "minor",
    },
    stage_marker("orient", DEMO_TRACE_ID): {
        "goal": (
            "Decide ICMP/investor-review handling for loan 1234567890 and "
            "produce a final PMI DDN servicing verdict."
        ),
        "orient_notes": [
            "pmi_ddn_fetch_msp_loan_data appears twice in sequence.",
            "pmi_ddn_verdict_synthesizer's prompt is unusually long for its "
            "declared input_keys.",
            "pmi_ddn_approval_gate resolves almost instantly.",
        ],
    },
    stage_marker("aggregator", DEMO_TRACE_ID): {
        "recommendations": [
            {
                "tied_to_root_cause": True,
                "root_cause_chain": [_S["fetch_loan_1"], _S["fetch_loan_2"]],
                "description": (
                    "pmi_ddn_fetch_msp_loan_data is invoked twice with "
                    "identical input and no cache/dedup check."
                ),
                "proposed_fix": (
                    "Memoize pmi_ddn_fetch_msp_loan_data by loan_number "
                    "within a single pipeline run, or add a duplicate-call "
                    "guard in pmi_ddn_pre_process."
                ),
            },
            {
                "tied_to_root_cause": True,
                "root_cause_chain": [_S["verdict_synthesizer"]],
                "description": (
                    "pmi_ddn_verdict_synthesizer's instruction reads "
                    "pmi_ddn_msp_loan_data (including borrower SSN) without "
                    "it being declared in input_keys."
                ),
                "proposed_fix": (
                    "Either declare pmi_ddn_msp_loan_data in "
                    "pmi_ddn_verdict_synthesizer's input_keys, or remove the "
                    "raw loan record from its prompt and pass only the "
                    "fields it actually needs."
                ),
            },
            {
                "tied_to_root_cause": True,
                "root_cause_chain": [_S["qa_q25527"], _S["verdict_synthesizer"]],
                "description": (
                    "pmi_ddn_qa_q25527 and pmi_ddn_verdict_synthesizer's "
                    "instructions share an identical 'ABSOLUTE RULE' "
                    "boilerplate block (flagged via matching prompt-sentence "
                    "fingerprints on both steps); a future edit to one is "
                    "likely to silently drift from the other."
                ),
                "proposed_fix": (
                    "Extract the shared 'ABSOLUTE RULE' block into one "
                    "referenced fragment both instructions include, instead "
                    "of two independently-maintained copies."
                ),
            },
        ]
    },
    stage_marker("fact_check", DEMO_TRACE_ID): {
        "confidence": "high",
        "notes": (
            "Both recommendations are directly supported by evidence already "
            "in the trajectory (duplicate input_value on repeated spans; "
            "instruction text vs. declared input_keys diff). No unsupported "
            "claims found."
        ),
        "routed_back": False,
    },
}
