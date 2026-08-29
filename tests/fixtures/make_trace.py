#!/usr/bin/env python3
"""Generates a deliberately-flawed span trace for the pmi_ddn_pipeline fixture.

Usage: python make_trace.py <output.jsonl>

Seeded flaws (see O2A_EVALUATOR_SPEC.md section 12.2):
    D1   - pmi_ddn_fetch_msp_loan_data runs twice with identical input_value
    N3   - pmi_ddn_verdict_synthesizer references pmi_ddn_msp_loan_data but its
           input_keys only declares pmi_ddn_answer_q25527 (+ qa_q25527_result)
    N2   - pmi_ddn_qa_q25527 and pmi_ddn_verdict_synthesizer instructions share
           an "ABSOLUTE RULE" block
    R-S3 - pmi_ddn_icmp_decision_router uses only eq operators
    G-R1 - pmi_ddn_approval_gate span duration is ~4ms
    O-R3 - total trace duration is 101s; tests use a 60s budget
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TRACE_ID = "trace-fixture-0000000000000001"
PIPELINE_NAME = "pmi_ddn_pipeline"

FIXTURES_DIR = Path(__file__).resolve().parent
AGENT_DIR = FIXTURES_DIR / "agents"


def _pipeline_hash() -> str:
    sys.path.insert(0, str(FIXTURES_DIR.parent.parent))
    from o2a_eval.core.graph import load_pipeline

    graph = load_pipeline(AGENT_DIR, PIPELINE_NAME)
    return graph.pipeline_hash


PIPELINE_HASH = _pipeline_hash()

_SEQ = 0


def _next_span_id() -> str:
    global _SEQ
    _SEQ += 1
    return f"s{_SEQ:03d}"


SHARED_RULE_BLOCK = (
    "ABSOLUTE RULE: NEVER fabricate loan data, borrower identity, or investor "
    "figures that are not explicitly present in the structured session fields "
    "provided to you. If a required field is missing, state that it is missing "
    "rather than guessing a plausible value. Always cite the exact session key "
    "you drew each fact from. Do not invent dates, dollar amounts, or "
    "classification codes under any circumstances. Treat every structured field "
    "as the single source of truth for this task."
)

LOAN_NUMBER = "1234567890"
BORROWER_NAME = "Jane Doe"
LOAN_RECORD = [
    {
        "ln_no": LOAN_NUMBER,
        "inv_class_code": "COMM",
        "letter_effective_date": "2024-01-15",
        "borrower_name": BORROWER_NAME,
        "borrower_ssn": "123-45-6789",
    }
]
INVESTOR_RECORD = [{"investor_id": "INV001", "investor_name": "Acme Capital"}]


class SpanBuilder:
    def __init__(self):
        self.spans: list[dict] = []
        self.session: dict = {"loan_number": LOAN_NUMBER}

    def _base_attrs(self, name: str, agent_class: str, output_key: str | None,
                     input_keys: list[str]) -> dict:
        return {
            "o2a.agent.name": name,
            "o2a.agent.class": agent_class,
            "o2a.agent.output_key": output_key or "",
            "o2a.agent.input_keys": json.dumps(input_keys),
            "o2a.pipeline.name": PIPELINE_NAME,
            "o2a.pipeline.hash": PIPELINE_HASH,
        }

    def span(
        self,
        name: str,
        agent_class: str,
        start_s: float,
        end_s: float,
        parent_id: str | None,
        output_key: str | None = None,
        input_keys: list[str] | None = None,
        write_value: object = None,
        input_value: object = None,
        extra_attrs: dict | None = None,
        status: str = "OK",
    ) -> str:
        span_id = _next_span_id()
        keys_before = sorted(self.session.keys())
        attrs = self._base_attrs(name, agent_class, output_key, input_keys or [])
        attrs["o2a.session.keys_before"] = json.dumps(keys_before)

        if input_value is not None:
            attrs["input.value"] = json.dumps(input_value, default=str)

        keys_written = []
        if write_value is not None and output_key:
            is_new = output_key not in self.session
            self.session[output_key] = write_value
            if is_new:
                keys_written.append(output_key)
            attrs["output.value"] = json.dumps(write_value, default=str)

        attrs["o2a.session.keys_after"] = json.dumps(sorted(self.session.keys()))
        attrs["o2a.session.keys_written"] = json.dumps(keys_written)

        if extra_attrs:
            attrs.update(extra_attrs)

        self.spans.append(
            {
                "trace_id": TRACE_ID,
                "span_id": span_id,
                "parent_id": parent_id,
                "name": name,
                "start_ns": int(start_s * 1e9),
                "end_ns": int(end_s * 1e9),
                "status": status,
                "attributes": attrs,
                "events": [],
            }
        )
        return span_id


def build_spans() -> list[dict]:
    b = SpanBuilder()

    root_id = b.span(
        "pmi_ddn_pipeline", "resumable_orchestrator", 0.0, 101.0, None,
        output_key="pmi_ddn_final_output", input_keys=[],
    )

    pre_id = b.span(
        "pmi_ddn_pre_process", "SequentialAgent", 0.0, 1.9, root_id,
    )

    fetch_input = {"loan_number": LOAN_NUMBER}
    b.span(
        "pmi_ddn_fetch_msp_loan_data", "database_agent", 0.0, 0.5, pre_id,
        output_key="pmi_ddn_msp_loan_data", input_keys=["loan_number"],
        write_value=LOAN_RECORD, input_value=fetch_input,
        extra_attrs={
            "o2a.db.query_hash": hashlib.sha256(b"fetch_msp_loan_data").hexdigest()[:16],
            "o2a.db.row_count": 1,
        },
    )
    # D1 seed: same agent, same input_value, ran a second time.
    b.span(
        "pmi_ddn_fetch_msp_loan_data", "database_agent", 0.5, 1.0, pre_id,
        output_key="pmi_ddn_msp_loan_data", input_keys=["loan_number"],
        write_value=LOAN_RECORD, input_value=fetch_input,
        extra_attrs={
            "o2a.db.query_hash": hashlib.sha256(b"fetch_msp_loan_data").hexdigest()[:16],
            "o2a.db.row_count": 1,
        },
    )
    b.span(
        "pmi_ddn_fetch_investor_data", "database_agent", 1.0, 1.5, pre_id,
        output_key="pmi_ddn_investor_data", input_keys=["loan_number"],
        write_value=INVESTOR_RECORD, input_value={"loan_number": LOAN_NUMBER},
        extra_attrs={
            "o2a.db.query_hash": hashlib.sha256(b"fetch_investor_data").hexdigest()[:16],
            "o2a.db.row_count": 1,
        },
    )
    b.span(
        "pmi_ddn_icmp_required_flag", "slv_transformation_agent", 1.5, 1.6, pre_id,
        output_key="pmi_ddn_icmp_required_flag", input_keys=["pmi_ddn_msp_loan_data"],
        write_value="Y",
    )
    b.span(
        "pmi_ddn_investor_flag", "slv_transformation_agent", 1.6, 1.7, pre_id,
        output_key="pmi_ddn_investor_flag", input_keys=["pmi_ddn_investor_data"],
        write_value="Y",
    )
    b.span(
        "pmi_ddn_answer_q25527", "transformation_agent", 1.7, 1.8, pre_id,
        output_key="pmi_ddn_answer_q25527", input_keys=["pmi_ddn_msp_loan_data"],
        write_value="COMM",
    )
    b.span(
        "pmi_ddn_investor_score", "transformation_agent", 1.8, 1.9, pre_id,
        output_key="pmi_ddn_investor_score", input_keys=["pmi_ddn_investor_data"],
        write_value=87,
    )

    router_id = b.span(
        "pmi_ddn_icmp_decision_router", "decision_router_agent", 1.9, 2.0, root_id,
        output_key="routing_decision",
        write_value="pmi_ddn_icmp_process",
        extra_attrs={
            "o2a.router.evaluated_routes": json.dumps(
                [
                    {
                        "target": "pmi_ddn_icmp_process",
                        "priority": 10,
                        "conditions": [
                            {"context_key": "pmi_ddn_icmp_required_flag", "operator": "eq", "value": "Y"}
                        ],
                        "matched": True,
                    },
                    {
                        "target": "pmi_ddn_no_icmp_path_handler",
                        "priority": 20,
                        "conditions": [
                            {"context_key": "pmi_ddn_icmp_required_flag", "operator": "eq", "value": "N"}
                        ],
                        "matched": False,
                    },
                ]
            ),
            "o2a.router.chosen_target": "pmi_ddn_icmp_process",
        },
    )
    b.span(
        "pmi_ddn_icmp_process", "transformation_agent", 2.0, 32.0, router_id,
        output_key="pmi_ddn_icmp_result", input_keys=["pmi_ddn_msp_loan_data"],
        write_value={"status": "required"},
    )

    # G-R1 seed: gate barely pauses (~4ms).
    b.span(
        "pmi_ddn_approval_gate", "agent_gate", 32.0, 32.004, root_id,
        input_keys=["pmi_ddn_icmp_result"],
    )

    post_id = b.span(
        "pmi_ddn_post_process", "SequentialAgent", 32.004, 101.0, root_id,
    )

    qa_prompt = (
        "You are answering structured QA question 25527 for a PMI DDN loan review.\n\n"
        + SHARED_RULE_BLOCK
        + f"\n\nUse the pre-computed answer: COMM\nCross-check against the loan record: {LOAN_RECORD}\n"
    )
    b.span(
        "pmi_ddn_qa_q25527", "LlmAgent", 32.004, 62.004, post_id,
        output_key="pmi_ddn_qa_q25527_result",
        input_keys=["pmi_ddn_answer_q25527", "pmi_ddn_msp_loan_data"],
        write_value={"answer": "Y", "rationale": "matches classification"},
        input_value={"pmi_ddn_answer_q25527": "COMM"},
        extra_attrs={
            "o2a.llm.rendered_prompt": qa_prompt,
            "o2a.llm.session_snapshot": json.dumps(
                {k: v for k, v in b.session.items()}, default=str
            ),
            "o2a.llm.template_references": json.dumps(
                ["pmi_ddn_answer_q25527", "pmi_ddn_msp_loan_data"]
            ),
        },
    )

    # N3 seed: verdict_synthesizer's declared input_keys omit pmi_ddn_msp_loan_data,
    # but its rendered prompt (and template_references) read it anyway.
    verdict_prompt = (
        "You are synthesizing the final PMI DDN servicing verdict for this loan.\n\n"
        + SHARED_RULE_BLOCK
        + f"\n\nPre-computed answer: COMM\nQA result: Y\nRaw loan record: {LOAN_RECORD}\n"
    )
    b.span(
        "pmi_ddn_verdict_synthesizer", "LlmAgent", 62.004, 100.004, post_id,
        output_key="pmi_ddn_final_output",
        input_keys=["pmi_ddn_answer_q25527", "pmi_ddn_qa_q25527_result"],
        write_value={"verdict": "approved", "reason": "classification matches investor rules"},
        input_value={"pmi_ddn_answer_q25527": "COMM", "pmi_ddn_qa_q25527_result": "Y"},
        extra_attrs={
            "o2a.llm.rendered_prompt": verdict_prompt,
            "o2a.llm.session_snapshot": json.dumps(
                {k: v for k, v in b.session.items()}, default=str
            ),
            "o2a.llm.template_references": json.dumps(
                ["pmi_ddn_answer_q25527", "pmi_ddn_qa_q25527_result", "pmi_ddn_msp_loan_data"]
            ),
        },
    )

    review_router_id = b.span(
        "pmi_ddn_investor_review_router", "decision_router_agent", 100.004, 100.5, post_id,
        extra_attrs={
            "o2a.router.evaluated_routes": json.dumps(
                [
                    {
                        "target": "pmi_ddn_investor_review",
                        "priority": 10,
                        "conditions": [
                            {"context_key": "pmi_ddn_investor_flag", "operator": "neq", "value": "SKIP"}
                        ],
                        "matched": True,
                    }
                ]
            ),
            "o2a.router.chosen_target": "pmi_ddn_investor_review",
        },
    )
    b.span(
        "pmi_ddn_investor_review", "transformation_agent", 100.5, 101.0, review_router_id,
        input_keys=["pmi_ddn_investor_flag"],
    )

    return b.spans


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python make_trace.py <output.jsonl>", file=sys.stderr)
        sys.exit(1)
    out_path = Path(sys.argv[1])
    spans = build_spans()
    with out_path.open("w") as f:
        for span in spans:
            f.write(json.dumps(span) + "\n")
    print(f"wrote {len(spans)} spans to {out_path}")


if __name__ == "__main__":
    main()
