"""Toy evaluator: diffs declared pipeline intent against what a trace shows
actually happened, and scores the result.

This is a small, readable version of the pattern described at production
scale in this repo's O2A_EVALUATOR_SPEC.md - see course/07-reliability-
evaluation-observability.md for the concepts. Check ids below (O-S3, R-R1,
D1, G-R1, N5) intentionally match that spec's naming convention for the same
categories of check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_framework import Node
from tracing import Tracer

SEVERITY_PENALTY = {"INFO": 0.0, "SEV-3": 0.05, "SEV-2": 0.15, "SEV-1": 0.40}

# A gate that resolves faster than this for a case requiring real review
# isn't actually gating anything - see course/08-safety-and-guardrails.md.
GATE_MIN_REVIEW_S = 0.005

ROUTER_BRANCHES = {"refund_flow", "technical_flow", "general_flow"}


@dataclass
class Flaw:
    check_id: str
    severity: str
    node: str
    description: str


@dataclass
class Scorecard:
    overall: float
    flaws: list[Flaw] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(f.severity == "SEV-1" for f in self.flaws):
            return "FAIL"
        if any(f.severity in ("SEV-2", "SEV-3") for f in self.flaws):
            return "FLAGGED"
        return "PASS"


def evaluate(root: Node, session: dict, tracer: Tracer) -> Scorecard:
    flaws: list[Flaw] = []
    executed = tracer.executed_agents()

    # O-S3 analog: the pipeline's declared output was never produced.
    if root.output_key and root.output_key not in session:
        flaws.append(Flaw(
            "O-S3", "SEV-1", root.name,
            f"pipeline output_key '{root.output_key}' was never written",
        ))

    # R-R1 analog: exactly one router branch should have executed.
    ran_branches = ROUTER_BRANCHES & set(executed)
    if len(ran_branches) == 0:
        flaws.append(Flaw("R-R1", "SEV-1", "intent_router", "no branch executed"))
    elif len(ran_branches) > 1:
        flaws.append(Flaw(
            "R-R1", "SEV-2", "intent_router",
            f"more than one branch executed: {sorted(ran_branches)}",
        ))

    # D1 analog: the same (non-control-flow) agent ran more than once.
    counts: dict[str, int] = {}
    classes: dict[str, str] = {}
    for span in tracer.spans:
        counts[span.agent_name] = counts.get(span.agent_name, 0) + 1
        classes[span.agent_name] = span.agent_class
    for name, count in counts.items():
        if count > 1 and classes[name] not in {"sequential_agent", "router_agent", "orchestrator"}:
            flaws.append(Flaw("D1", "SEV-3", name, f"{name} ran {count} times in one trace"))

    # G-R1 analog: an approved gate on a refund big enough to require real
    # review resolved implausibly fast.
    if (
        session.get("refund_approved")
        and session.get("refund_amount", 0) > 100
        and session.get("refund_gate_wait_s", 1) < GATE_MIN_REVIEW_S
    ):
        flaws.append(Flaw(
            "G-R1", "SEV-3", "refund_gate",
            "gate for a >$100 refund resolved instantly - it isn't really gating",
        ))

    # N5-ish content completeness check: a refund reply should name the
    # order it refunds. Not a real judge call (see MockLLM) - a stand-in
    # for the LLM-as-judge pattern in course/07.
    if session.get("intent") == "refund" and "final_reply" in session:
        order_id = (session.get("order") or {}).get("order_id", "")
        if order_id and order_id not in session["final_reply"]:
            flaws.append(Flaw(
                "N5", "SEV-2", "draft_refund_reply",
                "refund reply does not reference the order it refunds",
            ))

    penalty = sum(SEVERITY_PENALTY[f.severity] for f in flaws)
    overall = max(0.0, 1.0 - penalty)
    return Scorecard(overall=overall, flaws=flaws)
