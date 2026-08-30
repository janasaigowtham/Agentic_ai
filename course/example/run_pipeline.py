"""CLI entry point for the course's working example.

    python run_pipeline.py

Runs four scenarios through the same declared pipeline.yaml and prints a
trace + scorecard for each - only the input session changes between them,
never the code, which is the point of course/05-multi-agent-orchestration.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import yaml

from agent_framework import Engine, GateRejected, Node
from eval import evaluate
from llm_client import build_llm
from tools import calculate_refund, lookup_order
from tracing import Tracer

TRANSFORMS = {
    "calc_refund": lambda order, message: calculate_refund(order, message),
}
TOOLS = {
    "lookup_order": lookup_order,
}


def approve_gate(node, session) -> bool:
    """The refund-gate approval policy: reject outright above $1000,
    simulate real manual-review latency above $100, auto-approve below
    that. See course/08-safety-and-guardrails.md - a gate needs to be
    capable of actually rejecting, not just recording a rubber stamp."""
    amount = session.get("refund_amount", 0)
    if amount > 1000:
        return False
    if amount > 100:
        time.sleep(0.01)
    return True


def load_root() -> Node:
    spec = yaml.safe_load((Path(__file__).parent / "pipeline.yaml").read_text())
    return Node.from_dict(spec)


def run_scenario(root: Node, session: dict) -> None:
    tracer = Tracer()
    engine = Engine(build_llm(), TOOLS, TRANSFORMS, tracer, approve_gate=approve_gate)

    print(f"\n=== scenario: {session} ===")
    try:
        engine.run(root, session)
    except GateRejected as exc:
        print(f"gate rejected: {exc}")

    print("trace:")
    for span in tracer.spans:
        status = "ERROR" if span.error else "ok"
        print(f"  {span.agent_name:<22} [{span.agent_class:<16}] {span.duration_s * 1000:6.2f}ms {status}")

    card = evaluate(root, session, tracer)
    print(f"scorecard: {card.status} (overall={card.overall:.2f})")
    for flaw in card.flaws:
        print(f"  {flaw.severity:<7} {flaw.check_id:<6} [{flaw.node}] {flaw.description}")

    if "final_reply" in session:
        print(f"final_reply: {session['final_reply']}")


def main() -> None:
    root = load_root()
    run_scenario(root, {"order_id": "A1001", "message": "I was charged twice, please refund me"})
    run_scenario(root, {"order_id": "A1002", "message": "the monitor is broken and shows errors"})
    run_scenario(root, {"order_id": "A1002", "message": "please refund this, it's not working"})
    run_scenario(root, {"order_id": "A1003", "message": "refund please, this is too expensive"})


if __name__ == "__main__":
    main()
