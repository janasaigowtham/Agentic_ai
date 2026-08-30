"""Tests for the course's working example.

Run with: pytest (from course/example). No network, no API key - the
control flow is tested against the deterministic MockLLM, exactly the
"pin the non-deterministic parts to test the rest" principle from
course/07-reliability-evaluation-observability.md.
"""
from agent_framework import Engine, GateRejected
from eval import evaluate
from llm_client import MockLLM
from run_pipeline import TOOLS, TRANSFORMS, approve_gate, load_root
from tracing import Tracer


def _run(session):
    root = load_root()
    tracer = Tracer()
    engine = Engine(MockLLM(), TOOLS, TRANSFORMS, tracer, approve_gate=approve_gate)
    return root, tracer, engine, session


def test_technical_route_runs_only_technical_branch():
    root, tracer, engine, session = _run({"order_id": "A1002", "message": "the monitor is broken"})
    engine.run(root, session)
    names = tracer.executed_agents()
    assert "technical_flow" in names
    assert "refund_flow" not in names
    assert session["intent"] == "technical"
    assert "final_reply" in session


def test_small_refund_auto_approves_without_flaws():
    root, tracer, engine, session = _run({"order_id": "A1001", "message": "please refund me, charged twice"})
    engine.run(root, session)
    card = evaluate(root, session, tracer)
    assert card.status == "PASS"
    assert session["order"]["order_id"] in session["final_reply"]


def test_general_route_is_the_catch_all():
    root, tracer, engine, session = _run({"order_id": "A1001", "message": "what are your store hours?"})
    engine.run(root, session)
    names = tracer.executed_agents()
    assert "general_flow" in names
    assert session["intent"] == "general"
    card = evaluate(root, session, tracer)
    assert card.status == "PASS"


def test_large_refund_is_rejected_by_gate():
    root, tracer, engine, session = _run({"order_id": "A1003", "message": "refund please"})
    try:
        engine.run(root, session)
        assert False, "expected GateRejected"
    except GateRejected:
        pass
    assert "final_reply" not in session
    card = evaluate(root, session, tracer)
    assert card.status == "FAIL"
    assert any(f.check_id == "O-S3" for f in card.flaws)


def test_evaluator_flags_a_gate_that_does_not_really_gate():
    # A gate wired to always approve instantly, even for a refund big
    # enough to require real review, should be flagged - see
    # course/08-safety-and-guardrails.md.
    root, tracer, engine, session = _run({"order_id": "A1002", "message": "refund please, not working"})
    engine.approve_gate = lambda node, s: True
    engine.run(root, session)
    card = evaluate(root, session, tracer)
    assert any(f.check_id == "G-R1" for f in card.flaws)


def test_evaluator_flags_a_reply_missing_the_order_id():
    root, tracer, engine, session = _run({"order_id": "A1001", "message": "please refund me"})
    engine.run(root, session)
    session["final_reply"] = "Your refund has been processed."  # simulate a bad draft
    card = evaluate(root, session, tracer)
    assert any(f.check_id == "N5" for f in card.flaws)


def test_only_one_router_branch_ever_executes():
    for order_id, message in [
        ("A1001", "charged twice, refund me"),
        ("A1002", "this is broken"),
        ("A1001", "hello there"),
    ]:
        root, tracer, engine, session = _run({"order_id": order_id, "message": message})
        engine.run(root, session)
        ran = {"refund_flow", "technical_flow", "general_flow"} & set(tracer.executed_agents())
        assert len(ran) == 1
