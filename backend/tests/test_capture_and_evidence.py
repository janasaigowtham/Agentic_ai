from pathlib import Path

from trh.capture.simulate import TAP_A, TAP_B_ONLY, load_and_capture, simulate_capture
from trh.core.graph import load_pipeline
from trh.pipeline.evidence_assembly import assemble_evidence
from trh.pipeline.trajectory_assembler import assemble_trajectory

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
AGENT_DIR = FIXTURES_DIR / "agents"
TRACE_PATH = FIXTURES_DIR / "trace.jsonl"


def _load_trajectory():
    graph = load_pipeline(AGENT_DIR, "pmi_ddn_pipeline")
    run, captured = load_and_capture(str(TRACE_PATH))
    trajectory = assemble_trajectory("traj-test", "pmi_ddn_pipeline", captured, graph)
    assemble_evidence(trajectory, graph)
    return trajectory, graph


def test_capture_tags_every_span_tap_a_by_default():
    run, captured = load_and_capture(str(TRACE_PATH))
    assert len(captured) == 17
    assert all(c.tap == TAP_A for c in captured)
    assert all(c.has_pre_execution_snapshot for c in captured)


def test_capture_honors_bypassed_classes():
    run, captured = load_and_capture(str(TRACE_PATH))
    bypassed = simulate_capture(run, bypassed_classes=frozenset({"decision_router_agent"}))
    tapped = {c.span.agent_class: c.tap for c in bypassed}
    assert tapped["decision_router_agent"] == TAP_B_ONLY
    assert tapped["database_agent"] == TAP_A


def test_trajectory_assembler_preserves_order_and_hierarchy():
    trajectory, _ = _load_trajectory()
    assert len(trajectory.steps) == 17
    assert trajectory.steps == sorted(trajectory.steps, key=lambda s: s.started_at)
    assert trajectory.completed_at == 101.0
    root = trajectory.steps[0]
    assert root.agent_name == "pmi_ddn_pipeline"
    assert root.parent_step_id is None
    child = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_pre_process")
    assert child.parent_step_id == root.step_id


def test_evidence_flags_seeded_duplicate_db_call():
    trajectory, _ = _load_trajectory()
    fetch_steps = [s for s in trajectory.steps if s.agent_name == "pmi_ddn_fetch_msp_loan_data"]
    assert len(fetch_steps) == 2
    first, second = fetch_steps
    assert first.evidence["duplicate_of_step_ids"] == []
    assert second.evidence["duplicate_of_step_ids"] == [first.step_id]


def test_evidence_flags_seeded_undeclared_prompt_reference():
    trajectory, _ = _load_trajectory()
    verdict_step = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_verdict_synthesizer")
    assert verdict_step.evidence["undeclared_template_refs"] == ["pmi_ddn_msp_loan_data"]

    qa_step = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_qa_q25527")
    assert qa_step.evidence["undeclared_template_refs"] == []


def test_evidence_resolved_query_keeps_params_symbolic():
    trajectory, _ = _load_trajectory()
    fetch_step = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_fetch_msp_loan_data")
    assert fetch_step.evidence["query_params"] == ["loan_number"]
    assert ":loan_number" in fetch_step.evidence["resolved_query"]
    assert "1234567890" not in fetch_step.evidence["resolved_query"]


def test_evidence_computes_retrieved_columns_and_row_count():
    trajectory, _ = _load_trajectory()
    fetch_step = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_fetch_msp_loan_data")
    assert fetch_step.evidence["row_count"] == 1
    assert set(fetch_step.evidence["retrieved_columns"]) == {
        "ln_no",
        "inv_class_code",
        "letter_effective_date",
        "borrower_name",
        "borrower_ssn",
    }


def test_evidence_seeded_gate_and_router_facts_still_readable_from_raw_span():
    trajectory, _ = _load_trajectory()
    gate_step = next(s for s in trajectory.steps if s.agent_class == "agent_gate")
    assert (gate_step.completed_at - gate_step.started_at) < 0.01

    router_step = next(s for s in trajectory.steps if s.agent_name == "pmi_ddn_icmp_decision_router")
    operators = {
        cond["operator"]
        for route in router_step.raw_span.evaluated_routes
        for cond in route["conditions"]
    }
    assert operators == {"eq"}
