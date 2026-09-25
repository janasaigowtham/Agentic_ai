from pathlib import Path

from trh.capture.simulate import load_and_capture
from trh.core.graph import load_pipeline
from trh.core.harness_models import CaseSummary, VerdictSeverity
from trh.judge.client import MockJudgeClient
from trh.pipeline.evidence_assembly import assemble_evidence
from trh.pipeline.specialists import ALL_SPECIALISTS, coverage_report, run_specialists
from trh.pipeline.trajectory_assembler import assemble_trajectory

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
AGENT_DIR = FIXTURES_DIR / "agents"
TRACE_PATH = FIXTURES_DIR / "trace.jsonl"


def _build_trajectory():
    graph = load_pipeline(AGENT_DIR, "pmi_ddn_pipeline")
    _, captured = load_and_capture(str(TRACE_PATH))
    trajectory = assemble_trajectory("trace-fixture-0000000000000001", "pmi_ddn_pipeline", captured, graph)
    assemble_evidence(trajectory, graph)
    return trajectory


def _case_summary(trajectory):
    return CaseSummary(
        trajectory_id=trajectory.trajectory_id,
        goal="Process a PMI DDN loan servicing decision end to end.",
        plan_declared=None,
        plan_actual=[s.step_id for s in trajectory.steps],
        outcome=trajectory.outcome,
        step_type_map={s.step_id: s.step_type for s in trajectory.steps},
    )


def test_registry_has_all_nine_specialists_with_distinct_names():
    names = {s.name for s in ALL_SPECIALISTS}
    assert len(names) == 9
    assert names == {
        "llm_reasoning",
        "database",
        "rest_api",
        "transformation",
        "decision_routing",
        "iteration",
        "gate_escalation",
        "orchestration_plan_fidelity",
        "lam_domain",
    }


def test_coverage_report_matches_fixture_expectations():
    trajectory = _build_trajectory()
    coverage = coverage_report(trajectory)
    assert coverage["llm_reasoning"] is True
    assert coverage["database"] is True
    assert coverage["transformation"] is True
    assert coverage["decision_routing"] is True
    assert coverage["gate_escalation"] is True
    assert coverage["orchestration_plan_fidelity"] is True
    # structurally absent from this fixture's pipeline
    assert coverage["rest_api"] is False
    assert coverage["iteration"] is False
    assert coverage["lam_domain"] is False


def test_run_specialists_produces_a_verdict_for_every_matching_step():
    trajectory = _build_trajectory()
    case_summary = _case_summary(trajectory)
    judge = MockJudgeClient()
    verdicts = run_specialists(trajectory, case_summary, judge)

    matching_steps = [
        step
        for step in trajectory.steps
        for specialist in ALL_SPECIALISTS
        if specialist.fires_for(step)
    ]
    assert len(verdicts) == len(matching_steps)


def test_seeded_duplicate_db_call_surfaces_as_significant_database_verdict():
    trajectory = _build_trajectory()
    case_summary = _case_summary(trajectory)
    verdicts = run_specialists(trajectory, case_summary, MockJudgeClient())

    dup_verdicts = [
        v for v in verdicts if v.specialist == "database" and v.severity is not VerdictSeverity.NONE
    ]
    assert len(dup_verdicts) == 1
    assert "twice" in dup_verdicts[0].finding


def test_seeded_undeclared_prompt_reference_surfaces_as_critical_llm_verdict():
    trajectory = _build_trajectory()
    case_summary = _case_summary(trajectory)
    verdicts = run_specialists(trajectory, case_summary, MockJudgeClient())

    critical = [
        v for v in verdicts if v.specialist == "llm_reasoning" and v.severity is VerdictSeverity.CRITICAL
    ]
    assert len(critical) == 1
    assert "undeclared" in critical[0].finding


def test_seeded_gate_and_router_findings_surface():
    trajectory = _build_trajectory()
    case_summary = _case_summary(trajectory)
    verdicts = run_specialists(trajectory, case_summary, MockJudgeClient())

    gate_findings = [v for v in verdicts if v.specialist == "gate_escalation"]
    assert any(v.severity is VerdictSeverity.SIGNIFICANT for v in gate_findings)

    router_findings = [v for v in verdicts if v.specialist == "decision_routing"]
    assert any(v.severity is VerdictSeverity.MINOR for v in router_findings)


def test_clean_steps_still_produce_a_recorded_none_severity_verdict():
    trajectory = _build_trajectory()
    case_summary = _case_summary(trajectory)
    verdicts = run_specialists(trajectory, case_summary, MockJudgeClient())

    # pmi_ddn_fetch_investor_data has no seeded issue -- it should still be
    # recorded as "examined, nothing found", not silently dropped.
    investor_fetch_verdicts = [v for v in verdicts if "s005" in v.step_ids]
    assert len(investor_fetch_verdicts) == 1
    assert investor_fetch_verdicts[0].severity is VerdictSeverity.NONE
