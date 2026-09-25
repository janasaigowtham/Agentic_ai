import json
from pathlib import Path

from trh.core.graph import load_pipeline
from trh.core.trace import build_run
from trh.execution.executor import Executor
from trh.execution.mock_db import MockDatabase
from trh.execution.synthetic_data import LOAN_SCENARIOS, build_mock_database

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
AGENT_DIR = FIXTURES_DIR / "agents"


def _run_scenario(loan_number: str):
    graph = load_pipeline(AGENT_DIR, "pmi_ddn_pipeline")
    scenario = LOAN_SCENARIOS[loan_number]
    db = build_mock_database()
    executor = Executor(mock_db=db, gate_duration_s=scenario.gate_duration_s)
    spans = executor.run(graph, initial_session={"loan_number": loan_number}, trace_id=f"synthetic-{loan_number}")
    return scenario, build_run(spans)


def _span_by_name(run, name: str) -> dict:
    return next(s for s in run.spans if s.agent_name == name)


def test_icmp_required_branch_selected_when_letter_effective_date_present():
    scenario, run = _run_scenario("1234567890")
    assert scenario.letter_effective_date is not None
    router = _span_by_name(run, "pmi_ddn_icmp_decision_router")
    assert router.chosen_target == "pmi_ddn_icmp_process"
    names = {s.agent_name for s in run.spans}
    assert "pmi_ddn_icmp_process" in names
    assert "pmi_ddn_no_icmp_path_handler" not in names


def test_icmp_skip_branch_selected_when_letter_effective_date_absent():
    scenario, run = _run_scenario("2222222222")
    assert scenario.letter_effective_date is None
    router = _span_by_name(run, "pmi_ddn_icmp_decision_router")
    assert router.chosen_target == "pmi_ddn_no_icmp_path_handler"
    names = {s.agent_name for s in run.spans}
    assert "pmi_ddn_no_icmp_path_handler" in names
    assert "pmi_ddn_icmp_process" not in names


def test_investor_flag_reflects_investor_record_presence():
    _, run_with_investor = _run_scenario("1234567890")
    _, run_without_investor = _run_scenario("3333333333")

    flag_present = _span_by_name(run_with_investor, "pmi_ddn_investor_flag")
    flag_absent = _span_by_name(run_without_investor, "pmi_ddn_investor_flag")

    assert json.loads(flag_present.output_value) == "Y"
    assert json.loads(flag_absent.output_value) == "N"


def test_gate_duration_matches_scenario_config():
    scenario, run = _run_scenario("3333333333")
    gate = _span_by_name(run, "pmi_ddn_approval_gate")
    assert abs(gate.duration_s - scenario.gate_duration_s) < 1e-6


def test_llm_step_renders_undeclared_template_reference_organically():
    _, run = _run_scenario("1234567890")
    verdict_step = _span_by_name(run, "pmi_ddn_verdict_synthesizer")
    # pmi_ddn_msp_loan_data is not in this node's declared input_keys, but its
    # instruction template references it -- a real prompt render includes it
    # regardless, exactly like the N3-class finding does against the
    # hand-recorded fixture.
    assert "pmi_ddn_msp_loan_data" in verdict_step.template_references
    assert "pmi_ddn_msp_loan_data" not in verdict_step.input_keys
    assert "COMM" in verdict_step.rendered_prompt


def test_database_step_projects_only_declared_select_columns():
    _, run = _run_scenario("1234567890")
    fetch_step = _span_by_name(run, "pmi_ddn_fetch_msp_loan_data")
    rows = json.loads(fetch_step.output_value)
    assert len(rows) == 1
    row = rows[0]
    # the query's own SELECT list never asks for borrower_name, so the mock
    # DB -- unlike the hand-seeded fixture -- never returns it, even though
    # the seed data holds it.
    assert set(row.keys()) == {"ln_no", "inv_class_code", "letter_effective_date"}


def test_mock_database_returns_nothing_for_an_unseeded_loan_number():
    db = MockDatabase()
    db.seed_table("t", [{"k": "1", "v": "x"}], key_column="k")
    assert db.execute("SELECT v FROM t WHERE k = :k", {"k": "does-not-exist"}) == []


def test_full_run_review_against_every_synthetic_source_completes():
    from trh.config import HarnessConfig, JudgeMode
    from trh.fixtures_registry import TRAJECTORY_SOURCES
    from trh.pipeline.run_review import run_review

    config = HarnessConfig(mode=JudgeMode.MOCK, gemini_api_key=None, anthropic_api_key=None)
    synthetic_sources = [s for s in TRAJECTORY_SOURCES.values() if s.loan_number is not None]
    assert len(synthetic_sources) == len(LOAN_SCENARIOS)

    for source in synthetic_sources:
        result = run_review(source, config)
        assert result.trajectory.trajectory_id == source.trajectory_id
        assert result.case_summary.goal
        assert len(result.verdicts) > 0
        assert result.fact_check.confidence in {"high", "medium", "low"}
