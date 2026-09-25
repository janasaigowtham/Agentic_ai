from trh.config import HarnessConfig, JudgeMode
from trh.core.harness_models import VerdictSeverity
from trh.fixtures_registry import TRAJECTORY_SOURCES
from trh.pipeline.run_review import run_review

SEEDED_SOURCE = TRAJECTORY_SOURCES["trace-fixture-0000000000000001"]


def _mock_config() -> HarnessConfig:
    return HarnessConfig(mode=JudgeMode.MOCK, gemini_api_key=None, anthropic_api_key=None)


def _run():
    return run_review(SEEDED_SOURCE, _mock_config())


def test_run_review_produces_case_summary_and_lineage_all_mock():
    result = _run()
    assert result.case_summary.goal
    assert result.trajectory.trajectory_id == "trace-fixture-0000000000000001"
    assert set(result.stage_lineage_used.values()) == {"mock"}


def test_run_review_surfaces_duplicate_database_call_verdict():
    result = _run()
    dup_verdicts = [
        v
        for v in result.verdicts
        if v.specialist == "database" and v.severity is VerdictSeverity.SIGNIFICANT
    ]
    assert len(dup_verdicts) == 1


def test_run_review_surfaces_undeclared_prompt_injection_verdict():
    result = _run()
    critical_verdicts = [v for v in result.verdicts if v.severity is VerdictSeverity.CRITICAL]
    assert len(critical_verdicts) == 1
    assert critical_verdicts[0].specialist == "llm_reasoning"


def test_run_review_aggregator_links_shared_prompt_block_across_llm_steps():
    result = _run()
    shared_block_recs = [r for r in result.recommendations if "ABSOLUTE RULE" in r.description]
    assert len(shared_block_recs) == 1
    assert shared_block_recs[0].tied_to_root_cause
    assert len(shared_block_recs[0].root_cause_chain) == 2


def test_run_review_eq_only_router_or_plan_fidelity_verdict_present():
    result = _run()
    assert any(
        v.specialist == "decision_routing" and v.severity is VerdictSeverity.MINOR
        for v in result.verdicts
    )


def test_run_review_produces_at_least_one_root_caused_recommendation():
    result = _run()
    root_caused = [r for r in result.recommendations if r.tied_to_root_cause and r.root_cause_chain]
    assert len(root_caused) >= 1


def test_run_review_fact_check_ran_and_returned_confidence():
    result = _run()
    assert result.fact_check.confidence in {"high", "medium", "low"}
