import pytest

from trh.config import HarnessConfig, JudgeMode
from trh.core.harness_models import GateDecisionValue, RecommendationStatus
from trh.db import store
from trh.fixtures_registry import TRAJECTORY_SOURCES
from trh.pipeline.run_review import run_review

SEEDED_SOURCE = TRAJECTORY_SOURCES["trace-fixture-0000000000000001"]


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "trh.sqlite3"
    connection = store.connect(str(db_path))
    yield connection
    connection.close()


@pytest.fixture
def review_result():
    config = HarnessConfig(mode=JudgeMode.MOCK, gemini_api_key=None, anthropic_api_key=None)
    return run_review(SEEDED_SOURCE, config)


def test_create_and_update_review_status(conn):
    review_id = store.create_review(conn)
    status = store.get_review_status(conn, review_id)
    assert status["status"] == "pending"

    store.update_review_status(conn, review_id, "running")
    assert store.get_review_status(conn, review_id)["status"] == "running"


def test_save_and_reload_full_review_result(conn, review_result):
    review_id = store.create_review(conn)
    store.save_review_result(conn, review_id, review_result)

    status = store.get_review_status(conn, review_id)
    assert status["status"] == "complete"
    trajectory_id = status["trajectory_id"]
    assert trajectory_id == review_result.trajectory.trajectory_id
    assert status["stage_lineage_used"] == review_result.stage_lineage_used

    trajectories = store.list_trajectories(conn)
    assert len(trajectories) == 1
    assert trajectories[0]["step_count"] == len(review_result.trajectory.steps)

    loaded_trajectory = store.load_trajectory(conn, trajectory_id)
    assert loaded_trajectory is not None
    assert [s.step_id for s in loaded_trajectory.steps] == [
        s.step_id for s in review_result.trajectory.steps
    ]
    assert loaded_trajectory.steps[3].evidence["duplicate_of_step_ids"] == ["s003"]

    case_summary = store.load_case_summary(conn, trajectory_id)
    assert case_summary.goal == review_result.case_summary.goal

    verdicts = store.load_verdicts(conn, trajectory_id)
    assert len(verdicts) == len(review_result.verdicts)

    recommendations = store.load_recommendations(conn, trajectory_id)
    assert len(recommendations) == len(review_result.recommendations)
    assert all(r.status is RecommendationStatus.PROPOSED for r in recommendations)

    fact_check = store.load_fact_check(conn, trajectory_id)
    assert fact_check.confidence == review_result.fact_check.confidence


def test_rerunning_review_replaces_recommendations_instead_of_accumulating(conn, review_result):
    review_id_1 = store.create_review(conn)
    store.save_review_result(conn, review_id_1, review_result)
    trajectory_id = review_result.trajectory.trajectory_id
    first_run_count = len(store.load_recommendations(conn, trajectory_id))

    review_id_2 = store.create_review(conn)
    store.save_review_result(conn, review_id_2, review_result)
    second_run_count = len(store.load_recommendations(conn, trajectory_id))

    assert second_run_count == first_run_count


def test_gate_approve_updates_recommendation_status(conn, review_result):
    review_id = store.create_review(conn)
    store.save_review_result(conn, review_id, review_result)
    trajectory_id = review_result.trajectory.trajectory_id
    recommendation = store.load_recommendations(conn, trajectory_id)[0]

    store.apply_gate_decision(conn, recommendation.recommendation_id, GateDecisionValue.APPROVED, "reviewer-1")

    updated = store.get_recommendation(conn, recommendation.recommendation_id)
    assert updated.status is RecommendationStatus.APPROVED


def test_gate_decline_seeds_calibration_record(conn, review_result):
    review_id = store.create_review(conn)
    store.save_review_result(conn, review_id, review_result)
    trajectory_id = review_result.trajectory.trajectory_id
    recommendation = store.load_recommendations(conn, trajectory_id)[0]

    store.apply_gate_decision(
        conn, recommendation.recommendation_id, GateDecisionValue.DECLINED, "reviewer-1", note="not applicable"
    )

    updated = store.get_recommendation(conn, recommendation.recommendation_id)
    assert updated.status is RecommendationStatus.DECLINED

    rows = conn.execute("SELECT * FROM calibration_records WHERE trajectory_id = ?", (trajectory_id,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["agreement"] == 0


def test_hold_recommendation_sets_status_without_gate_decision_row(conn, review_result):
    review_id = store.create_review(conn)
    store.save_review_result(conn, review_id, review_result)
    trajectory_id = review_result.trajectory.trajectory_id
    recommendation = store.load_recommendations(conn, trajectory_id)[0]

    store.hold_recommendation(conn, recommendation.recommendation_id)

    updated = store.get_recommendation(conn, recommendation.recommendation_id)
    assert updated.status is RecommendationStatus.HELD
    rows = conn.execute(
        "SELECT * FROM gate_decisions WHERE recommendation_id = ?", (recommendation.recommendation_id,)
    ).fetchall()
    assert len(rows) == 0
