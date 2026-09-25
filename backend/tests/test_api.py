import time

from fastapi.testclient import TestClient

from trh.api.main import create_app


def _client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    return TestClient(app)


def _complete_review(client, trajectory_id):
    review_id = client.post(f"/api/trajectories/{trajectory_id}/review").json()["review_id"]
    status = None
    for _ in range(20):
        status = client.get(f"/api/reviews/{review_id}").json()
        if status["status"] == "complete":
            break
        time.sleep(0.05)
    return status


def test_list_and_get_trajectory(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/trajectories")
    assert resp.status_code == 200
    trajectories = resp.json()
    assert len(trajectories) == 5

    seeded = next(t for t in trajectories if t["trajectory_id"] == "trace-fixture-0000000000000001")
    detail = client.get(f"/api/trajectories/{seeded['trajectory_id']}")
    assert detail.status_code == 200
    assert len(detail.json()["steps"]) == 17

    synthetic = next(t for t in trajectories if t["trajectory_id"] == "synthetic-1234567890")
    synthetic_detail = client.get(f"/api/trajectories/{synthetic['trajectory_id']}")
    assert synthetic_detail.status_code == 200
    assert len(synthetic_detail.json()["steps"]) == 16


def test_get_trajectory_404_for_unknown_id(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/trajectories/does-not-exist")
    assert resp.status_code == 404


def test_trigger_review_and_poll_to_complete(tmp_path):
    client = _client(tmp_path)
    trajectory_id = client.get("/api/trajectories").json()[0]["trajectory_id"]

    trigger = client.post(f"/api/trajectories/{trajectory_id}/review")
    assert trigger.status_code == 200
    assert trigger.json()["status"] == "pending"

    status = _complete_review(client, trajectory_id)

    assert status["status"] == "complete"
    assert status["stage_lineage_used"] == {
        "orient": "mock",
        "specialist": "mock",
        "aggregator": "mock",
        "fact_check": "mock",
    }
    assert len(status["verdicts"]) > 0
    assert len(status["recommendations"]) >= 1
    assert status["case_summary"]["goal"]
    assert status["fact_check"]["confidence"] in {"high", "medium", "low"}


def test_gate_approve_decline_hold_flow(tmp_path):
    client = _client(tmp_path)
    trajectory_id = client.get("/api/trajectories").json()[0]["trajectory_id"]
    status = _complete_review(client, trajectory_id)
    recommendations = status["recommendations"]
    assert len(recommendations) >= 3

    approve_resp = client.post(
        f"/api/recommendations/{recommendations[0]['recommendation_id']}/gate",
        json={"decision": "approved", "reviewer": "alice"},
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "approved"

    decline_resp = client.post(
        f"/api/recommendations/{recommendations[1]['recommendation_id']}/gate",
        json={"decision": "declined", "reviewer": "alice", "note": "not applicable"},
    )
    assert decline_resp.json()["status"] == "declined"

    hold_resp = client.post(
        f"/api/recommendations/{recommendations[2]['recommendation_id']}/gate",
        json={"decision": "hold", "reviewer": "alice"},
    )
    assert hold_resp.json()["status"] == "held"


def test_gate_404_for_unknown_recommendation(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/api/recommendations/does-not-exist/gate", json={"decision": "approved", "reviewer": "alice"}
    )
    assert resp.status_code == 404


def test_config_endpoint_reports_mock_mode_with_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _client(tmp_path)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "mock"
