"""POST /api/trajectories/{id}/review, GET /api/reviews/{review_id}.

The async trigger/poll pattern around the judgment pipeline (Orient ->
Specialists -> Aggregator -> Fact-check). Triggering returns immediately with
a review_id; the pipeline itself runs in a background task so a slow live-mode
run never blocks the request, and polling reads status from the `reviews`
table plus, once complete, the persisted case summary/verdicts/recommendations.
"""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from trh.api.deps import get_conn
from trh.config import load_config
from trh.db import store
from trh.fixtures_registry import TRAJECTORY_SOURCES, TrajectorySource
from trh.pipeline.run_review import run_review

router = APIRouter(tags=["reviews"])


def _run_and_persist(db_path: str, review_id: str, source: TrajectorySource) -> None:
    conn = store.connect(db_path)
    try:
        store.update_review_status(conn, review_id, "running")
        config = load_config()
        result = run_review(source, config)
        store.save_review_result(conn, review_id, result)
    except Exception as exc:  # surfaced through polling, never raised back to the caller
        store.update_review_status(conn, review_id, "failed", error=str(exc))
    finally:
        conn.close()


@router.post("/api/trajectories/{trajectory_id}/review")
def trigger_review(
    trajectory_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_conn),
):
    source = TRAJECTORY_SOURCES.get(trajectory_id)
    if source is None:
        raise HTTPException(status_code=404, detail="trajectory not found")
    review_id = store.create_review(conn, trajectory_id=trajectory_id)
    background_tasks.add_task(_run_and_persist, request.app.state.db_path, review_id, source)
    return {"review_id": review_id, "status": "pending"}


@router.get("/api/reviews/{review_id}")
def get_review(review_id: str, conn: sqlite3.Connection = Depends(get_conn)):
    status = store.get_review_status(conn, review_id)
    if status is None:
        raise HTTPException(status_code=404, detail="review not found")
    response = dict(status)
    if status["status"] == "complete" and status["trajectory_id"]:
        trajectory_id = status["trajectory_id"]
        case_summary = store.load_case_summary(conn, trajectory_id)
        verdicts = store.load_verdicts(conn, trajectory_id)
        recommendations = store.load_recommendations(conn, trajectory_id)
        fact_check = store.load_fact_check(conn, trajectory_id)
        response["case_summary"] = case_summary.to_dict() if case_summary else None
        response["verdicts"] = [v.to_dict() for v in verdicts]
        response["recommendations"] = [r.to_dict() for r in recommendations]
        response["fact_check"] = fact_check.to_dict() if fact_check else None
    return response
