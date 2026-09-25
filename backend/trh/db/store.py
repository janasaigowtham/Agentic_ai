"""SQLite persistence for the harness's own tables. stdlib sqlite3 only.

Full isolation (spec section 2 / SDK section constraints): this is the
harness's own database, its own schema, imported by nothing outside trh.* --
a harness failure here can't touch a target system's own storage.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trh.core.harness_models import (
    CaseSummary,
    GateDecisionValue,
    Recommendation,
    RecommendationStatus,
    Step,
    Trajectory,
    Verdict,
    VerdictSeverity,
)
from trh.pipeline.fact_check import FactCheckResult
from trh.pipeline.run_review import ReviewResult

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(db_path: str) -> sqlite3.Connection:
    # FastAPI offloads sync routes (and their generator dependencies) to a
    # worker thread pool; a request's connection setup and teardown aren't
    # guaranteed to land on the same pool thread, which sqlite3 rejects by
    # default. Each connection here is still only ever used by one logical
    # request/background-task flow at a time, so disabling the same-thread
    # check is safe.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA_PATH.read_text())
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _load(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    return json.loads(text)


# --- reviews (async trigger/poll status tracking) -------------------------

def create_review(conn: sqlite3.Connection, trajectory_id: str | None = None) -> str:
    review_id = f"rev-{uuid.uuid4().hex[:12]}"
    now = _now()
    conn.execute(
        "INSERT INTO reviews (review_id, trajectory_id, status, error, stage_lineage_used, "
        "created_at, updated_at) VALUES (?, ?, 'pending', NULL, NULL, ?, ?)",
        (review_id, trajectory_id, now, now),
    )
    conn.commit()
    return review_id


def update_review_status(
    conn: sqlite3.Connection,
    review_id: str,
    status: str,
    *,
    trajectory_id: str | None = None,
    stage_lineage_used: dict | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE reviews SET status = ?, error = COALESCE(?, error), "
        "trajectory_id = COALESCE(?, trajectory_id), "
        "stage_lineage_used = COALESCE(?, stage_lineage_used), updated_at = ? "
        "WHERE review_id = ?",
        (status, error, trajectory_id, _dump(stage_lineage_used), _now(), review_id),
    )
    conn.commit()


def get_review_status(conn: sqlite3.Connection, review_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
    if row is None:
        return None
    return {
        "review_id": row["review_id"],
        "trajectory_id": row["trajectory_id"],
        "status": row["status"],
        "error": row["error"],
        "stage_lineage_used": _load(row["stage_lineage_used"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# --- saving a full review result -------------------------------------------

def save_review_result(conn: sqlite3.Connection, review_id: str, result: ReviewResult) -> None:
    try:
        _save_trajectory(conn, result.trajectory)
        _save_case_summary(conn, result.case_summary)
        _save_verdicts(conn, result.trajectory.trajectory_id, result.verdicts)
        _save_recommendations(conn, result.trajectory.trajectory_id, result.recommendations)
        _save_fact_check(conn, result.trajectory.trajectory_id, result.fact_check)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    update_review_status(
        conn,
        review_id,
        "complete",
        trajectory_id=result.trajectory.trajectory_id,
        stage_lineage_used=result.stage_lineage_used,
    )


def _save_trajectory(conn: sqlite3.Connection, trajectory: Trajectory) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO trajectories "
        "(trajectory_id, source_pipeline, source_spec_ref, started_at, completed_at, outcome) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            trajectory.trajectory_id,
            trajectory.source_pipeline,
            trajectory.source_spec_ref,
            trajectory.started_at,
            trajectory.completed_at,
            _dump(trajectory.outcome),
        ),
    )
    conn.execute("DELETE FROM steps WHERE trajectory_id = ?", (trajectory.trajectory_id,))
    for seq, step in enumerate(trajectory.steps):
        conn.execute(
            "INSERT INTO steps (step_id, trajectory_id, seq, step_type, agent_name, agent_class, "
            "input, output, started_at, completed_at, parent_step_id, declared_contract_ref, tap, "
            "evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                step.step_id,
                trajectory.trajectory_id,
                seq,
                step.step_type,
                step.agent_name,
                step.agent_class,
                _dump(step.input),
                _dump(step.output),
                step.started_at,
                step.completed_at,
                step.parent_step_id,
                step.declared_contract_ref,
                step.tap,
                _dump(step.evidence),
            ),
        )


def _save_case_summary(conn: sqlite3.Connection, case_summary: CaseSummary) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO case_summaries "
        "(trajectory_id, goal, plan_declared, plan_actual, outcome, step_type_map, orient_notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            case_summary.trajectory_id,
            case_summary.goal,
            _dump(case_summary.plan_declared),
            _dump(case_summary.plan_actual),
            _dump(case_summary.outcome),
            _dump(case_summary.step_type_map),
            _dump(case_summary.orient_notes),
        ),
    )


def _save_verdicts(conn: sqlite3.Connection, trajectory_id: str, verdicts: list[Verdict]) -> None:
    conn.execute("DELETE FROM verdicts WHERE trajectory_id = ?", (trajectory_id,))
    for v in verdicts:
        conn.execute(
            "INSERT INTO verdicts (verdict_id, trajectory_id, specialist, step_ids, trigger_event, "
            "finding, severity, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                v.verdict_id,
                trajectory_id,
                v.specialist,
                _dump(v.step_ids),
                v.trigger_event,
                v.finding,
                v.severity.value,
                _dump(v.evidence),
            ),
        )


def _save_recommendations(
    conn: sqlite3.Connection, trajectory_id: str, recommendations: list[Recommendation]
) -> None:
    # Re-running a review re-derives recommendations with fresh ids each time;
    # clear this trajectory's old rows first so they don't accumulate
    # indefinitely (mirrors _save_verdicts' delete-then-insert).
    conn.execute("DELETE FROM recommendations WHERE trajectory_id = ?", (trajectory_id,))
    for r in recommendations:
        conn.execute(
            "INSERT OR REPLACE INTO recommendations (recommendation_id, trajectory_id, "
            "tied_to_root_cause, root_cause_chain, description, proposed_fix, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                r.recommendation_id,
                r.trajectory_id,
                int(r.tied_to_root_cause),
                _dump(r.root_cause_chain),
                r.description,
                r.proposed_fix,
                r.status.value,
            ),
        )


def _save_fact_check(conn: sqlite3.Connection, trajectory_id: str, fact_check: FactCheckResult) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO fact_checks (trajectory_id, confidence, notes, routed_back) "
        "VALUES (?, ?, ?, ?)",
        (trajectory_id, fact_check.confidence, fact_check.notes, int(fact_check.routed_back)),
    )


# --- reads -------------------------------------------------------------

def list_trajectories(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT t.trajectory_id, t.source_pipeline, t.started_at, t.completed_at, "
        "(SELECT COUNT(*) FROM steps s WHERE s.trajectory_id = t.trajectory_id) AS step_count "
        "FROM trajectories t ORDER BY t.started_at"
    ).fetchall()
    return [dict(row) for row in rows]


def load_trajectory(conn: sqlite3.Connection, trajectory_id: str) -> Trajectory | None:
    traj_row = conn.execute(
        "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchone()
    if traj_row is None:
        return None
    step_rows = conn.execute(
        "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY seq", (trajectory_id,)
    ).fetchall()
    steps = [
        Step(
            step_id=row["step_id"],
            step_type=row["step_type"],
            agent_name=row["agent_name"],
            agent_class=row["agent_class"],
            input=_load(row["input"]),
            output=_load(row["output"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            parent_step_id=row["parent_step_id"],
            declared_contract_ref=row["declared_contract_ref"],
            tap=row["tap"],
            evidence=_load(row["evidence"], {}),
        )
        for row in step_rows
    ]
    return Trajectory(
        trajectory_id=traj_row["trajectory_id"],
        source_pipeline=traj_row["source_pipeline"],
        source_spec_ref=traj_row["source_spec_ref"],
        steps=steps,
        started_at=traj_row["started_at"],
        completed_at=traj_row["completed_at"],
        outcome=_load(traj_row["outcome"]),
    )


def load_case_summary(conn: sqlite3.Connection, trajectory_id: str) -> CaseSummary | None:
    row = conn.execute(
        "SELECT * FROM case_summaries WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchone()
    if row is None:
        return None
    return CaseSummary(
        trajectory_id=row["trajectory_id"],
        goal=row["goal"],
        plan_declared=_load(row["plan_declared"]),
        plan_actual=_load(row["plan_actual"], []),
        outcome=_load(row["outcome"]),
        step_type_map=_load(row["step_type_map"], {}),
        orient_notes=_load(row["orient_notes"], []),
    )


def load_verdicts(conn: sqlite3.Connection, trajectory_id: str) -> list[Verdict]:
    rows = conn.execute(
        "SELECT * FROM verdicts WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchall()
    return [
        Verdict(
            verdict_id=row["verdict_id"],
            specialist=row["specialist"],
            step_ids=_load(row["step_ids"], []),
            trigger_event=row["trigger_event"],
            finding=row["finding"],
            severity=VerdictSeverity(row["severity"]),
            evidence=_load(row["evidence"], {}),
        )
        for row in rows
    ]


def load_recommendations(conn: sqlite3.Connection, trajectory_id: str) -> list[Recommendation]:
    rows = conn.execute(
        "SELECT * FROM recommendations WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchall()
    return [
        Recommendation(
            recommendation_id=row["recommendation_id"],
            trajectory_id=row["trajectory_id"],
            tied_to_root_cause=bool(row["tied_to_root_cause"]),
            root_cause_chain=_load(row["root_cause_chain"], []),
            description=row["description"],
            proposed_fix=row["proposed_fix"],
            status=RecommendationStatus(row["status"]),
        )
        for row in rows
    ]


def load_fact_check(conn: sqlite3.Connection, trajectory_id: str) -> FactCheckResult | None:
    row = conn.execute(
        "SELECT * FROM fact_checks WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchone()
    if row is None:
        return None
    return FactCheckResult(
        confidence=row["confidence"], notes=row["notes"], routed_back=bool(row["routed_back"])
    )


def get_recommendation(conn: sqlite3.Connection, recommendation_id: str) -> Recommendation | None:
    row = conn.execute(
        "SELECT * FROM recommendations WHERE recommendation_id = ?", (recommendation_id,)
    ).fetchone()
    if row is None:
        return None
    return Recommendation(
        recommendation_id=row["recommendation_id"],
        trajectory_id=row["trajectory_id"],
        tied_to_root_cause=bool(row["tied_to_root_cause"]),
        root_cause_chain=_load(row["root_cause_chain"], []),
        description=row["description"],
        proposed_fix=row["proposed_fix"],
        status=RecommendationStatus(row["status"]),
    )


# --- gate: approve / decline / hold -------------------------------------

def apply_gate_decision(
    conn: sqlite3.Connection,
    recommendation_id: str,
    decision: GateDecisionValue,
    reviewer: str,
    note: str | None = None,
) -> None:
    """Approve or decline. Never applies the recommendation automatically to
    anything -- this only records the human's decision (spec section 5.5). A
    decline additionally seeds a CalibrationRecord candidate, never silently
    discarded (section 5.6)."""
    now = _now()
    conn.execute(
        "INSERT INTO gate_decisions (recommendation_id, decision, reviewer, note, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (recommendation_id, decision.value, reviewer, note, now),
    )
    status = (
        RecommendationStatus.APPROVED
        if decision is GateDecisionValue.APPROVED
        else RecommendationStatus.DECLINED
    )
    conn.execute(
        "UPDATE recommendations SET status = ? WHERE recommendation_id = ?",
        (status.value, recommendation_id),
    )
    if decision is GateDecisionValue.DECLINED:
        rec = get_recommendation(conn, recommendation_id)
        trajectory_id = rec.trajectory_id if rec else None
        conn.execute(
            "INSERT INTO calibration_records (case_id, trajectory_id, harness_verdict, "
            "human_label, agreement, reviewed_by, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"cal-{uuid.uuid4().hex[:12]}",
                trajectory_id,
                _dump(rec.to_dict() if rec else None),
                _dump({"decision": "declined", "note": note}),
                0,
                reviewer,
                now,
            ),
        )
    conn.commit()


def hold_recommendation(conn: sqlite3.Connection, recommendation_id: str) -> None:
    """'Hold' isn't a GateDecision value (spec section 4.5 has no such enum
    member); it updates Recommendation.status directly instead of inventing one."""
    conn.execute(
        "UPDATE recommendations SET status = ? WHERE recommendation_id = ?",
        (RecommendationStatus.HELD.value, recommendation_id),
    )
    conn.commit()
