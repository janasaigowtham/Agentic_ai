"""POST /api/recommendations/{id}/gate -- approve / decline / hold.

Never applies a recommendation to a live system from here; this only records
the human's decision (spec section 5.5). A decline additionally seeds a
CalibrationRecord candidate rather than being silently discarded (section 5.6).
"""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from trh.api.deps import get_conn
from trh.api.schemas import GateDecisionRequest
from trh.core.harness_models import GateDecisionValue
from trh.db import store

router = APIRouter(prefix="/api/recommendations", tags=["gate"])


@router.post("/{recommendation_id}/gate")
def apply_gate(
    recommendation_id: str,
    body: GateDecisionRequest,
    conn: sqlite3.Connection = Depends(get_conn),
):
    recommendation = store.get_recommendation(conn, recommendation_id)
    if recommendation is None:
        raise HTTPException(status_code=404, detail="recommendation not found")

    if body.decision == "hold":
        store.hold_recommendation(conn, recommendation_id)
    else:
        decision = (
            GateDecisionValue.APPROVED if body.decision == "approved" else GateDecisionValue.DECLINED
        )
        store.apply_gate_decision(conn, recommendation_id, decision, body.reviewer, body.note)

    updated = store.get_recommendation(conn, recommendation_id)
    return updated.to_dict()
