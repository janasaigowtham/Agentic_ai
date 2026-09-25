"""GET /api/trajectories, GET /api/trajectories/{id}.

Browsing available trajectory sources and their assembled steps (Capture ->
Trajectory Assembler -> Evidence Assembly) is independent of whether a
judgment review has been run against them yet -- the step viewer needs to
show the trajectory as soon as a record is selected, before Orient ever runs.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trh.capture.simulate import load_and_capture
from trh.core.graph import load_pipeline
from trh.core.harness_models import Trajectory
from trh.fixtures_registry import TRAJECTORY_SOURCES, TrajectorySource
from trh.pipeline.evidence_assembly import assemble_evidence
from trh.pipeline.trajectory_assembler import assemble_trajectory

router = APIRouter(prefix="/api/trajectories", tags=["trajectories"])


def _assemble(source: TrajectorySource) -> Trajectory:
    graph = load_pipeline(source.agent_dir, source.pipeline_name)
    _, captured = load_and_capture(source.trace_path)
    trajectory = assemble_trajectory(source.trajectory_id, source.pipeline_name, captured, graph)
    assemble_evidence(trajectory, graph)
    return trajectory


@router.get("")
def list_trajectories():
    summaries = []
    for source in TRAJECTORY_SOURCES.values():
        trajectory = _assemble(source)
        summaries.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "source_pipeline": trajectory.source_pipeline,
                "label": source.label,
                "step_count": len(trajectory.steps),
                "started_at": trajectory.started_at,
                "completed_at": trajectory.completed_at,
            }
        )
    return summaries


@router.get("/{trajectory_id}")
def get_trajectory(trajectory_id: str):
    source = TRAJECTORY_SOURCES.get(trajectory_id)
    if source is None:
        raise HTTPException(status_code=404, detail="trajectory not found")
    trajectory = _assemble(source)
    return trajectory.to_dict()
