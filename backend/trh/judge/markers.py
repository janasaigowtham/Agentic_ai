"""Marker construction shared by judge callers (specialists, orient, aggregator,
fact-check) and MockJudgeClient's answer key. A marker identifies *which call
this is* (which specialist, which step(s), or which stage/trajectory) -- it
carries no opinion about the trajectory itself.
"""
from __future__ import annotations


def specialist_marker(specialist: str, step_ids: list[str]) -> str:
    return f"specialist:{specialist}:{'+'.join(step_ids)}"


def stage_marker(stage: str, trajectory_id: str) -> str:
    return f"{stage}:{trajectory_id}"
