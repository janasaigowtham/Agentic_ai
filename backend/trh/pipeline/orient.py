"""Stage 01 -- Orienting agent.

Single point of failure: nothing downstream re-derives its work (spec section
5.1). Reads the raw trajectory and, when available, the declared pipeline
shape; writes the one Case Summary every later stage reads instead of the raw
trajectory. Model tier: strongest available, no exceptions.

plan_actual, step_type_map, and outcome are code-computed facts (the order
steps actually ran, their types, and the final output). Orient's own judge
call only supplies `goal` and `orient_notes`, which genuinely require reading
and summarizing rather than lookup.
"""
from __future__ import annotations

from trh.core.harness_models import CaseSummary, Trajectory
from trh.core.models import PipelineGraph
from trh.judge.client import JudgeClient
from trh.judge.markers import stage_marker

SYSTEM_PROMPT = (
    "You are Stage 01 (Orient) in the Trajectory Review Harness. You read a "
    "full agent execution trajectory once and produce the one summary every "
    "downstream stage will use instead of re-reading the raw trajectory. "
    'Respond with strict JSON: {"goal": "<one sentence, the task actually '
    'being carried out>", "orient_notes": ["<pointer, not a verdict>", ...]}. '
    "orient_notes are things worth a closer look, not judgments -- specialists "
    "make the actual calls."
)


def build_user_prompt(trajectory: Trajectory, graph: PipelineGraph | None) -> str:
    step_lines = [
        f"- {s.step_id} {s.agent_name} ({s.agent_class}, {s.step_type}) "
        f"[{s.started_at:.3f}s -> {s.completed_at:.3f}s]"
        for s in trajectory.steps
    ]
    declared = ""
    if graph is not None:
        declared = (
            f"\nDeclared pipeline: {graph.name} v{graph.version}, "
            f"root agent_class {graph.root.agent_class}\n"
        )
    return (
        f"Trajectory {trajectory.trajectory_id} for pipeline {trajectory.source_pipeline}.\n"
        f"{declared}"
        "Steps in execution order:\n" + "\n".join(step_lines) + f"\n\nFinal outcome: {trajectory.outcome!r}\n"
    )


def run_orient(
    trajectory: Trajectory, judge: JudgeClient, graph: PipelineGraph | None = None
) -> CaseSummary:
    marker = stage_marker("orient", trajectory.trajectory_id)
    answer = judge.complete(
        stage="orient",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(trajectory, graph),
        marker=marker,
    )
    plan_declared = [child.name for child in graph.root.children] if graph is not None else None
    return CaseSummary(
        trajectory_id=trajectory.trajectory_id,
        goal=answer.get("goal", ""),
        plan_declared=plan_declared,
        plan_actual=[s.step_id for s in trajectory.steps],
        outcome=trajectory.outcome,
        step_type_map={s.step_id: s.step_type for s in trajectory.steps},
        orient_notes=list(answer.get("orient_notes", [])),
    )
