"""Runs the check registry against a Run/PipelineGraph and rolls up a Scorecard."""
from __future__ import annotations

from typing import Any

from .models import Flaw, PipelineGraph, Run, Scorecard, Severity
from .registry import AXES, all_checks

SEVERITY_PENALTY = {
    Severity.INFO: 0.0,
    Severity.SEV_3: 0.05,
    Severity.SEV_2: 0.15,
    Severity.SEV_1: 0.40,
}

DEFAULT_WEIGHTS = {
    "structural": 0.30,
    "output_quality": 0.25,
    "efficiency": 0.20,
    "routing": 0.15,
    "prompt_quality": 0.10,
}


def _should_skip(meta, run: Run, config: dict, structural_sev1_seen: bool) -> bool:
    checks_enabled = config.get("checks_enabled")
    if checks_enabled and meta.id not in checks_enabled:
        return True
    checks_disabled = config.get("checks_disabled")
    if checks_disabled and meta.id in checks_disabled:
        return True
    if meta.needs_judge and config.get("_judge") is None:
        return True
    if not run.spans and meta.stage in ("runtime", "cross_trace"):
        return True
    if (
        meta.needs_judge
        and meta.axis in ("output_quality", "prompt_quality")
        and structural_sev1_seen
    ):
        return True
    return False


def build_scorecard(run: Run, graph: PipelineGraph, config: dict) -> Scorecard:
    config = config or {}
    flaws: list[Flaw] = []
    structural_sev1_seen = False
    judge_calls_used = 0

    for meta in all_checks():
        if _should_skip(meta, run, config, structural_sev1_seen):
            continue
        try:
            result = meta.fn(run, graph, config) or []
        except Exception as e:  # noqa: BLE001 - checks must never sink the run
            flaws.append(
                Flaw(
                    type="check_error",
                    severity=Severity.INFO,
                    node="",
                    description=f"check {meta.id} raised {type(e).__name__}: {e}",
                    fix=None,
                    evidence={},
                    check_id=meta.id,
                    agent_class="",
                )
            )
            continue

        for flaw in result:
            flaw.check_id = flaw.check_id or meta.id
            flaws.append(flaw)
            if flaw.severity == Severity.SEV_1 and meta.axis == "structural":
                structural_sev1_seen = True

    judge = config.get("_judge")
    if judge is not None:
        judge_calls_used = getattr(judge, "calls_used", 0)
    judge_calls_budget = getattr(judge, "max_calls", 0) if judge is not None else 0

    axis_weights = dict(DEFAULT_WEIGHTS)
    axis_weights.update(config.get("axis_weights") or {})

    axis_scores: dict[str, float] = {}
    for axis in AXES:
        axis_flaws = [f for f in flaws if _flaw_axis(f) == axis]
        penalty_sum = sum(SEVERITY_PENALTY[f.severity] for f in axis_flaws)
        axis_scores[axis] = max(0.0, 1.0 - penalty_sum)

    weight_total = sum(axis_weights[a] for a in AXES)
    overall = (
        sum(axis_scores[a] * axis_weights[a] for a in AXES) / weight_total
        if weight_total
        else 0.0
    )

    flaws_sorted = sorted(flaws, key=lambda f: -int(f.severity))

    return Scorecard(
        pipeline=graph.name,
        trace_id=run.trace_id,
        pipeline_hash=graph.pipeline_hash,
        duration_s=run.duration_s,
        axes=axis_scores,
        overall=overall,
        flaws=flaws_sorted,
        judge_calls_used=judge_calls_used,
        judge_calls_budget=judge_calls_budget,
        span_count=len(run.spans),
    )


_CHECK_AXIS_BY_ID: dict[str, str] = {}


def _flaw_axis(flaw: Flaw) -> str:
    from .registry import get_registry

    meta = get_registry().get(flaw.check_id)
    if meta is not None:
        return meta.axis
    return "structural"
