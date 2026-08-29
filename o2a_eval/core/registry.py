"""Check registry: @register decorator and CheckMeta bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Flaw, PipelineGraph, Run

AXES = ("structural", "efficiency", "routing", "prompt_quality", "output_quality")
STAGES = ("static", "runtime", "cross_trace")

CheckFn = Callable[[Run, PipelineGraph, dict], list]


@dataclass
class CheckMeta:
    id: str
    name: str
    axis: str
    stage: str
    fn: CheckFn
    needs_judge: bool
    description: str


_REGISTRY: dict[str, CheckMeta] = {}


def register(check_id: str, *, axis: str, stage: str, needs_judge: bool = False):
    if axis not in AXES:
        raise ValueError(f"invalid axis {axis!r}; must be one of {AXES}")
    if stage not in STAGES:
        raise ValueError(f"invalid stage {stage!r}; must be one of {STAGES}")

    def _decorator(fn: CheckFn) -> CheckFn:
        doc = (fn.__doc__ or "").strip()
        description = doc.splitlines()[0].strip() if doc else ""
        _REGISTRY[check_id] = CheckMeta(
            id=check_id,
            name=fn.__name__,
            axis=axis,
            stage=stage,
            fn=fn,
            needs_judge=needs_judge,
            description=description,
        )
        return fn

    return _decorator


def all_checks() -> list[CheckMeta]:
    stage_order = {"static": 0, "runtime": 1, "cross_trace": 2}
    return sorted(_REGISTRY.values(), key=lambda c: (stage_order[c.stage], c.id))


def get_registry() -> dict[str, CheckMeta]:
    return _REGISTRY
