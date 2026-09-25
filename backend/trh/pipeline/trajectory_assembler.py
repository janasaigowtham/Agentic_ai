"""Trajectory Assembler: plain code, drains tapped spans into Trajectory/Step rows.

Per TRAJECTORY_REVIEW_HARNESS_SPEC.md section 5 ("a background consumer... writes
Trajectory + Step rows") this is pure control flow -- it reshapes captured spans
into the harness's own Trajectory/Step contract (spec section 4.1). It forms no
opinion about anything it moves; that starts at Stage 01 (Orient).
"""
from __future__ import annotations

import json
from typing import Any

from trh.capture.simulate import CapturedSpan
from trh.core.harness_models import Step, Trajectory, step_type_for
from trh.core.models import PipelineGraph


def _declared_contract_ref(agent_name: str, graph: PipelineGraph | None) -> str | None:
    if graph is None:
        return None
    node = graph.nodes.get(agent_name)
    return node.source_path if node else None


def _decode(raw: Any) -> Any:
    """input.value / output.value are stored as JSON text; decode for real
    downstream use instead of leaving every consumer to re-parse it."""
    if raw is None or isinstance(raw, (dict, list, int, float, bool)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def assemble_trajectory(
    trajectory_id: str,
    source_pipeline: str,
    captured: list[CapturedSpan],
    graph: PipelineGraph | None = None,
) -> Trajectory:
    ordered = sorted(captured, key=lambda c: c.span.start_ns)
    known_ids = {c.span.span_id for c in ordered}

    steps: list[Step] = []
    for c in ordered:
        span = c.span
        parent_step_id = span.parent_id if span.parent_id in known_ids else None
        steps.append(
            Step(
                step_id=span.span_id,
                step_type=step_type_for(span.agent_class),
                agent_name=span.agent_name,
                agent_class=span.agent_class,
                input=_decode(span.input_value),
                output=_decode(span.output_value),
                started_at=span.start_ns / 1e9,
                completed_at=span.end_ns / 1e9,
                parent_step_id=parent_step_id,
                declared_contract_ref=_declared_contract_ref(span.agent_name, graph),
                tap=c.tap,
                raw_span=span,
            )
        )

    started_at = min((s.started_at for s in steps), default=0.0)
    completed_at = max((s.completed_at for s in steps), default=None)
    outcome = steps[-1].output if steps else None

    source_spec_ref = None
    if graph is not None:
        source_spec_ref = graph.root.source_path or graph.name

    return Trajectory(
        trajectory_id=trajectory_id,
        source_pipeline=source_pipeline,
        source_spec_ref=source_spec_ref,
        steps=steps,
        started_at=started_at,
        completed_at=completed_at,
        outcome=outcome,
    )
