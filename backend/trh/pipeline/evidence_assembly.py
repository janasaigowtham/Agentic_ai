"""Evidence Assembly: plain code, forms no opinion, runs before Orient ever sees a step.

Computes per-step facts that specialists read instead of raw step data:
retrieved_columns/row_count from a step's own output, resolved_query with params
left symbolic (never bound to a literal input value -- the PII discipline from
TRAJECTORY_REVIEW_HARNESS_SPEC.md), the declared-vs-referenced cross-reference
(referenced_columns/unused_columns, undeclared_template_refs) against downstream
steps' declared input_keys/instruction templates, and duplicate-call detection.

Every fact here is arithmetic or string/set comparison -- no judgment about
whether any of it is a problem. That's Stage 02's job.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from trh.core.graph import extract_sql_params, extract_template_keys
from trh.core.harness_models import Step, Trajectory
from trh.core.models import PipelineGraph


@dataclass
class EvidenceFacts:
    # producer-side: populated for steps whose output is structured data
    retrieved_columns: list[str] = field(default_factory=list)
    row_count: int | None = None
    resolved_query: str | None = None
    query_params: list[str] = field(default_factory=list)

    # consumer-side: populated for steps with a declared instruction template
    declared_input_keys: list[str] = field(default_factory=list)
    template_referenced_keys: list[str] = field(default_factory=list)
    undeclared_template_refs: list[str] = field(default_factory=list)

    # cross-step: this step's own retrieved_columns against every downstream
    # step's declared instruction text / input_keys
    referenced_columns: list[str] = field(default_factory=list)
    unused_columns: list[str] = field(default_factory=list)

    # duplicate-call detection: earlier step_ids with the same agent_name and
    # a structurally identical input
    duplicate_of_step_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_columns": self.retrieved_columns,
            "row_count": self.row_count,
            "resolved_query": self.resolved_query,
            "query_params": self.query_params,
            "declared_input_keys": self.declared_input_keys,
            "template_referenced_keys": self.template_referenced_keys,
            "undeclared_template_refs": self.undeclared_template_refs,
            "referenced_columns": self.referenced_columns,
            "unused_columns": self.unused_columns,
            "duplicate_of_step_ids": self.duplicate_of_step_ids,
        }


def _rows_from_output(output: Any) -> list[dict] | None:
    if isinstance(output, list) and output and all(isinstance(r, dict) for r in output):
        return output
    return None


def _column_names(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _duplicate_of(step: Step, all_steps: list[Step]) -> list[str]:
    if step.input is None:
        return []
    canon = _canonical(step.input)
    return [
        other.step_id
        for other in all_steps
        if other.step_id != step.step_id
        and other.agent_name == step.agent_name
        and other.input is not None
        and other.started_at < step.started_at
        and _canonical(other.input) == canon
    ]


def _downstream_steps(step: Step, all_steps: list[Step]) -> list[Step]:
    return [s for s in all_steps if s.step_id != step.step_id and s.started_at >= step.completed_at]


def assemble_evidence(
    trajectory: Trajectory, graph: PipelineGraph | None = None
) -> dict[str, EvidenceFacts]:
    """Computes evidence facts for every step and writes them onto Step.evidence.

    Returns the typed dataclass form keyed by step_id as well, for callers that
    want structured access rather than the plain-dict copy stored on the step.
    """
    facts_by_step: dict[str, EvidenceFacts] = {}
    steps = trajectory.steps

    for step in steps:
        facts = EvidenceFacts()
        node = graph.nodes.get(step.agent_name) if graph else None

        rows = _rows_from_output(step.output)
        if rows is not None:
            facts.retrieved_columns = _column_names(rows)
            facts.row_count = len(rows)
        elif step.raw_span is not None and step.raw_span.row_count is not None:
            facts.row_count = step.raw_span.row_count

        if node is not None and node.query:
            facts.resolved_query = node.query
            facts.query_params = sorted(extract_sql_params(node.query))

        if node is not None:
            facts.declared_input_keys = list(node.input_keys)
            facts.template_referenced_keys = sorted(extract_template_keys(node.instruction))
            facts.undeclared_template_refs = sorted(
                set(facts.template_referenced_keys) - set(facts.declared_input_keys)
            )

        if facts.retrieved_columns and graph is not None:
            downstream_texts: list[str] = []
            for other in _downstream_steps(step, steps):
                other_node = graph.nodes.get(other.agent_name)
                if other_node is None:
                    continue
                downstream_texts.append(other_node.instruction or "")
                downstream_texts.extend(other_node.input_keys)
            referenced = [c for c in facts.retrieved_columns if any(c in text for text in downstream_texts)]
            facts.referenced_columns = referenced
            facts.unused_columns = [c for c in facts.retrieved_columns if c not in referenced]

        facts.duplicate_of_step_ids = _duplicate_of(step, steps)

        facts_by_step[step.step_id] = facts
        step.evidence = facts.to_dict()

    return facts_by_step
