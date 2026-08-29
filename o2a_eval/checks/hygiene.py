"""Efficiency and prompt-hygiene checks: D1, D2, N3, O-R3, O-R4."""
from __future__ import annotations

import hashlib
from collections import defaultdict

from ..core.models import CONTROL_FLOW, Flaw, Severity
from ..core.registry import register


def _input_hash(span) -> str:
    return hashlib.sha256(str(span.input_value).encode()).hexdigest()[:16]


@register("D1", axis="efficiency", stage="runtime")
def redundant_execution(run, graph, config):
    """Same agent ran twice with identical input in one trace."""
    flaws: list[Flaw] = []
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for span in run.spans:
        if span.agent_class in CONTROL_FLOW:
            continue
        groups[(span.agent_name, _input_hash(span))].append(span)

    for (agent_name, input_hash), spans in groups.items():
        if len(spans) < 2:
            continue
        ordered = sorted(spans, key=lambda s: s.start_ns)
        wasted = sum(s.duration_s for s in ordered[1:])
        severity = Severity.SEV_2 if ordered[0].agent_class == "database_agent" else Severity.SEV_3
        flaws.append(
            Flaw(
                type="redundant_execution",
                severity=severity,
                node=agent_name,
                description=f"{agent_name} ran {len(spans)} times with identical input in this trace.",
                fix="Cache or dedupe this call, or move it upstream so it only runs once per trace.",
                evidence={
                    "call_count": len(spans),
                    "input_hash": input_hash,
                    "wasted_seconds": wasted,
                },
                agent_class=ordered[0].agent_class,
            )
        )
    return flaws


@register("D2", axis="efficiency", stage="runtime")
def dead_output(run, graph, config):
    """An executed node's output_key is never consumed."""
    flaws: list[Flaw] = []
    root_last_child = graph.root.sub_agent_names[-1] if graph.root.sub_agent_names else None

    seen_nodes: set[str] = set()
    for span in run.spans:
        node = graph.nodes.get(span.agent_name)
        if node is None or not node.output_key or node.name in seen_nodes:
            continue
        seen_nodes.add(node.name)

        if node.agent_class in CONTROL_FLOW:
            continue
        if node.output_key == graph.root.output_key:
            continue
        if node.name in graph.root.sub_agent_names and node.name == root_last_child:
            continue
        declared_consumers = [
            c for c in graph.consumers.get(node.output_key, []) if c != node.name
        ]
        if declared_consumers:
            continue

        downstream_checked = 0
        consumed = False
        for other in run.spans:
            if other.agent_name == node.name:
                continue
            downstream_checked += 1
            if node.output_key in other.keys_read or node.output_key in other.template_references:
                consumed = True
                break
        if consumed:
            continue

        flaws.append(
            Flaw(
                type="dead_output",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name} writes {node.output_key}, but nothing consumes it.",
                fix="Remove the write, or wire the key to a real downstream consumer.",
                evidence={
                    "output_key": node.output_key,
                    "downstream_checked": downstream_checked,
                },
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("N3", axis="prompt_quality", stage="runtime")
def undeclared_prompt_injection(run, graph, config):
    """LlmAgent prompt injects a session key not in its input_keys."""
    flaws: list[Flaw] = []
    llm_spans = run.llm_spans()

    for span in llm_spans:
        node = graph.nodes.get(span.agent_name)
        if node is None:
            continue
        used = set(span.template_references) | set(span.keys_read)
        declared = set(node.input_keys)
        undeclared = sorted(k for k in (used - declared) if k in graph.producers)
        if not undeclared:
            continue

        for key in undeclared:
            other_readers = sorted(
                {
                    s.agent_name
                    for s in llm_spans
                    if s.agent_name != span.agent_name
                    and (key in s.template_references or key in s.keys_read)
                }
            )
            flaws.append(
                Flaw(
                    type="undeclared_prompt_injection",
                    severity=Severity.SEV_2,
                    node=span.agent_name,
                    description=(
                        f"{span.agent_name} reads session key {key!r} that isn't in its "
                        "declared input_keys."
                    ),
                    fix=f"Add {key!r} to input_keys, or stop referencing it in the instruction.",
                    evidence={
                        "session_key": key,
                        "produced_by": graph.producers.get(key),
                        "declared_consumers": graph.consumers.get(key, []),
                        "actual_llm_readers": other_readers,
                        "undeclared": undeclared,
                    },
                    agent_class=node.agent_class,
                )
            )
    return flaws


@register("O-R3", axis="efficiency", stage="runtime")
def latency_budget_exceeded(run, graph, config):
    """Pipeline duration exceeds the configured latency budget."""
    budget = config.get("latency_budget_s")
    if budget is None or run.duration_s <= budget:
        return []

    leaves = [s for s in run.spans if not s.children]
    top = sorted(leaves, key=lambda s: -s.duration_s)[:3]
    total = run.duration_s or 1.0
    top_contributors = [
        {
            "agent": s.agent_name,
            "class": s.agent_class,
            "duration_s": s.duration_s,
            "pct_of_total": round(100.0 * s.duration_s / total, 2),
        }
        for s in top
    ]
    return [
        Flaw(
            type="latency_budget_exceeded",
            severity=Severity.SEV_1,
            node=graph.root.name,
            description=f"Pipeline took {run.duration_s:.1f}s, over the {budget:.1f}s budget.",
            fix="Investigate the top contributors and parallelize or shorten them.",
            evidence={
                "duration_s": run.duration_s,
                "budget_s": budget,
                "top_contributors": top_contributors,
            },
            agent_class=graph.root.agent_class,
        )
    ]


@register("O-R4", axis="efficiency", stage="runtime")
def session_bloat(run, graph, config):
    """Session key count at pipeline end exceeds threshold."""
    threshold = config.get("session_key_threshold", 25)
    ordered = run.ordered()
    if not ordered:
        return []
    final_span = ordered[-1]
    keys = final_span.keys_after or final_span.keys_before
    if len(keys) <= threshold:
        return []
    return [
        Flaw(
            type="session_bloat",
            severity=Severity.SEV_3,
            node=final_span.agent_name,
            description=f"Session holds {len(keys)} keys at pipeline end (threshold {threshold}).",
            fix="Prune session keys that are no longer needed downstream.",
            evidence={
                "final_key_count": len(keys),
                "threshold": threshold,
                "keys": sorted(keys),
            },
            agent_class=final_span.agent_class,
        )
    ]
