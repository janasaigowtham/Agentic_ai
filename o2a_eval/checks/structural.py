"""Structural checks: D9, D5, Q-R2, D6, T-S7."""
from __future__ import annotations

from ..core.graph import extract_template_keys
from ..core.models import Flaw, Severity
from ..core.registry import register

TRANSFORM_CLASSES = {"slv_transformation_agent", "transformation_agent"}


def _conditional_targets(graph) -> set[str]:
    conditional: set[str] = set()
    for node in graph.nodes.values():
        for route in node.routes:
            if route.target_agent:
                conditional.add(route.target_agent)
                conditional.update(n.name for n in graph.downstream_of(route.target_agent))
    return conditional


@register("D9", axis="structural", stage="runtime")
def structural_drift(run, graph, config):
    """Every declared node fired; no undeclared span appeared."""
    declared = {n.name for n in graph.nodes.values() if n.agent_class != "<missing>"}
    executed = {s.agent_name for s in run.spans}
    conditional = _conditional_targets(graph)

    missing = sorted((declared - executed) - conditional)
    unexpected = sorted(executed - declared - {""})

    flaws: list[Flaw] = []
    if missing:
        flaws.append(
            Flaw(
                type="structural_drift",
                severity=Severity.SEV_1,
                node=graph.root.name,
                description=f"{len(missing)} declared node(s) never fired: {', '.join(missing)}.",
                fix="Check for a routing bug or a silently skipped step.",
                evidence={"missing": missing, "unexpected": unexpected},
                agent_class=graph.root.agent_class,
            )
        )
    if unexpected:
        flaws.append(
            Flaw(
                type="structural_drift",
                severity=Severity.SEV_2,
                node=graph.root.name,
                description=f"{len(unexpected)} undeclared span(s) appeared: {', '.join(unexpected)}.",
                fix="Add the node to the pipeline YAML, or confirm it should not run.",
                evidence={"missing": missing, "unexpected": unexpected},
                agent_class=graph.root.agent_class,
            )
        )
    return flaws


@register("D5", axis="structural", stage="runtime")
def sequential_order(run, graph, config):
    """SequentialAgent children all present, in declared order."""
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != "SequentialAgent" or not span.children:
            continue
        node = graph.nodes.get(span.agent_name)
        if node is None:
            continue
        declared_order = list(node.sub_agent_names)
        actual_order_raw = [c.agent_name for c in span.children]

        actual_order: list[str] = []
        for name in actual_order_raw:
            if name not in actual_order:
                actual_order.append(name)

        missing = [n for n in declared_order if n not in actual_order]
        for m in missing:
            flaws.append(
                Flaw(
                    type="sequential_missing",
                    severity=Severity.SEV_1,
                    node=span.agent_name,
                    description=f"{span.agent_name} declares child {m!r} but it never ran.",
                    fix="Check the sequential agent's step list against runtime behaviour.",
                    evidence={"declared": declared_order, "actual": actual_order, "missing": missing},
                    agent_class=span.agent_class,
                )
            )

        declared_intersection = [n for n in declared_order if n in actual_order]
        actual_intersection = [n for n in actual_order if n in declared_order]
        if declared_intersection != actual_intersection:
            flaws.append(
                Flaw(
                    type="sequential_reorder",
                    severity=Severity.SEV_2,
                    node=span.agent_name,
                    description=f"{span.agent_name}'s children ran out of declared order.",
                    fix="Fix the runner's execution order or update the YAML to match.",
                    evidence={
                        "declared_order": declared_intersection,
                        "actual_order": actual_intersection,
                    },
                    agent_class=span.agent_class,
                )
            )
    return flaws


@register("Q-R2", axis="structural", stage="runtime")
def error_swallowed(run, graph, config):
    """A child errored but later siblings still ran."""
    flaws: list[Flaw] = []
    for span in run.spans:
        if not span.children:
            continue
        for i, child in enumerate(span.children):
            if not child.errored:
                continue
            later_siblings = [c.agent_name for c in span.children[i + 1 :]]
            if not later_siblings:
                continue
            flaws.append(
                Flaw(
                    type="error_swallowed",
                    severity=Severity.SEV_1,
                    node=span.agent_name,
                    description=(
                        f"{child.agent_name} errored under {span.agent_name}, but "
                        f"{len(later_siblings)} later sibling(s) still ran."
                    ),
                    fix="Stop the sequence on error, or confirm the later siblings tolerate it.",
                    evidence={
                        "parent": span.agent_name,
                        "errored_child": child.agent_name,
                        "later_siblings": later_siblings,
                    },
                    agent_class=span.agent_class,
                )
            )
    return flaws


@register("D6", axis="structural", stage="runtime")
def transform_ref_unresolved(run, graph, config):
    """A transform's {{ key }} refs were absent from session when it ran."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if node.agent_class not in TRANSFORM_CLASSES or not node.transform:
            continue
        refs = extract_template_keys(node.transform) - {"item"}
        if not refs:
            continue
        for span in run.spans_for(node.name):
            if not span.keys_before:
                continue
            unresolved = sorted(refs - set(span.keys_before))
            if not unresolved:
                continue
            severity = Severity.SEV_1 if not node.strict else Severity.SEV_2
            flaws.append(
                Flaw(
                    type="transform_ref_unresolved",
                    severity=severity,
                    node=node.name,
                    description=(
                        f"{node.name}'s transform references {unresolved} which weren't in "
                        "session when it ran."
                    ),
                    fix="Ensure the producing node runs first, or add the key to input_keys.",
                    evidence={
                        "unresolved": unresolved,
                        "available": sorted(span.keys_before),
                        "strict": node.strict,
                    },
                    agent_class=node.agent_class,
                )
            )
    return flaws


@register("T-S7", axis="structural", stage="static")
def strict_false_audit(run, graph, config):
    """strict=False audit — silences errors."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if node.agent_class not in TRANSFORM_CLASSES or node.strict:
            continue
        flaws.append(
            Flaw(
                type="strict_disabled",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name} sets strict: false, silencing missing-key errors.",
                fix="Set strict: true unless silent nulls are intentional here.",
                evidence={"source_path": node.source_path},
                agent_class=node.agent_class,
            )
        )
    return flaws
