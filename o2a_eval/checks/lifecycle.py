"""Lifecycle/wiring checks: G-S1, G-R1, G-R2, O-S1, O-S3, Q-S1, Q-S3."""
from __future__ import annotations

from collections import Counter

from ..core.models import Flaw, Severity
from ..core.registry import register

GATE_CLASS = "agent_gate"


@register("G-S1", axis="structural", stage="static")
def gate_unknown_input(run, graph, config):
    """Gate's input_key not produced upstream."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if node.agent_class != GATE_CLASS:
            continue
        missing = [k for k in node.input_keys if k not in graph.producers]
        if not missing:
            continue
        flaws.append(
            Flaw(
                type="gate_unknown_input",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name} waits on {missing} which nothing produces.",
                fix="Fix the input_key name, or add the producing node upstream.",
                evidence={"missing": missing},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("G-R1", axis="structural", stage="runtime")
def gate_never_paused(run, graph, config):
    """Gate span shorter than gate_min_pause_s — it isn't gating."""
    threshold = config.get("gate_min_pause_s", 0.5)
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != GATE_CLASS:
            continue
        if span.duration_s >= threshold:
            continue
        flaws.append(
            Flaw(
                type="gate_never_paused",
                severity=Severity.SEV_3,
                node=span.agent_name,
                description=f"{span.agent_name} completed in {span.duration_s:.3f}s — it never actually paused.",
                fix="Confirm the gate is wired to a real external resume signal.",
                evidence={"duration_s": span.duration_s, "threshold_s": threshold},
                agent_class=span.agent_class,
            )
        )
    return flaws


@register("G-R2", axis="structural", stage="runtime")
def gate_waited_too_long(run, graph, config):
    """Gate waited past ceiling."""
    ceiling = config.get("gate_max_wait_s", 86400)
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != GATE_CLASS:
            continue
        if span.duration_s <= ceiling:
            continue
        flaws.append(
            Flaw(
                type="gate_waited_too_long",
                severity=Severity.SEV_2,
                node=span.agent_name,
                description=f"{span.agent_name} waited {span.duration_s:.0f}s, over the {ceiling:.0f}s ceiling.",
                fix="Add a timeout/escalation path for this gate.",
                evidence={"wait_s": span.duration_s, "ceiling_s": ceiling},
                agent_class=span.agent_class,
            )
        )
    return flaws


@register("O-S1", axis="structural", stage="static")
def consumer_before_producer(run, graph, config):
    """A consumer appears before its producer in declared walk order."""
    position: dict[str, int] = {}
    for i, node in enumerate(graph.root.walk()):
        position.setdefault(node.name, i)

    flaws: list[Flaw] = []
    for key, producer in graph.producers.items():
        producer_pos = position.get(producer)
        if producer_pos is None:
            continue
        for consumer in graph.consumers.get(key, []):
            if consumer == producer:
                continue
            consumer_pos = position.get(consumer)
            if consumer_pos is None or consumer_pos >= producer_pos:
                continue
            flaws.append(
                Flaw(
                    type="consumer_before_producer",
                    severity=Severity.SEV_1,
                    node=consumer,
                    description=(
                        f"{consumer} consumes {key!r} but is declared before its "
                        f"producer {producer!r}."
                    ),
                    fix="Reorder sub_agents so the producer runs before this consumer.",
                    evidence={
                        "session_key": key,
                        "producer": producer,
                        "consumer_position": consumer_pos,
                        "producer_position": producer_pos,
                    },
                    agent_class=graph.nodes[consumer].agent_class if consumer in graph.nodes else "",
                )
            )
    return flaws


@register("O-S3", axis="structural", stage="static")
def pipeline_output_unproduced(run, graph, config):
    """Pipeline output_key nothing produces."""
    output_key = graph.root.output_key
    if not output_key or output_key in graph.producers:
        return []
    return [
        Flaw(
            type="pipeline_output_unproduced",
            severity=Severity.SEV_1,
            node=graph.root.name,
            description=f"Pipeline declares output_key {output_key!r} but nothing produces it.",
            fix="Point output_key at a real producer, or add one.",
            evidence={"declared_output_key": output_key, "available_keys": sorted(graph.producers)},
            agent_class=graph.root.agent_class,
        )
    ]


@register("Q-S1", axis="structural", stage="static")
def unresolved_reference(run, graph, config):
    """sub_agents references a name with no YAML file."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        if node.agent_class != "<missing>":
            continue
        referenced_by = sorted(
            n.name for n in graph.nodes.values() if node.name in n.sub_agent_names
        )
        flaws.append(
            Flaw(
                type="unresolved_reference",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name!r} is referenced in sub_agents but has no YAML file.",
                fix=f"Create {node.name}.yaml, or remove the reference.",
                evidence={"referenced_by": referenced_by},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("Q-S3", axis="structural", stage="static")
def duplicate_sub_agent(run, graph, config):
    """Same child listed twice in one sub_agents."""
    flaws: list[Flaw] = []
    for node in graph.nodes.values():
        counts = Counter(node.sub_agent_names)
        duplicates = sorted(name for name, c in counts.items() if c > 1)
        if not duplicates:
            continue
        flaws.append(
            Flaw(
                type="duplicate_sub_agent",
                severity=Severity.SEV_2,
                node=node.name,
                description=f"{node.name} lists {duplicates} more than once in sub_agents.",
                fix="Remove the duplicate entry.",
                evidence={"duplicates": duplicates, "sub_agents": node.sub_agent_names},
                agent_class=node.agent_class,
            )
        )
    return flaws
