"""Routing checks: R-S1, R-S2, R-S3, R-S5, R-S7, R-R1."""
from __future__ import annotations

from collections import Counter

from ..core.models import Flaw, Severity
from ..core.registry import register

ROUTER_CLASS = "decision_router_agent"


def _routers(graph):
    return [n for n in graph.nodes.values() if n.agent_class == ROUTER_CLASS]


@register("R-S1", axis="routing", stage="static")
def router_unknown_context_key(run, graph, config):
    """A context_key no agent produces."""
    flaws: list[Flaw] = []
    for node in _routers(graph):
        context_keys = sorted({k for route in node.routes for k in route.context_keys})
        missing = [k for k in context_keys if k not in graph.producers]
        if not missing:
            continue
        flaws.append(
            Flaw(
                type="router_unknown_context_key",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name} routes on {missing} which nothing produces.",
                fix="Fix the context_key name, or add the producing node.",
                evidence={"missing_keys": missing, "known_producers": sorted(graph.producers)},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("R-S2", axis="routing", stage="static")
def router_invalid_target(run, graph, config):
    """target_agent missing or not in sub_agents."""
    flaws: list[Flaw] = []
    for node in _routers(graph):
        for route in node.routes:
            target = route.target_agent
            target_node = graph.nodes.get(target)
            if target_node is None or target_node.agent_class == "<missing>":
                flaws.append(
                    Flaw(
                        type="router_invalid_target",
                        severity=Severity.SEV_1,
                        node=node.name,
                        description=f"{node.name} routes to {target!r} which has no YAML file.",
                        fix="Create the target agent's YAML, or fix the target_agent name.",
                        evidence={
                            "target": target,
                            "priority": route.priority,
                            "sub_agents": node.sub_agent_names,
                        },
                        agent_class=node.agent_class,
                    )
                )
                continue
            if node.sub_agent_names and target not in node.sub_agent_names:
                flaws.append(
                    Flaw(
                        type="router_target_not_declared",
                        severity=Severity.SEV_2,
                        node=node.name,
                        description=f"{node.name} routes to {target!r} which isn't in its sub_agents.",
                        fix="Add the target to sub_agents so it's declared reachable.",
                        evidence={
                            "target": target,
                            "priority": route.priority,
                            "sub_agents": node.sub_agent_names,
                        },
                        agent_class=node.agent_class,
                    )
                )
    return flaws


@register("R-S3", axis="routing", stage="static")
def router_not_exhaustive(run, graph, config):
    """eq-only routes leave null/unexpected values unmatched."""
    flaws: list[Flaw] = []
    for node in _routers(graph):
        if not node.routes:
            continue
        operators = [
            c.get("operator") for route in node.routes for c in route.conditions
        ]
        if not operators or any(op != "eq" for op in operators):
            continue
        has_catch_all = any(not route.conditions for route in node.routes)
        if has_catch_all:
            continue
        context_keys = sorted({k for route in node.routes for k in route.context_keys})
        covered_values = sorted(
            {
                c.get("value")
                for route in node.routes
                for c in route.conditions
                if c.get("operator") == "eq"
            },
            key=str,
        )
        flaws.append(
            Flaw(
                type="router_not_exhaustive",
                severity=Severity.SEV_2,
                node=node.name,
                description=(
                    f"{node.name} only matches eq routes with no catch-all; "
                    "unexpected values fall through unmatched."
                ),
                fix="Add a catch-all route, or a neq/not_null default branch.",
                evidence={"context_keys": context_keys, "covered_values": covered_values},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("R-S5", axis="routing", stage="static")
def router_duplicate_priority(run, graph, config):
    """Two routes share a priority."""
    flaws: list[Flaw] = []
    for node in _routers(graph):
        counts = Counter(route.priority for route in node.routes)
        duplicates = {p: c for p, c in counts.items() if c > 1}
        if not duplicates:
            continue
        flaws.append(
            Flaw(
                type="router_duplicate_priority",
                severity=Severity.SEV_2,
                node=node.name,
                description=f"{node.name} has routes sharing priorities {sorted(duplicates)}.",
                fix="Give each route a distinct priority so evaluation order is unambiguous.",
                evidence={
                    "duplicate_priorities": duplicates,
                    "routes": [r.target_agent for r in node.routes],
                },
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("R-S7", axis="routing", stage="static")
def router_orphan_sub_agent(run, graph, config):
    """A sub_agents entry no route targets."""
    flaws: list[Flaw] = []
    for node in _routers(graph):
        routed_targets = {route.target_agent for route in node.routes}
        orphans = sorted(set(node.sub_agent_names) - routed_targets)
        if not orphans:
            continue
        flaws.append(
            Flaw(
                type="router_orphan_sub_agent",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name} declares {orphans} in sub_agents but no route targets them.",
                fix="Remove the orphaned sub_agent, or add a route to it.",
                evidence={"orphans": orphans, "routed_targets": sorted(routed_targets)},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("R-R1", axis="routing", stage="runtime")
def router_misdecision(run, graph, config):
    """Router chose the wrong branch, or no branch matched."""
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != ROUTER_CLASS:
            continue
        evaluated = span.evaluated_routes
        if not evaluated or not any("matched" in r for r in evaluated):
            continue
        chosen = span.chosen_target
        matched = [r for r in evaluated if r.get("matched")]

        if not matched:
            flaws.append(
                Flaw(
                    type="router_no_match",
                    severity=Severity.SEV_1,
                    node=span.agent_name,
                    description=f"{span.agent_name} evaluated its routes but none matched.",
                    fix="Add a catch-all route, or fix the condition that should have matched.",
                    evidence={"chosen": chosen, "evaluated_routes": evaluated},
                    agent_class=span.agent_class,
                )
            )
            continue

        expected = min(matched, key=lambda r: r.get("priority", 0))
        if len(matched) > 1:
            if chosen != expected.get("target"):
                flaws.append(
                    Flaw(
                        type="router_misdecision",
                        severity=Severity.SEV_1,
                        node=span.agent_name,
                        description=(
                            f"{span.agent_name} chose {chosen!r} but the lowest-priority "
                            f"match was {expected.get('target')!r}."
                        ),
                        fix="Check the router's priority-resolution logic.",
                        evidence={
                            "chosen": chosen,
                            "expected": expected.get("target"),
                            "matched_routes": matched,
                            "evaluated_routes": evaluated,
                        },
                        agent_class=span.agent_class,
                    )
                )
            else:
                flaws.append(
                    Flaw(
                        type="router_multi_match",
                        severity=Severity.SEV_3,
                        node=span.agent_name,
                        description=f"{span.agent_name} had more than one matching route.",
                        fix="Tighten the route conditions so only one route can match.",
                        evidence={
                            "chosen": chosen,
                            "expected": expected.get("target"),
                            "matched_routes": matched,
                            "evaluated_routes": evaluated,
                        },
                        agent_class=span.agent_class,
                    )
                )
        else:
            only = matched[0]
            if chosen != only.get("target"):
                flaws.append(
                    Flaw(
                        type="router_misdecision",
                        severity=Severity.SEV_1,
                        node=span.agent_name,
                        description=(
                            f"{span.agent_name} chose {chosen!r} but {only.get('target')!r} "
                            "was the only match."
                        ),
                        fix="Check the router's target-selection logic.",
                        evidence={
                            "chosen": chosen,
                            "expected": only.get("target"),
                            "matched_routes": matched,
                            "evaluated_routes": evaluated,
                        },
                        agent_class=span.agent_class,
                    )
                )
    return flaws
