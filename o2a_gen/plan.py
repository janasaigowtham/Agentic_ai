"""Structured procedure -> tree of agents (names, classes, session keys, wiring).

This step is deterministic: no model calls. Every agent_class comes from the
extracted procedure, which chose it from the agent syntax document. The fields of
each agent are filled in afterwards by ground.py, from that class's syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from o2a_gen.config import GenConfig
from o2a_gen.procedure import Procedure, Step

@dataclass
class AgentNode:
    name: str
    agent_class: str
    description: str = ""
    step: Step | None = None
    children: list["AgentNode"] = field(default_factory=list)
    output_key: str = ""
    required_keys: list[str] = field(default_factory=list)   # from the step's `uses`
    available_keys: list[str] = field(default_factory=list)  # visible when this agent runs
    branch_targets: list[tuple[str, "AgentNode"]] = field(default_factory=list)  # routers
    fields: dict = field(default_factory=dict)               # filled by grounding
    role: str = "step"                                       # step | router | group | orchestrator

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class Plan:
    root: AgentNode
    by_step: dict[str, AgentNode]
    inputs: list[str]

    def nodes(self) -> list[AgentNode]:
        return list(self.root.walk())


def snake(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len].rstrip("_") or "step"


class _Names:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.used: set[str] = set()

    def take(self, text: str) -> str:
        base = snake(text)
        if self.prefix and not base.startswith(self.prefix + "_") and base != self.prefix:
            base = f"{self.prefix}_{base}"
        name, i = base, 2
        while name in self.used:
            name, i = f"{base}_{i}", i + 1
        self.used.add(name)
        return name


def build_plan(proc: Procedure, cfg: GenConfig) -> Plan:
    prefix = cfg.prefix or snake(proc.name, 20)
    names, keys = _Names(prefix), _Names(prefix)
    names.used.add(f"{prefix}_pipeline")
    router_key = str(cfg.generation.get("router_output_key", "routing_decision"))
    inputs = list(dict.fromkeys(cfg.pipeline_inputs + proc.inputs))
    step_key: dict[str, str] = {}
    by_step: dict[str, AgentNode] = {}

    def build_steps(steps: list[Step], visible: list[str],
                    shared: dict[str, str] | None = None) -> tuple[list[AgentNode], list[str]]:
        """``shared`` maps a `produces` phrase to the key already used for it in a
        sibling branch, so mutually exclusive branches write the same session key."""
        nodes: list[AgentNode] = []
        visible = list(visible)
        for s in steps:
            required = [step_key.get(u, u) for u in s.uses]
            if s.branches:
                title = s.title if re.search(r"router$", snake(s.title)) else f"{s.title} router"
                node = AgentNode(names.take(title), s.agent_class, s.text, s,
                                 output_key=router_key, available_keys=list(visible),
                                 required_keys=[step_key.get(s.depends_on, s.depends_on)],
                                 role="router")
                produced: list[str] = []
                branch_keys: dict[str, str] = {}
                for b in s.branches:
                    b_nodes, b_visible = build_steps(b.steps, visible, branch_keys)
                    target = b_nodes[0] if len(b_nodes) == 1 else AgentNode(
                        names.take(f"{b.label} process"), group_class,
                        f"Branch '{b.label}': {b.when}", children=b_nodes, role="group",
                        available_keys=list(visible))
                    node.branch_targets.append((b.label, target))
                    node.children.append(target)
                    produced += [k for k in b_visible if k not in visible]
                by_step[s.id] = node
                nodes.append(node)
                visible += list(dict.fromkeys(produced))
                continue
            phrase = snake(s.produces or s.title)
            if shared is not None and phrase in shared and shared[phrase] not in visible:
                key = shared[phrase]
            else:
                key = keys.take(s.produces or s.title)
                if shared is not None:
                    shared.setdefault(phrase, key)
            step_key[s.id] = key
            node = AgentNode(names.take(s.title), s.agent_class, s.text, s,
                             output_key=key, required_keys=required,
                             available_keys=list(visible))
            by_step[s.id] = node
            nodes.append(node)
            visible.append(key)
        return nodes, visible

    group_class = proc.group_class
    root = AgentNode(f"{prefix}_pipeline", proc.orchestrator_class, proc.description,
                     output_key=f"{prefix}_final_output", role="orchestrator",
                     available_keys=list(inputs))
    visible = list(inputs)
    for phase in proc.phases:
        before = list(visible)
        nodes, visible = build_steps(phase.steps, visible)
        # A single router, or a single step the procedure says the orchestrator runs
        # itself, sits directly under the root; any other phase is wrapped in a group.
        if len(nodes) == 1 and (phase.direct or nodes[0].role == "router"):
            root.children.append(nodes[0])
        elif nodes:
            root.children.append(AgentNode(names.take(phase.title), group_class, phase.title,
                                           children=nodes, role="group",
                                           available_keys=before))
    return Plan(root, by_step, inputs)
