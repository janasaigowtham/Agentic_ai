"""Minimal declarative agent execution engine.

Reads a tree of Node objects (parsed from pipeline.yaml) and executes it
against a shared session dict, emitting one trace span per step. The engine
only knows how to run six agent_class values - it has no idea this
particular pipeline is about customer support. See course/05-multi-agent-
orchestration.md for why keeping the engine generic like this matters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tracing import Tracer

Session = dict[str, Any]

CONTROL_FLOW_CLASSES = {"sequential_agent", "router_agent", "orchestrator"}


class UnknownAgentClass(RuntimeError):
    pass


class GateRejected(RuntimeError):
    pass


@dataclass
class Node:
    name: str
    agent_class: str
    output_key: str | None = None
    input_keys: list[str] = field(default_factory=list)
    instruction: str | None = None
    tool: str | None = None
    transform: str | None = None
    routes: list[dict] = field(default_factory=list)
    sub_agents: list["Node"] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: dict) -> "Node":
        children = [cls.from_dict(c) for c in spec.get("sub_agents", [])]
        return cls(
            name=spec["name"],
            agent_class=spec["agent_class"],
            output_key=spec.get("output_key"),
            input_keys=list(spec.get("input_keys", [])),
            instruction=spec.get("instruction"),
            tool=spec.get("tool"),
            transform=spec.get("transform"),
            routes=spec.get("routes", []),
            sub_agents=children,
            raw=spec,
        )


class Engine:
    """Executes a Node tree against a session dict, emitting trace spans."""

    def __init__(
        self,
        llm,
        tools: dict[str, Callable],
        transforms: dict[str, Callable],
        tracer: Tracer,
        approve_gate: Callable[[Node, Session], bool] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.transforms = transforms
        self.tracer = tracer
        self.approve_gate = approve_gate or (lambda node, session: True)

    def run(self, node: Node, session: Session) -> Session:
        handler = getattr(self, f"_run_{node.agent_class}", None)
        if handler is None:
            raise UnknownAgentClass(node.agent_class)
        keys_before = sorted(session.keys())
        span = self.tracer.start(node, keys_before)
        try:
            handler(node, session)
        except Exception as exc:
            self.tracer.error(span, exc)
            raise
        finally:
            self.tracer.end(span, sorted(session.keys()))
        return session

    def _run_llm_agent(self, node: Node, session: Session) -> None:
        prompt = (node.instruction or "").format(**session)
        result = self.llm.complete(prompt, session=session)
        if node.output_key:
            session[node.output_key] = result

    def _run_tool_agent(self, node: Node, session: Session) -> None:
        fn = self.tools[node.tool]
        args = {k: session.get(k) for k in node.input_keys}
        result = fn(**args)
        if node.output_key:
            session[node.output_key] = result

    def _run_transform_agent(self, node: Node, session: Session) -> None:
        fn = self.transforms[node.transform]
        args = {k: session.get(k) for k in node.input_keys}
        result = fn(**args)
        if node.output_key:
            session[node.output_key] = result

    def _run_sequential_agent(self, node: Node, session: Session) -> None:
        for child in node.sub_agents:
            self.run(child, session)

    def _run_orchestrator(self, node: Node, session: Session) -> None:
        self._run_sequential_agent(node, session)

    def _run_router_agent(self, node: Node, session: Session) -> None:
        chosen = None
        for route in sorted(node.routes, key=lambda r: r.get("priority", 0)):
            if self._matches(route.get("conditions", []), session):
                chosen = route["target_agent"]
                break
        session[node.output_key or "chosen_target"] = chosen
        target = next((c for c in node.sub_agents if c.name == chosen), None)
        if target is not None:
            self.run(target, session)

    def _run_gate_agent(self, node: Node, session: Session) -> None:
        start = time.monotonic()
        approved = self.approve_gate(node, session)
        session[node.output_key or f"{node.name}_approved"] = approved
        session[f"{node.name}_wait_s"] = time.monotonic() - start
        if not approved:
            raise GateRejected(f"gate '{node.name}' did not approve")

    @staticmethod
    def _matches(conditions: list[dict], session: Session) -> bool:
        if not conditions:
            return True
        for cond in conditions:
            value = session.get(cond["context_key"])
            op = cond.get("operator", "eq")
            target = cond.get("value")
            ok = {
                "eq": value == target,
                "neq": value != target,
                "gt": value is not None and value > target,
                "lt": value is not None and value < target,
                "not_null": value is not None,
            }.get(op, False)
            if not ok:
                return False
        return True
