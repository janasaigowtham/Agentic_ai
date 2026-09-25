"""Core dataclasses for the o2a-eval evaluator. No external dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator


class Severity(IntEnum):
    INFO = 0
    SEV_3 = 1
    SEV_2 = 2
    SEV_1 = 3

    @property
    def label(self) -> str:
        return {
            Severity.INFO: "INFO",
            Severity.SEV_3: "SEV-3",
            Severity.SEV_2: "SEV-2",
            Severity.SEV_1: "SEV-1",
        }[self]

    @classmethod
    def parse(cls, s: str) -> "Severity":
        s = s.strip().upper().replace("_", "-")
        mapping = {
            "INFO": cls.INFO,
            "SEV-3": cls.SEV_3,
            "SEV-2": cls.SEV_2,
            "SEV-1": cls.SEV_1,
        }
        if s not in mapping:
            raise ValueError(f"unknown severity: {s!r}")
        return mapping[s]


@dataclass
class Flaw:
    type: str
    severity: Severity
    node: str
    description: str
    fix: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    check_id: str = ""
    agent_class: str = ""

    def to_dict(self, include_evidence: bool = True) -> dict:
        d = {
            "type": self.type,
            "severity": self.severity.label,
            "node": self.node,
            "description": self.description,
            "fix": self.fix,
            "check_id": self.check_id,
            "agent_class": self.agent_class,
        }
        if include_evidence:
            d["evidence"] = self.evidence
        return d


@dataclass
class Route:
    target_agent: str
    priority: int = 0
    conditions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_keys(self) -> list[str]:
        return [c["context_key"] for c in self.conditions if "context_key" in c]


NON_DETERMINISTIC = {"LlmAgent"}
CONTROL_FLOW = {
    "resumable_orchestrator",
    "SequentialAgent",
    "decision_router_agent",
    "agent_gate",
}


@dataclass
class Node:
    name: str
    agent_class: str
    source_path: str = ""
    yaml_hash: str = ""
    description: str = ""
    instruction: str = ""
    output_key: str | None = None
    input_keys: list[str] = field(default_factory=list)
    strict: bool = True
    raw: dict = field(default_factory=dict)
    sub_agent_names: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    path: tuple[str, ...] = field(default_factory=tuple)
    routes: list[Route] = field(default_factory=list)
    query: str | None = None
    db_type: str | None = None
    db_url: str | None = None
    transform: dict | None = None
    model: str | None = None

    @property
    def kind(self) -> str:
        if self.agent_class in NON_DETERMINISTIC:
            return "non_deterministic"
        if self.agent_class in CONTROL_FLOW:
            return "control_flow"
        return "deterministic"

    @property
    def is_llm(self) -> bool:
        return self.agent_class in NON_DETERMINISTIC

    @property
    def qualified(self) -> str:
        return ".".join(self.path)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class PipelineGraph:
    name: str
    version: str
    root: Node
    nodes: dict[str, Node] = field(default_factory=dict)
    producers: dict[str, str] = field(default_factory=dict)
    consumers: dict[str, list[str]] = field(default_factory=dict)
    pipeline_hash: str = ""

    def by_class(self, agent_class: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.agent_class == agent_class]

    def deterministic(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "deterministic"]

    def non_deterministic(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "non_deterministic"]

    def downstream_of(self, node_name: str) -> list[Node]:
        node = self.nodes.get(node_name)
        if node is None:
            return []
        result: list[Node] = []
        for child in node.children:
            result.append(child)
            result.extend(self.downstream_of(child.name))
        return result


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    start_ns: int
    end_ns: int
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    children: list["Span"] = field(default_factory=list)

    def attr(self, key, default=None):
        return self.attributes.get(key, default)

    def jattr(self, key, default=None):
        val = self.attributes.get(key)
        if val is None:
            return default
        if isinstance(val, (dict, list)):
            return val
        try:
            return json.loads(val)
        except (TypeError, ValueError):
            return default

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)

    @property
    def agent_name(self) -> str:
        return self.attr("o2a.agent.name") or self.name

    @property
    def agent_class(self) -> str:
        return self.attr("o2a.agent.class", "")

    @property
    def output_key(self) -> str:
        return self.attr("o2a.agent.output_key", "") or ""

    @property
    def input_keys(self) -> list[str]:
        return self.jattr("o2a.agent.input_keys", []) or []

    @property
    def keys_before(self) -> list[str]:
        return self.jattr("o2a.session.keys_before", []) or []

    @property
    def keys_after(self) -> list[str]:
        return self.jattr("o2a.session.keys_after", []) or []

    @property
    def keys_written(self) -> list[str]:
        return self.jattr("o2a.session.keys_written", []) or []

    @property
    def keys_read(self) -> list[str]:
        return self.jattr("o2a.session.keys_read", []) or []

    @property
    def input_value(self) -> Any:
        return self.attr("input.value")

    @property
    def output_value(self) -> Any:
        return self.attr("output.value")

    @property
    def rendered_prompt(self) -> str:
        return self.attr("o2a.llm.rendered_prompt", "") or ""

    @property
    def session_snapshot(self) -> dict:
        return self.jattr("o2a.llm.session_snapshot", {}) or {}

    @property
    def template_references(self) -> list[str]:
        return self.jattr("o2a.llm.template_references", []) or []

    @property
    def evaluated_routes(self) -> list[dict]:
        return self.jattr("o2a.router.evaluated_routes", []) or []

    @property
    def chosen_target(self) -> str | None:
        return self.attr("o2a.router.chosen_target")

    @property
    def query_hash(self) -> str | None:
        return self.attr("o2a.db.query_hash")

    @property
    def row_count(self) -> int | None:
        return self.attr("o2a.db.row_count")

    @property
    def errored(self) -> bool:
        return (self.status or "").upper() == "ERROR"


@dataclass
class Run:
    trace_id: str
    pipeline_name: str
    pipeline_hash: str
    spans: list[Span]
    root: Span | None = None
    by_id: dict[str, Span] = field(default_factory=dict)
    by_agent: dict[str, list[Span]] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_ns for s in self.spans)
        end = max(s.end_ns for s in self.spans)
        return max(0.0, (end - start) / 1e9)

    def spans_of_class(self, agent_class: str) -> list[Span]:
        return [s for s in self.spans if s.agent_class == agent_class]

    def llm_spans(self) -> list[Span]:
        return self.spans_of_class("LlmAgent")

    def ordered(self) -> list[Span]:
        return sorted(self.spans, key=lambda s: s.start_ns)

    def spans_for(self, agent_name: str) -> list[Span]:
        return self.by_agent.get(agent_name, [])


@dataclass
class Scorecard:
    pipeline: str
    trace_id: str
    pipeline_hash: str
    duration_s: float
    axes: dict[str, float]
    overall: float
    flaws: list[Flaw]
    judge_calls_used: int = 0
    judge_calls_budget: int = 0
    span_count: int = 0

    @property
    def status(self) -> str:
        worst = self.max_severity
        if worst >= Severity.SEV_1:
            return "FAIL"
        if worst >= Severity.SEV_2:
            return "FLAGGED"
        return "PASS"

    @property
    def max_severity(self) -> Severity:
        if not self.flaws:
            return Severity.INFO
        return max(f.severity for f in self.flaws)
