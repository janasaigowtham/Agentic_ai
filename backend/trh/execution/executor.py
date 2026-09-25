"""Executor: interprets a declared PipelineGraph against mocked source
systems and synthetic seed data, producing real execution spans in the same
shape fixtures/make_trace.py hand-authors -- so Capture -> Trajectory
Assembler -> Evidence Assembly -> judgment can review an actually-executed
trajectory instead of only a pre-recorded one.

This is a small, pipeline-scoped interpreter, not a general workflow
engine: it implements exactly the control-flow shapes the bundled agent
YAMLs declare (sequential composition, a router with eq/neq conditions on
a context key, a gate with no children, LlmAgent, database_agent,
transformation/slv_transformation_agent). A genuinely new shape would need
a new case added here, the same way a real target's own orchestrator would
need one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from trh.core.graph import extract_sql_params, extract_template_keys
from trh.core.models import Node, PipelineGraph
from trh.execution.llm_mock import synthesize_llm_call
from trh.execution.mock_db import MockDatabase
from trh.execution.mock_rest import MockRestClient
from trh.execution.transform import evaluate_transform

GATE_DEFAULT_DURATION_S = 45.0
DB_CALL_DURATION_S = 0.3
TRANSFORM_DURATION_S = 0.05
LLM_CALL_DURATION_S = 6.0
REST_CALL_DURATION_S = 0.2


@dataclass
class _SpanBuilder:
    trace_id: str
    pipeline_name: str
    pipeline_hash: str
    spans: list[dict] = field(default_factory=list)
    session: dict[str, Any] = field(default_factory=dict)
    _seq: int = 0
    _clock_s: float = 0.0

    def _next_id(self) -> str:
        self._seq += 1
        return f"x{self._seq:03d}"

    def span(
        self,
        node: Node,
        duration_s: float,
        parent_id: str | None,
        *,
        input_value: Any = None,
        write_value: Any = None,
        extra_attrs: dict | None = None,
    ) -> str:
        span_id = self._next_id()
        start_s = self._clock_s
        end_s = start_s + max(duration_s, 0.0)
        self._clock_s = end_s

        attrs: dict[str, Any] = {
            "o2a.agent.name": node.name,
            "o2a.agent.class": node.agent_class,
            "o2a.agent.output_key": node.output_key or "",
            "o2a.agent.input_keys": json.dumps(node.input_keys),
            "o2a.pipeline.name": self.pipeline_name,
            "o2a.pipeline.hash": self.pipeline_hash,
            "o2a.session.keys_before": json.dumps(sorted(self.session.keys())),
        }
        if input_value is not None:
            attrs["input.value"] = json.dumps(input_value, default=str)

        keys_written: list[str] = []
        if write_value is not None:
            # Always surface what this step produced -- even a node with no
            # declared output_key (e.g. pmi_ddn_investor_review) still
            # computes something a real capture tap would see. Only a
            # declared output_key propagates it into session state, since
            # that's the only thing downstream nodes can actually reference.
            attrs["output.value"] = json.dumps(write_value, default=str)
            if node.output_key:
                is_new = node.output_key not in self.session
                self.session[node.output_key] = write_value
                if is_new:
                    keys_written.append(node.output_key)

        attrs["o2a.session.keys_after"] = json.dumps(sorted(self.session.keys()))
        attrs["o2a.session.keys_written"] = json.dumps(keys_written)
        if extra_attrs:
            attrs.update(extra_attrs)

        self.spans.append(
            {
                "trace_id": self.trace_id,
                "span_id": span_id,
                "parent_id": parent_id,
                "name": node.name,
                "start_ns": int(start_s * 1e9),
                "end_ns": int(end_s * 1e9),
                "status": "OK",
                "attributes": attrs,
                "events": [],
            }
        )
        return span_id

    def extend_to_cover_children(self, span_id: str) -> None:
        span = next(s for s in self.spans if s["span_id"] == span_id)
        children_end = max(
            (s["end_ns"] for s in self.spans if s["parent_id"] == span_id), default=span["end_ns"]
        )
        span["end_ns"] = max(span["end_ns"], children_end)


class Executor:
    def __init__(
        self,
        mock_db: MockDatabase,
        mock_rest: MockRestClient | None = None,
        gate_duration_s: float = GATE_DEFAULT_DURATION_S,
    ) -> None:
        self._db = mock_db
        self._rest = mock_rest or MockRestClient()
        self._gate_duration_s = gate_duration_s

    def run(self, graph: PipelineGraph, initial_session: dict[str, Any], trace_id: str) -> list[dict]:
        builder = _SpanBuilder(trace_id=trace_id, pipeline_name=graph.name, pipeline_hash=graph.pipeline_hash)
        builder.session.update(initial_session)
        self._execute_node(graph.root, builder, parent_id=None)
        return builder.spans

    def _execute_node(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        handler = getattr(self, f"_exec_{node.agent_class}", self._exec_passthrough)
        return handler(node, builder, parent_id)

    # -- control flow ----------------------------------------------------

    def _exec_resumable_orchestrator(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_sequential(node, builder, parent_id)

    def _exec_SequentialAgent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_sequential(node, builder, parent_id)

    def _exec_FailFastLoopAgent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_sequential(node, builder, parent_id)

    def _exec_iteration_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_sequential(node, builder, parent_id)

    def _exec_sequential(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        span_id = builder.span(node, duration_s=0.0, parent_id=parent_id)
        for child in node.children:
            self._execute_node(child, builder, parent_id=span_id)
        builder.extend_to_cover_children(span_id)
        return span_id

    def _exec_decision_router_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        evaluated_routes = []
        chosen_route = None
        for route in sorted(node.routes, key=lambda r: r.priority):
            matched = all(self._route_condition_matches(c, builder.session) for c in route.conditions)
            evaluated_routes.append(
                {
                    "target": route.target_agent,
                    "priority": route.priority,
                    "conditions": route.conditions,
                    "matched": matched,
                }
            )
            if matched and chosen_route is None:
                chosen_route = route

        span_id = builder.span(
            node,
            duration_s=TRANSFORM_DURATION_S,
            parent_id=parent_id,
            write_value=chosen_route.target_agent if chosen_route else None,
            extra_attrs={
                "o2a.router.evaluated_routes": json.dumps(evaluated_routes),
                "o2a.router.chosen_target": chosen_route.target_agent if chosen_route else None,
            },
        )
        if chosen_route is not None:
            target = next((c for c in node.children if c.name == chosen_route.target_agent), None)
            if target is not None:
                self._execute_node(target, builder, parent_id=span_id)
                builder.extend_to_cover_children(span_id)
        return span_id

    @staticmethod
    def _route_condition_matches(condition: dict, session: dict[str, Any]) -> bool:
        actual = session.get(condition["context_key"])
        expected = condition["value"]
        op = condition.get("operator", "eq")
        if op == "eq":
            return actual == expected
        if op == "neq":
            return actual != expected
        raise ValueError(f"unsupported route operator: {op!r}")

    def _exec_agent_gate(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return builder.span(node, duration_s=self._gate_duration_s, parent_id=parent_id)

    # -- leaf work ---------------------------------------------------------

    def _exec_database_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        params = {p: builder.session.get(p) for p in extract_sql_params(node.query or "")}
        rows = self._db.execute(node.query, params) if node.query else []
        return builder.span(
            node,
            duration_s=DB_CALL_DURATION_S,
            parent_id=parent_id,
            input_value=params,
            write_value=rows,
            extra_attrs={
                "o2a.db.query_hash": hashlib.sha256((node.query or "").encode()).hexdigest()[:16],
                "o2a.db.row_count": len(rows),
            },
        )

    def _exec_rest_api_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        result = self._rest.call(node.name, node.input_keys, builder.session)
        return builder.span(
            node,
            duration_s=REST_CALL_DURATION_S,
            parent_id=parent_id,
            input_value={k: builder.session.get(k) for k in node.input_keys},
            write_value=result,
        )

    def _exec_transformation_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_transform(node, builder, parent_id)

    def _exec_slv_transformation_agent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_transform(node, builder, parent_id)

    def _exec_transform(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        if not node.transform:
            return builder.span(node, duration_s=TRANSFORM_DURATION_S, parent_id=parent_id)
        results = evaluate_transform(node.transform, builder.session)
        # every transform node in this fixture declares exactly one output.
        value = next(iter(results.values())) if results else None
        return builder.span(node, duration_s=TRANSFORM_DURATION_S, parent_id=parent_id, write_value=value)

    def _exec_LlmAgent(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_llm(node, builder, parent_id)

    def _exec_lam_verdict_synthesizer(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_llm(node, builder, parent_id)

    def _exec_llm(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        rendered_prompt, output = synthesize_llm_call(node.instruction or "", builder.session)
        return builder.span(
            node,
            duration_s=LLM_CALL_DURATION_S,
            parent_id=parent_id,
            input_value={k: builder.session.get(k) for k in node.input_keys},
            write_value=output,
            extra_attrs={
                "o2a.llm.rendered_prompt": rendered_prompt,
                "o2a.llm.session_snapshot": json.dumps(dict(builder.session), default=str),
                "o2a.llm.template_references": json.dumps(sorted(extract_template_keys(node.instruction))),
            },
        )

    def _exec_passthrough(self, node: Node, builder: _SpanBuilder, parent_id: str | None) -> str:
        return self._exec_sequential(node, builder, parent_id)
