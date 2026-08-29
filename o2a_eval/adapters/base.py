"""FrameworkAdapter — the one place o2a-eval couples to a specific runner."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.graph import extract_template_keys

MAX_ATTR_BYTES = 65536


def _truncate(text: str, max_bytes: int = MAX_ATTR_BYTES) -> str:
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


@dataclass
class FrameworkAdapter:
    base_class_path: str
    run_method: str = "run"

    get_session: Callable = field(
        default=lambda args, kwargs: getattr(args[0], "session", {}) if args else {}
    )
    get_agent_name: Callable = field(
        default=lambda agent: getattr(agent, "name", type(agent).__name__)
    )
    get_agent_class: Callable = field(default=lambda agent: type(agent).__name__)
    get_output_key: Callable = field(default=lambda agent: getattr(agent, "output_key", None))
    get_input_keys: Callable = field(
        default=lambda agent: list(getattr(agent, "input_keys", []) or [])
    )
    render_prompt: Callable = field(default=lambda agent, session: None)
    get_routes: Callable = field(default=lambda agent, session: [])
    get_query_info: Callable = field(default=lambda agent, session, result: {})

    llm_class_names: tuple = ("LlmAgent",)
    router_class_names: tuple = ("decision_router_agent",)
    db_class_names: tuple = ("database_agent",)
    pipeline_name: str = ""
    pipeline_hash: str = ""
    capture_full_fidelity: bool = True


def set_pre_attributes(span, agent, session: dict, adapter: FrameworkAdapter) -> None:
    agent_class = adapter.get_agent_class(agent)
    span.set_attribute("o2a.agent.name", adapter.get_agent_name(agent))
    span.set_attribute("o2a.agent.class", agent_class)
    span.set_attribute("o2a.agent.output_key", adapter.get_output_key(agent) or "")
    span.set_attribute("o2a.agent.input_keys", json.dumps(adapter.get_input_keys(agent)))
    span.set_attribute("o2a.session.keys_before", json.dumps(sorted(session.keys())))
    span.set_attribute("o2a.pipeline.name", adapter.pipeline_name)
    span.set_attribute("o2a.pipeline.hash", adapter.pipeline_hash)

    if agent_class in adapter.llm_class_names and adapter.capture_full_fidelity:
        prompt = adapter.render_prompt(agent, session)
        if prompt:
            span.set_attribute("o2a.llm.rendered_prompt", _truncate(prompt))
            span.set_attribute(
                "o2a.llm.session_snapshot", _truncate(json.dumps(session, default=str))
            )
            refs = getattr(agent, "template_references", None)
            if refs is None:
                refs = sorted(extract_template_keys(prompt))
            span.set_attribute("o2a.llm.template_references", json.dumps(list(refs)))

    if agent_class in adapter.router_class_names:
        routes = adapter.get_routes(agent, session)
        if routes:
            span.set_attribute("o2a.router.evaluated_routes", json.dumps(routes, default=str))


def set_post_attributes(
    span, agent, session: dict, keys_before: set, result: Any, adapter: FrameworkAdapter
) -> None:
    agent_class = adapter.get_agent_class(agent)
    keys_after = set(session.keys())
    span.set_attribute("o2a.session.keys_after", json.dumps(sorted(keys_after)))
    span.set_attribute("o2a.session.keys_written", json.dumps(sorted(keys_after - keys_before)))

    output_key = adapter.get_output_key(agent)

    if adapter.capture_full_fidelity and output_key and output_key in session:
        span.set_attribute(
            "output.value", _truncate(json.dumps(session[output_key], default=str))
        )

    if agent_class in adapter.router_class_names:
        chosen = None
        if output_key and output_key in session and isinstance(session[output_key], str):
            chosen = session[output_key]
        if chosen:
            span.set_attribute("o2a.router.chosen_target", chosen)

    if agent_class in adapter.db_class_names:
        info = adapter.get_query_info(agent, session, result) or {}
        if "query_hash" in info:
            span.set_attribute("o2a.db.query_hash", info["query_hash"])
        if "row_count" in info:
            span.set_attribute("o2a.db.row_count", info["row_count"])
        if adapter.capture_full_fidelity and "query" in info:
            span.set_attribute("o2a.db.query", _truncate(info["query"]))
