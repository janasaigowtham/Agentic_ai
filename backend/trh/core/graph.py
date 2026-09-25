"""YAML pipeline loader: builds a PipelineGraph from a directory of agent YAMLs."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .models import Node, PipelineGraph, Route

_DOUBLE_BRACE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)")
_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_SQL_PARAM_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


def extract_template_keys(blob: Any) -> set[str]:
    """Walk any nested structure and collect {{ key }} and {key} template refs."""
    keys: set[str] = set()

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            keys.update(_DOUBLE_BRACE_RE.findall(node))
            keys.update(_SINGLE_BRACE_RE.findall(node))
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)

    _walk(blob)
    return keys


def extract_sql_params(query: str) -> set[str]:
    if not query:
        return set()
    return set(_SQL_PARAM_RE.findall(query))


def _merge_input_keys(spec: dict) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()

    def _add(k: str) -> None:
        if k and k not in seen:
            seen.add(k)
            keys.append(k)

    for k in spec.get("input_keys") or []:
        _add(k)
    single = spec.get("input_key")
    if isinstance(single, str):
        _add(single)
    return keys


def _parse_db_yaml(spec: dict) -> tuple[str | None, str | None, str | None]:
    """Returns (query, db_type, db_url) from default_db_yaml (dict or embedded string)."""
    raw = spec.get("default_db_yaml")
    if raw is None:
        return None, None, None
    if isinstance(raw, str):
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError:
            parsed = None
        if not isinstance(parsed, dict):
            return None, None, None
    elif isinstance(raw, dict):
        parsed = raw
    else:
        return None, None, None

    query = parsed.get("query")
    db_type = parsed.get("type")
    connection = parsed.get("connection") or {}
    db_url = connection.get("url") if isinstance(connection, dict) else None
    return query, db_type, db_url


def _parse_routes(spec: dict) -> list[Route]:
    routes: list[Route] = []
    for entry in spec.get("routes") or []:
        if not isinstance(entry, dict):
            continue
        routes.append(
            Route(
                target_agent=entry.get("target_agent", ""),
                priority=entry.get("priority", 0) or 0,
                conditions=list(entry.get("conditions") or []),
                metadata=dict(entry.get("metadata") or {}),
            )
        )
    return routes


def _parse_sub_agent_names(spec: dict) -> list[str]:
    names: list[str] = []
    for entry in spec.get("sub_agents") or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if name:
                names.append(name)
        elif isinstance(entry, str):
            names.append(entry)
    return names


def _node_from_spec(spec: dict, path: Path) -> Node:
    query, db_type, db_url = _parse_db_yaml(spec)
    yaml_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path and path.exists() else ""
    return Node(
        name=spec.get("name", ""),
        agent_class=spec.get("agent_class", ""),
        source_path=str(path) if path else "",
        yaml_hash=yaml_hash,
        description=spec.get("description", "") or "",
        instruction=spec.get("instruction", "") or "",
        output_key=spec.get("output_key"),
        input_keys=_merge_input_keys(spec),
        strict=spec.get("strict", True),
        raw=spec,
        sub_agent_names=_parse_sub_agent_names(spec),
        children=[],
        path=(),
        routes=_parse_routes(spec),
        query=query,
        db_type=db_type,
        db_url=db_url,
        transform=spec.get("transform"),
        model=spec.get("model"),
    )


def index_directory(agent_dir: str | Path) -> dict[str, Node]:
    agent_dir = Path(agent_dir)
    index: dict[str, Node] = {}
    files = sorted(agent_dir.glob("*.yaml")) + sorted(agent_dir.glob("*.yml"))
    for path in files:
        try:
            spec = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(spec, dict) or "name" not in spec:
            continue
        node = _node_from_spec(spec, path)
        index[node.name] = node
    return index


def load_pipeline(agent_dir: str | Path, pipeline_name: str) -> PipelineGraph:
    index = index_directory(agent_dir)
    if pipeline_name not in index:
        raise ValueError(f"pipeline {pipeline_name!r} not found in {agent_dir}")

    reachable: dict[str, Node] = {}

    def _resolve(name: str, parent_path: tuple[str, ...], in_progress: set[str]) -> Node:
        node = index.get(name)
        if node is None:
            node = Node(name=name, agent_class="<missing>")
            index[name] = node

        if name in reachable:
            existing = reachable[name]
            return existing

        node.path = parent_path + (name,)
        reachable[name] = node

        if name in in_progress:
            return node
        in_progress = in_progress | {name}

        children: list[Node] = []
        for child_name in node.sub_agent_names:
            if child_name in in_progress:
                child = index.get(child_name)
                if child is None:
                    child = Node(name=child_name, agent_class="<missing>")
                    index[child_name] = child
                children.append(child)
                continue
            children.append(_resolve(child_name, node.path, in_progress))
        node.children = children
        return node

    root = _resolve(pipeline_name, (), set())

    producers: dict[str, str] = {}
    for node in reachable.values():
        if node.output_key:
            producers[node.output_key] = node.name

    consumers: dict[str, list[str]] = {}

    def _add_consumer(key: str, node_name: str) -> None:
        consumers.setdefault(key, [])
        if node_name not in consumers[key]:
            consumers[key].append(node_name)

    for node in reachable.values():
        refs: set[str] = set(node.input_keys)
        refs |= extract_template_keys(node.transform)
        refs |= extract_template_keys(node.instruction)
        for route in node.routes:
            refs |= set(route.context_keys)
        for key in refs:
            _add_consumer(key, node.name)

    hashes = sorted(n.yaml_hash for n in reachable.values() if n.yaml_hash)
    pipeline_hash = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]

    return PipelineGraph(
        name=root.name,
        version=root.raw.get("version", "") if root.raw else "",
        root=root,
        nodes=reachable,
        producers=producers,
        consumers=consumers,
        pipeline_hash=pipeline_hash,
    )
