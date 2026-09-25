"""Span dicts -> Run, tree reconstruction, jsonl loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Run, Span


def _span_from_dict(d: dict[str, Any]) -> Span:
    return Span(
        span_id=d.get("span_id", ""),
        trace_id=d.get("trace_id", ""),
        parent_id=d.get("parent_id"),
        name=d.get("name", ""),
        start_ns=d.get("start_ns", 0),
        end_ns=d.get("end_ns", 0),
        status=d.get("status", "OK"),
        attributes=dict(d.get("attributes") or {}),
        events=list(d.get("events") or []),
        children=[],
    )


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_run(span_dicts: list[dict]) -> Run:
    spans = [_span_from_dict(d) for d in span_dicts]
    by_id: dict[str, Span] = {s.span_id: s for s in spans}
    by_agent: dict[str, list[Span]] = {}

    root: Span | None = None
    for s in spans:
        by_agent.setdefault(s.agent_name, []).append(s)
        if s.parent_id and s.parent_id in by_id:
            by_id[s.parent_id].children.append(s)
        else:
            if root is None or s.start_ns < root.start_ns:
                root = s

    for children in by_agent.values():
        pass

    trace_id = spans[0].trace_id if spans else ""
    pipeline_name = root.agent_name if root else ""
    pipeline_hash = ""
    if root:
        pipeline_hash = root.attr("o2a.pipeline.hash", "") or ""

    return Run(
        trace_id=trace_id,
        pipeline_name=pipeline_name,
        pipeline_hash=pipeline_hash,
        spans=spans,
        root=root,
        by_id=by_id,
        by_agent=by_agent,
    )
