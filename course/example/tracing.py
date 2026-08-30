"""Tiny in-memory tracer.

Same shape as production span capture: one span per agent step, recording
which session keys existed before and after it ran. That's the minimum an
evaluator (see eval.py) needs to diff declared intent against actual
behavior. Module 7 of the course covers the full production version of this
idea (OpenTelemetry spans, PII scrubbing, disk export).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Span:
    agent_name: str
    agent_class: str
    keys_before: list[str]
    start: float
    end: float | None = None
    keys_after: list[str] | None = None
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return (self.end if self.end is not None else self.start) - self.start


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start(self, node, keys_before: list[str]) -> Span:
        span = Span(
            agent_name=node.name,
            agent_class=node.agent_class,
            keys_before=keys_before,
            start=time.monotonic(),
        )
        self.spans.append(span)
        return span

    def end(self, span: Span, keys_after: list[str]) -> None:
        span.end = time.monotonic()
        span.keys_after = keys_after

    def error(self, span: Span, exc: Exception) -> None:
        span.error = f"{type(exc).__name__}: {exc}"

    def executed_agents(self) -> list[str]:
        return [s.agent_name for s in self.spans]
