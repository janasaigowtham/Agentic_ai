"""Capture simulation: the demo's stand-in for the harness's two live taps.

Tap A is ADK's plugin manager (run_before_agent_callback / run_after_agent_callback),
fired from base_agent.run_async() -- it sees a pre-execution snapshot. Tap B is the
top-level event stream: it always sees a span after the fact, even when Tap A didn't
fire. TRAJECTORY_REVIEW_HARNESS_ARCHITECTURE.html traced a real gap in the target
repo: decision_router_agent and iteration_agent invoke their own sub-agents via
sub_agent._run_async_impl(...) directly, bypassing run_async() and therefore the
plugin manager -- Tap A never fires for those sub-agents, only Tap B's after-the-fact
state_delta does.

There is no live ADK app in this environment, so this module replays an
already-recorded fixture trace instead of tapping a running pipeline. The replay
mechanism is real: any span whose agent_class is in `bypassed_classes` is tagged
"b_only" instead of "a", exactly as it would be against a live target with that
gap. The fixture pipeline's own agent classes don't include the two bypassed
classes, so the default list is empty -- wired for the real gap, not hardcoded
around it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trh.core.models import PipelineGraph, Run, Span
from trh.core.trace import build_run, load_jsonl

TAP_A = "a"
TAP_B_ONLY = "b_only"

# agent_class values that bypass the plugin manager in the real O2A/ADK target.
# Empty by default: this fixture pipeline doesn't route through either class.
DEFAULT_BYPASSED_CLASSES: frozenset[str] = frozenset()


@dataclass
class CapturedSpan:
    span: Span
    tap: str

    @property
    def has_pre_execution_snapshot(self) -> bool:
        return self.tap == TAP_A


def simulate_capture(
    run: Run,
    bypassed_classes: frozenset[str] = DEFAULT_BYPASSED_CLASSES,
) -> list[CapturedSpan]:
    """Tags every span in `run` with the tap that would have captured it."""
    captured: list[CapturedSpan] = []
    for span in run.spans:
        tap = TAP_B_ONLY if span.agent_class in bypassed_classes else TAP_A
        captured.append(CapturedSpan(span=span, tap=tap))
    return captured


def load_and_capture(
    trace_path: str,
    bypassed_classes: frozenset[str] = DEFAULT_BYPASSED_CLASSES,
) -> tuple[Run, list[CapturedSpan]]:
    """Loads a recorded trace.jsonl and replays it through simulated capture."""
    span_dicts = load_jsonl(trace_path)
    run = build_run(span_dicts)
    return run, simulate_capture(run, bypassed_classes)


def generate_and_capture(
    trajectory_id: str,
    loan_number: str,
    graph: PipelineGraph,
    bypassed_classes: frozenset[str] = DEFAULT_BYPASSED_CLASSES,
    extra_session: dict[str, Any] | None = None,
) -> tuple[Run, list[CapturedSpan]]:
    """Actually executes the declared pipeline against mocked source systems
    and synthetic seed data (trh.execution.*), instead of replaying a
    pre-recorded trace. Produces a real Run the same way load_and_capture
    does, so both trajectory sources feed the rest of the harness identically.
    """
    from trh.execution.executor import Executor
    from trh.execution.synthetic_data import LOAN_SCENARIOS, build_mock_database

    scenario = LOAN_SCENARIOS[loan_number]
    db = build_mock_database()
    executor = Executor(mock_db=db, gate_duration_s=scenario.gate_duration_s)
    initial_session: dict[str, Any] = {"loan_number": loan_number}
    if extra_session:
        initial_session.update(extra_session)
    span_dicts = executor.run(graph, initial_session=initial_session, trace_id=trajectory_id)
    run = build_run(span_dicts)
    return run, simulate_capture(run, bypassed_classes)
