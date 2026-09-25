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

from trh.core.models import Run, Span
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
