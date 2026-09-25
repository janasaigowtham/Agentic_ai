"""The demo's set of reviewable trajectory sources.

A real deployment would point this at wherever completed target-system
trajectories land; the harness itself only ever consumes a trajectory that
already exists (spec section 2.1), it never produces one.

There is no live ADK app in this environment, so two kinds of source stand
in for one, and neither is the harness doing the producing:
  - a hand-authored trace.jsonl (fixtures/make_trace.py), replayed as-is.
  - an executor (trh.execution.*) that actually interprets the same agent
    YAMLs against mocked source systems and synthetic seed data, playing
    the role the absent ADK app would play. It is wired up here, entirely
    separate from the harness's judgment pipeline -- Capture still only
    *observes* whatever it produces, exactly as it would observe a live
    target. The harness itself still never drives anything (spec section 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trh.core.models import PipelineGraph
from trh.execution.synthetic_data import LOAN_SCENARIOS

BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BACKEND_DIR / "fixtures"
AGENT_DIR = str(FIXTURES_DIR / "agents")
PIPELINE_NAME = "pmi_ddn_pipeline"


@dataclass(frozen=True)
class TrajectorySource:
    trajectory_id: str
    pipeline_name: str
    agent_dir: str
    label: str
    trace_path: str | None = None
    loan_number: str | None = None


TRAJECTORY_SOURCES: dict[str, TrajectorySource] = {
    "trace-fixture-0000000000000001": TrajectorySource(
        trajectory_id="trace-fixture-0000000000000001",
        pipeline_name=PIPELINE_NAME,
        agent_dir=AGENT_DIR,
        trace_path=str(FIXTURES_DIR / "trace.jsonl"),
        label="PMI DDN loan servicing decision (seeded fixture, hand-recorded)",
    ),
}

for _scenario in LOAN_SCENARIOS.values():
    _trajectory_id = f"synthetic-{_scenario.loan_number}"
    TRAJECTORY_SOURCES[_trajectory_id] = TrajectorySource(
        trajectory_id=_trajectory_id,
        pipeline_name=PIPELINE_NAME,
        agent_dir=AGENT_DIR,
        loan_number=_scenario.loan_number,
        label=f"Synthetic loan {_scenario.loan_number} -- {_scenario.label}",
    )


def load_trajectory_from_source(source: TrajectorySource, graph: PipelineGraph):
    """Returns (Run, list[CapturedSpan]) for a source, whichever kind it is."""
    from trh.capture.simulate import generate_and_capture, load_and_capture

    if source.loan_number is not None:
        return generate_and_capture(source.trajectory_id, source.loan_number, graph)
    return load_and_capture(source.trace_path)
