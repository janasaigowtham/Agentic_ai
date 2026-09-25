"""The demo's fixed set of reviewable trajectory sources -- one bundled fixture.

A real deployment would point this at wherever completed target-system
trajectories land; this harness only ever consumes one that already exists
(spec section 2.1), it never produces one itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BACKEND_DIR / "fixtures"


@dataclass(frozen=True)
class TrajectorySource:
    trajectory_id: str
    pipeline_name: str
    agent_dir: str
    trace_path: str
    label: str


TRAJECTORY_SOURCES: dict[str, TrajectorySource] = {
    "trace-fixture-0000000000000001": TrajectorySource(
        trajectory_id="trace-fixture-0000000000000001",
        pipeline_name="pmi_ddn_pipeline",
        agent_dir=str(FIXTURES_DIR / "agents"),
        trace_path=str(FIXTURES_DIR / "trace.jsonl"),
        label="PMI DDN loan servicing decision (seeded fixture)",
    )
}
