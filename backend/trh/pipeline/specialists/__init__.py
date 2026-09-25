"""Stage 02 fan-out registry: the 9-way agent_class -> specialist mapping,
verified against the real O2A/ADK repo's agent_class values
(TRAJECTORY_REVIEW_HARNESS_ARCHITECTURE.html), not the SPEC.md's older
5-specialist placeholder list.

`run_specialists` walks the trajectory once, in execution order, and for each
step invokes whichever specialist's agent_classes match it. That single pass
*is* the event-triggered fan-out: each specialist fires the instant its step is
reached, never on a shared schedule -- a specialist encountered later in the
walk is just a later-occurring event, not something waiting in a queue.

Three of the nine specialists structurally never fire against the bundled
demo fixture (rest_api, iteration, lam_domain) because none of its agent
classes appear in that fixture's pipeline -- that is the design working
correctly, not a gap. `coverage_report` distinguishes "fired, found nothing"
from "never applicable here" for the UI.
"""
from __future__ import annotations

from trh.core.harness_models import CaseSummary, Trajectory, Verdict
from trh.judge.client import JudgeClient
from trh.pipeline.specialists.base import Specialist
from trh.pipeline.specialists.database import DatabaseSpecialist
from trh.pipeline.specialists.decision_routing import DecisionRoutingSpecialist
from trh.pipeline.specialists.gate_escalation import GateEscalationSpecialist
from trh.pipeline.specialists.iteration import IterationSpecialist
from trh.pipeline.specialists.lam_domain import LamDomainSpecialist
from trh.pipeline.specialists.llm_reasoning import LlmReasoningSpecialist
from trh.pipeline.specialists.orchestration_plan_fidelity import (
    OrchestrationPlanFidelitySpecialist,
)
from trh.pipeline.specialists.rest_api import RestApiSpecialist
from trh.pipeline.specialists.transformation import TransformationSpecialist

ALL_SPECIALISTS: list[Specialist] = [
    LlmReasoningSpecialist(),
    DatabaseSpecialist(),
    RestApiSpecialist(),
    TransformationSpecialist(),
    DecisionRoutingSpecialist(),
    IterationSpecialist(),
    GateEscalationSpecialist(),
    OrchestrationPlanFidelitySpecialist(),
    LamDomainSpecialist(),
]


def specialists_for_step(step) -> list[Specialist]:
    return [s for s in ALL_SPECIALISTS if s.fires_for(step)]


def run_specialists(
    trajectory: Trajectory, case_summary: CaseSummary, judge: JudgeClient
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for step in trajectory.steps:
        for specialist in specialists_for_step(step):
            verdicts.extend(specialist.evaluate(step, trajectory, case_summary, judge))
    return verdicts


def coverage_report(trajectory: Trajectory) -> dict[str, bool]:
    """specialist name -> whether it fired (its agent_class occurred) in this trajectory."""
    present_classes = {step.agent_class for step in trajectory.steps}
    return {
        specialist.name: bool(specialist.agent_classes & present_classes)
        for specialist in ALL_SPECIALISTS
    }
