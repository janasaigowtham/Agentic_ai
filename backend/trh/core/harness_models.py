"""Harness-native dataclasses: Trajectory / Step / CaseSummary / Verdict /
Recommendation / GateDecision / CalibrationRecord.

Field lists follow TRAJECTORY_REVIEW_HARNESS_SPEC.md section 4. These are
distinct from trh.core.models (vendored from o2a_eval): that module models a
captured OTel-style trace (Span/Run); this module models the harness's own
judgment-pipeline objects, built on top of a trace via the capture and
trajectory-assembler stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import Span

# Open-ended per spec section 4.1; this is the initial set plus the two
# extra shapes this taxonomy's agent classes need (llm_reasoning, plan_node
# covers routing/iteration/orchestration).
AGENT_CLASS_STEP_TYPE: dict[str, str] = {
    "resumable_orchestrator": "plan_node",
    "SequentialAgent": "plan_node",
    "FailFastLoopAgent": "plan_node",
    "decision_router_agent": "plan_node",
    "iteration_agent": "plan_node",
    "agent_gate": "gate_permission",
    "database_agent": "tool_call",
    "rest_api_agent": "tool_call",
    "transformation_agent": "tool_call",
    "slv_transformation_agent": "tool_call",
    "LlmAgent": "llm_reasoning",
    "lam_verdict_synthesizer": "llm_reasoning",
}


def step_type_for(agent_class: str) -> str:
    return AGENT_CLASS_STEP_TYPE.get(agent_class, "tool_call")


class VerdictSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class GateDecisionValue(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"


class RecommendationStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DECLINED = "declined"
    HELD = "held"


@dataclass
class Step:
    step_id: str
    step_type: str
    agent_name: str
    agent_class: str
    input: Any
    output: Any
    started_at: float
    completed_at: float
    parent_step_id: str | None = None
    declared_contract_ref: str | None = None
    # Harness-native extension (not in spec section 4.1): which capture tap
    # this step arrived through. Not persisted as spec-contract data, but
    # needed downstream to reason about capture coverage gaps.
    tap: str = "a"
    evidence: dict[str, Any] = field(default_factory=dict)
    raw_span: Span | None = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "agent_name": self.agent_name,
            "agent_class": self.agent_class,
            "input": self.input,
            "output": self.output,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "parent_step_id": self.parent_step_id,
            "declared_contract_ref": self.declared_contract_ref,
            "tap": self.tap,
            "evidence": self.evidence,
        }


@dataclass
class Trajectory:
    trajectory_id: str
    source_pipeline: str
    source_spec_ref: str | None
    steps: list[Step]
    started_at: float
    completed_at: float | None
    outcome: Any

    def step(self, step_id: str) -> Step | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "source_pipeline": self.source_pipeline,
            "source_spec_ref": self.source_spec_ref,
            "steps": [s.to_dict() for s in self.steps],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
        }


@dataclass
class CaseSummary:
    trajectory_id: str
    goal: str
    plan_declared: Any | None
    plan_actual: list[str]
    outcome: Any
    step_type_map: dict[str, str]
    orient_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "plan_declared": self.plan_declared,
            "plan_actual": self.plan_actual,
            "outcome": self.outcome,
            "step_type_map": self.step_type_map,
            "orient_notes": self.orient_notes,
        }


@dataclass
class Verdict:
    verdict_id: str
    specialist: str
    step_ids: list[str]
    trigger_event: str
    finding: str
    severity: VerdictSeverity
    evidence: Any = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict_id": self.verdict_id,
            "specialist": self.specialist,
            "step_ids": self.step_ids,
            "trigger_event": self.trigger_event,
            "finding": self.finding,
            "severity": self.severity.value,
            "evidence": self.evidence,
        }


@dataclass
class Recommendation:
    recommendation_id: str
    trajectory_id: str
    tied_to_root_cause: bool
    root_cause_chain: list[str]
    description: str
    proposed_fix: str
    status: RecommendationStatus = RecommendationStatus.PROPOSED

    def to_dict(self) -> dict:
        return {
            "recommendation_id": self.recommendation_id,
            "trajectory_id": self.trajectory_id,
            "tied_to_root_cause": self.tied_to_root_cause,
            "root_cause_chain": self.root_cause_chain,
            "description": self.description,
            "proposed_fix": self.proposed_fix,
            "status": self.status.value,
        }


@dataclass
class GateDecision:
    recommendation_id: str
    decision: GateDecisionValue
    reviewer: str
    note: str | None
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "recommendation_id": self.recommendation_id,
            "decision": self.decision.value,
            "reviewer": self.reviewer,
            "note": self.note,
            "timestamp": self.timestamp,
        }


@dataclass
class CalibrationRecord:
    case_id: str
    trajectory_id: str
    harness_verdict: Any
    human_label: Any
    agreement: bool
    reviewed_by: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "trajectory_id": self.trajectory_id,
            "harness_verdict": self.harness_verdict,
            "human_label": self.human_label,
            "agreement": self.agreement,
            "reviewed_by": self.reviewed_by,
            "timestamp": self.timestamp,
        }
