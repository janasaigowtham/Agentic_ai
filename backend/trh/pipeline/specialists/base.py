"""Shared specialist base.

Stage 02's general contract (TRAJECTORY_REVIEW_HARNESS_SPEC.md section 5.2):
a specialist only fires for step types it applies to, fires once per matching
step, reads the Case Summary plus only that step (and its declared contract
and evidence) -- never the full raw trajectory -- and writes zero or more
Verdicts. "Zero or more" includes a clean result: severity "none" is a
first-class Verdict outcome (it is in the spec's own severity enum), because
recording "this step was examined, nothing found" is what makes exhaustive
analysis (principle 4) auditable -- a specialist that fired and found nothing
looks different from one that never fired at all.

This module is invoked from the fan-out loop in specialists/__init__.py, which
walks the trajectory once and calls each matching specialist "the instant" its
step is reached -- there is no shared schedule or batching (principle 3).
"""
from __future__ import annotations

import uuid
from abc import ABC

from trh.core.harness_models import CaseSummary, Step, Trajectory, Verdict, VerdictSeverity
from trh.judge.client import JudgeClient
from trh.judge.markers import specialist_marker

SYSTEM_PROMPT_TEMPLATE = (
    "You are the {focus_title} specialist in the Trajectory Review Harness. "
    "You judge exactly one step of an agent pipeline's execution against what "
    "it declared it would do. You never see the full trajectory, only the "
    "case summary, this step, and this step's evidence (plain-code facts "
    "computed before your review, carrying no opinion). Judgment focus: "
    "{focus}. Respond with strict JSON: "
    '{{"finding": "<one paragraph>", "severity": "none"|"minor"|"significant"|"critical"}}. '
    'Use severity "none" only when you genuinely find nothing worth flagging.'
)


class Specialist(ABC):
    name: str
    agent_classes: frozenset[str]
    focus: str = "necessity, correctness, and declared-vs-actual behavior"

    def fires_for(self, step: Step) -> bool:
        return step.agent_class in self.agent_classes

    def trigger_event(self, step: Step) -> str:
        return f"{step.agent_class} step {step.step_id} ({step.agent_name}) completed"

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            focus_title=self.name.replace("_", " "), focus=self.focus
        )

    def build_user_prompt(self, step: Step, trajectory: Trajectory, case_summary: CaseSummary) -> str:
        return (
            f"Case goal: {case_summary.goal}\n\n"
            f"Step: {step.agent_name} ({step.agent_class})\n"
            f"Declared contract: {step.declared_contract_ref or 'none'}\n"
            f"Input: {step.input!r}\n"
            f"Output: {step.output!r}\n"
            f"Evidence (plain-code facts, no opinion attached):\n{step.evidence!r}\n"
        )

    def evaluate(
        self,
        step: Step,
        trajectory: Trajectory,
        case_summary: CaseSummary,
        judge: JudgeClient,
    ) -> list[Verdict]:
        if not self.fires_for(step):
            return []
        marker = specialist_marker(self.name, [step.step_id])
        answer = judge.complete(
            stage="specialist",
            system_prompt=self.system_prompt(),
            user_prompt=self.build_user_prompt(step, trajectory, case_summary),
            marker=marker,
        )
        severity = VerdictSeverity(answer.get("severity", "none"))
        return [
            Verdict(
                verdict_id=f"v-{uuid.uuid4().hex[:12]}",
                specialist=self.name,
                step_ids=[step.step_id],
                trigger_event=self.trigger_event(step),
                finding=answer.get("finding", ""),
                severity=severity,
                evidence=step.evidence,
            )
        ]
