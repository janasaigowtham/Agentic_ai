"""Stage 03.5 -- Fact-check.

Exists specifically to catch the Aggregator's own blind spots (spec section
5.4), which requires a genuinely different model lineage than the Aggregator
-- not just a different prompt on the same model. trh.config.STAGE_LINEAGE
enforces that split in live mode (Gemini runs the Aggregator, Claude runs
Fact-check); trh.judge.client.build_judge_clients refuses to build a
half-live setup that would put both on the same lineage.

Reads the original goal, the trajectory's outcome, and the Aggregator's own
recommendations. Writes a confidence-checked report, or flags routed_back so
the caller can send the trajectory back through the Aggregator once.
"""
from __future__ import annotations

from dataclasses import dataclass

from trh.core.harness_models import CaseSummary, Recommendation, Trajectory
from trh.judge.client import JudgeClient
from trh.judge.markers import stage_marker

SYSTEM_PROMPT = (
    "You are Fact-check in the Trajectory Review Harness. You review the "
    "Aggregator's recommendations against the original goal, the trajectory's "
    "outcome, and the recommendations' own stated root-cause chains -- looking "
    "specifically for claims the evidence doesn't actually support. You do not "
    "share the Aggregator's blind spots by construction; use that. Respond "
    'with strict JSON: {"confidence": "high"|"medium"|"low", "notes": str, '
    '"routed_back": bool}. Set routed_back true only if a recommendation makes '
    "a claim the trajectory doesn't support and needs to be reworked."
)


@dataclass
class FactCheckResult:
    confidence: str
    notes: str
    routed_back: bool

    def to_dict(self) -> dict:
        return {"confidence": self.confidence, "notes": self.notes, "routed_back": self.routed_back}


def build_user_prompt(
    case_summary: CaseSummary, trajectory: Trajectory, recommendations: list[Recommendation]
) -> str:
    lines = [
        f"Original goal: {case_summary.goal}",
        f"Trajectory outcome: {trajectory.outcome!r}",
        "",
        "Aggregator recommendations:",
    ]
    lines.extend(f"- {r.to_dict()!r}" for r in recommendations)
    return "\n".join(lines)


def run_fact_check(
    trajectory: Trajectory,
    case_summary: CaseSummary,
    recommendations: list[Recommendation],
    judge: JudgeClient,
) -> FactCheckResult:
    marker = stage_marker("fact_check", trajectory.trajectory_id)
    answer = judge.complete(
        stage="fact_check",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(case_summary, trajectory, recommendations),
        marker=marker,
    )
    return FactCheckResult(
        confidence=answer.get("confidence", "low"),
        notes=answer.get("notes", ""),
        routed_back=bool(answer.get("routed_back", False)),
    )
