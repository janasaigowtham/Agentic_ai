"""Stage 03 -- Aggregator: the only stage doing cross-step reasoning.

Reads the Case Summary and every Verdict Stage 02 produced -- all of them,
full detail, nothing capped or dropped here (spec section 7). Plain code only
surfaces a cross-step *fact*: which llm_reasoning verdicts share a long prompt
sentence, via prompt_sentence_fingerprints (computed as plain plumbing inside
the specialist, not judgment -- see pipeline/text_fingerprint.py). Whether
that overlap is actually a problem, and what to do about it, is the
Aggregator's own judge call to make; a recommendation must reference the
specific step(s) involved, never a generic fix.
"""
from __future__ import annotations

import uuid

from trh.core.harness_models import (
    CaseSummary,
    Recommendation,
    RecommendationStatus,
    Trajectory,
    Verdict,
)
from trh.judge.client import JudgeClient
from trh.judge.markers import stage_marker

SYSTEM_PROMPT = (
    "You are the Aggregator in the Trajectory Review Harness -- the only "
    "stage that reasons across steps. You receive the case summary and every "
    "specialist verdict for this trajectory (all of them, none dropped). "
    "Produce recommendations tied to a specific root cause: reference the "
    "exact step_id(s) involved, never a generic fix. Respond with strict "
    'JSON: {"recommendations": [{"tied_to_root_cause": bool, '
    '"root_cause_chain": [step_id, ...], "description": str, '
    '"proposed_fix": str}, ...]}. Return an empty list if nothing in the '
    "verdicts warrants a recommendation."
)


def _shared_fingerprint_groups(verdicts: list[Verdict]) -> list[dict]:
    by_fingerprint: dict[str, list[str]] = {}
    for v in verdicts:
        if v.specialist != "llm_reasoning":
            continue
        for fp in v.evidence.get("prompt_sentence_fingerprints", []):
            by_fingerprint.setdefault(fp, []).extend(v.step_ids)

    groups: list[dict] = []
    for fp, step_ids in by_fingerprint.items():
        unique_steps = sorted(set(step_ids))
        if len(unique_steps) > 1:
            groups.append({"fingerprint": fp, "step_ids": unique_steps})
    return groups


def build_user_prompt(case_summary: CaseSummary, verdicts: list[Verdict]) -> str:
    lines = [f"Case summary: {case_summary.to_dict()!r}", "", "All verdicts (none dropped):"]
    lines.extend(f"- {v.to_dict()!r}" for v in verdicts)
    lines.append("")
    lines.append(
        "Cross-step fact (plain code, no opinion attached) -- steps whose "
        f"prompts share an identical long sentence: {_shared_fingerprint_groups(verdicts)!r}"
    )
    return "\n".join(lines)


def run_aggregator(
    trajectory: Trajectory,
    case_summary: CaseSummary,
    verdicts: list[Verdict],
    judge: JudgeClient,
) -> list[Recommendation]:
    marker = stage_marker("aggregator", trajectory.trajectory_id)
    answer = judge.complete(
        stage="aggregator",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(case_summary, verdicts),
        marker=marker,
    )
    recommendations: list[Recommendation] = []
    for item in answer.get("recommendations", []):
        recommendations.append(
            Recommendation(
                recommendation_id=f"r-{uuid.uuid4().hex[:12]}",
                trajectory_id=trajectory.trajectory_id,
                tied_to_root_cause=bool(item.get("tied_to_root_cause", False)),
                root_cause_chain=list(item.get("root_cause_chain", [])),
                description=item.get("description", ""),
                proposed_fix=item.get("proposed_fix", ""),
                status=RecommendationStatus.PROPOSED,
            )
        )
    return recommendations
