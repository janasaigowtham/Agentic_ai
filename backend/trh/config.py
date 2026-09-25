"""Judgment-engine configuration: mock-vs-live mode resolution and the
stage -> model-lineage mapping.

Mock mode is the default (zero API keys, deterministic canned responses).
Live mode activates only when both GEMINI_API_KEY and ANTHROPIC_API_KEY are
present -- partial credentials stay in mock rather than half-violate the
lineage requirement in TRAJECTORY_REVIEW_HARNESS_SPEC.md section 5.4 (Fact-check
on the same lineage as the Aggregator, or an unauthenticated Gemini stage).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class JudgeMode(str, Enum):
    MOCK = "mock"
    LIVE = "live"


class ModelLineage(str, Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    MOCK = "mock"


# The real target runs Google ADK, so Gemini runs every stage except
# Fact-check, which runs on Claude specifically -- a different lineage than
# the Aggregator it double-checks, per spec section 5.4.
STAGE_LINEAGE: dict[str, ModelLineage] = {
    "orient": ModelLineage.GEMINI,
    "specialist": ModelLineage.GEMINI,
    "aggregator": ModelLineage.GEMINI,
    "fact_check": ModelLineage.CLAUDE,
}


@dataclass(frozen=True)
class HarnessConfig:
    mode: JudgeMode
    gemini_api_key: str | None
    anthropic_api_key: str | None
    gemini_model: str = "gemini-2.5-pro"
    claude_model: str = "claude-sonnet-5"

    def lineage_for(self, stage: str) -> ModelLineage:
        if self.mode is JudgeMode.MOCK:
            return ModelLineage.MOCK
        return STAGE_LINEAGE[stage]


def load_config() -> HarnessConfig:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    mode = JudgeMode.LIVE if (gemini_key and anthropic_key) else JudgeMode.MOCK
    return HarnessConfig(
        mode=mode,
        gemini_api_key=gemini_key,
        anthropic_api_key=anthropic_key,
        gemini_model=os.environ.get("TRH_GEMINI_MODEL", "gemini-2.5-pro"),
        claude_model=os.environ.get("TRH_CLAUDE_MODEL", "claude-sonnet-5"),
    )
