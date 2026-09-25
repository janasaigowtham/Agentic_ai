"""Orchestrates the full pipeline: Capture -> Trajectory Assembler ->
Evidence Assembly -> Orient -> Specialists -> Aggregator -> Fact-check.

The Gate (human approval) is deliberately not run from here -- it's a
checkpoint the API/UI drives once a review's recommendations exist, never
something this function applies on its own (spec section 5.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from trh.config import HarnessConfig
from trh.core.graph import load_pipeline
from trh.core.harness_models import CaseSummary, Recommendation, Trajectory, Verdict
from trh.fixtures_registry import TrajectorySource, load_trajectory_from_source
from trh.judge.client import build_judge_clients, client_for_stage
from trh.pipeline.aggregator import run_aggregator
from trh.pipeline.evidence_assembly import assemble_evidence
from trh.pipeline.fact_check import FactCheckResult, run_fact_check
from trh.pipeline.orient import run_orient
from trh.pipeline.specialists import run_specialists
from trh.pipeline.trajectory_assembler import assemble_trajectory


@dataclass
class ReviewResult:
    trajectory: Trajectory
    case_summary: CaseSummary
    verdicts: list[Verdict]
    recommendations: list[Recommendation]
    fact_check: FactCheckResult
    stage_lineage_used: dict[str, str]


def run_review(source: TrajectorySource, config: HarnessConfig) -> ReviewResult:
    graph = load_pipeline(source.agent_dir, source.pipeline_name)
    _, captured = load_trajectory_from_source(source, graph)
    trajectory = assemble_trajectory(source.trajectory_id, source.pipeline_name, captured, graph)
    assemble_evidence(trajectory, graph)

    clients = build_judge_clients(config)

    orient_client = client_for_stage(clients, config, "orient")
    case_summary = run_orient(trajectory, orient_client, graph)

    specialist_client = client_for_stage(clients, config, "specialist")
    verdicts = run_specialists(trajectory, case_summary, specialist_client)

    aggregator_client = client_for_stage(clients, config, "aggregator")
    recommendations = run_aggregator(trajectory, case_summary, verdicts, aggregator_client)

    fact_check_client = client_for_stage(clients, config, "fact_check")
    fact_check_result = run_fact_check(trajectory, case_summary, recommendations, fact_check_client)

    stage_lineage_used = {
        stage: config.lineage_for(stage).value
        for stage in ("orient", "specialist", "aggregator", "fact_check")
    }

    return ReviewResult(
        trajectory=trajectory,
        case_summary=case_summary,
        verdicts=verdicts,
        recommendations=recommendations,
        fact_check=fact_check_result,
        stage_lineage_used=stage_lineage_used,
    )
