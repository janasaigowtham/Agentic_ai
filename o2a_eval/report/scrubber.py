"""PII-safe scrubbing and disk writers.

Design principle: allow-list, not deny-list. A new evidence key a check author
adds is silently dropped until it appears in SAFE_EVIDENCE_KEYS. This fails
closed (a new key is blocked until explicitly declared safe) rather than
failing open (a deny-list lets a new key through until noticed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.models import Run, Scorecard

SAFE_EVIDENCE_KEYS = {
    # Counts, scores, durations, thresholds — never values
    "call_count", "wasted_seconds", "duration_s", "budget_s", "row_count",
    "threshold", "approx_tokens", "char_count", "section_hits", "overlap_ratio",
    "shared_ngrams", "score", "ctes", "joins", "windows", "subqueries",
    "final_key_count", "downstream_checked", "declared_count", "wait_s",
    "ceiling_s", "threshold_s", "consumer_position", "producer_position",
    "pct_of_total", "mean_score", "db_row_count_threshold", "sample_size",
    "fire_rate", "spans_purged", "temp_shredded", "already_clean",
    # Names — agent names, session KEY names (never key values)
    "missing", "unexpected", "declared", "actual", "declared_order",
    "actual_order", "unresolved", "available", "session_key", "produced_by",
    "producer_class", "declared_consumers", "actual_llm_readers", "undeclared",
    "output_key", "keys", "missing_keys", "known_producers", "target",
    "sub_agents", "orphans", "routed_targets", "context_keys", "covered_values",
    "duplicate_priorities", "routes", "matched", "chosen", "expected",
    "matched_routes", "session_keys", "node_a", "node_b", "missing_refs",
    "parsing_consumers", "referenced_by", "duplicates", "bound_params",
    "declared_input_keys", "unused", "tables", "interpolated_refs",
    "downstream_indexers", "re_derived_fields", "available_field_names",
    "failure_mode", "top_contributors", "agent", "class",
    "declared_output_key", "available_keys", "input_keys", "strict",
    "dialect", "statement", "has_env_ref", "mode", "priority", "error",
}

BLOCKED_EVIDENCE_KEYS = {
    "input_value", "output_value", "rendered_prompt", "session_snapshot",
    "query", "sample_shared_phrase", "judge_reason", "reason", "axes",
    "snippet", "prompt", "value", "values", "row", "rows", "data",
}


def _scrub_value(v: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "<nested>"
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        return v[:200]
    if isinstance(v, (list, tuple)):
        return [_scrub_value(x, depth + 1) for x in v[:50]]
    if isinstance(v, dict):
        return {
            str(k)[:80]: _scrub_value(val, depth + 1)
            for k, val in list(v.items())[:50]
            if k not in BLOCKED_EVIDENCE_KEYS
        }
    return str(v)[:200]


def scrub_evidence(evidence: dict) -> dict:
    out: dict = {}
    dropped: list[str] = []
    for k, v in (evidence or {}).items():
        if k in BLOCKED_EVIDENCE_KEYS or k not in SAFE_EVIDENCE_KEYS:
            dropped.append(k)
            continue
        out[k] = _scrub_value(v)
    if dropped:
        out["_dropped_fields"] = sorted(dropped)
    return out


def scorecard_to_safe_dict(card: Scorecard) -> dict:
    return {
        "pipeline": card.pipeline,
        "trace_id": card.trace_id,
        "pipeline_hash": card.pipeline_hash,
        "duration_s": card.duration_s,
        "span_count": card.span_count,
        "status": card.status,
        "overall": card.overall,
        "axes": card.axes,
        "judge_calls_used": card.judge_calls_used,
        "judge_calls_budget": card.judge_calls_budget,
        "flaws": [
            {
                "check_id": f.check_id,
                "type": f.type,
                "severity": f.severity.label,
                "node": f.node,
                "agent_class": f.agent_class,
                "description": (f.description or "")[:500],
                "fix": (f.fix or "")[:500] if f.fix else None,
                "evidence": scrub_evidence(f.evidence),
            }
            for f in card.flaws
        ],
    }


def trace_summary(run: Run) -> list[dict]:
    summary = []
    for span in run.ordered():
        summary.append(
            {
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "agent": span.agent_name,
                "agent_class": span.agent_class,
                "duration_s": span.duration_s,
                "status": span.status,
                "output_key": span.output_key,
                "session_keys_before": sorted(span.keys_before),
                "session_keys_written": sorted(span.keys_written),
                "template_references": sorted(span.template_references),
                "router_chosen_target": span.chosen_target,
                "row_count": span.row_count,
            }
        )
    return summary


def write_scorecard(card: Scorecard, run: Run, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = out / "scorecard.json"
    sc.write_text(json.dumps(scorecard_to_safe_dict(card), indent=2))
    (out / "trace.summary.json").write_text(json.dumps(trace_summary(run), indent=2))
    return sc
