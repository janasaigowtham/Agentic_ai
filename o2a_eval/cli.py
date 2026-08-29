"""o2a-eval CLI: score | lint | graph | checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import checks as _checks  # noqa: F401 - registers checks
from .core.graph import load_pipeline
from .core.models import Run, Severity
from .core.registry import all_checks
from .core.rollup import build_scorecard
from .core.trace import build_run, load_jsonl
from .report.scrubber import scorecard_to_safe_dict, write_scorecard
from .report.terminal import render_scorecard

KIND_MARKERS = {
    "non_deterministic": "◆",  # ◆
    "control_flow": "▸",  # ▸
    "deterministic": "·",  # ·
}


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _add_common_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agent-dir", required=True)
    p.add_argument("--config")
    p.add_argument("--fail-on", default="SEV-1")


def cmd_score(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    if args.no_judge:
        config["_judge"] = None
    if args.latency_budget_s is not None:
        config["latency_budget_s"] = args.latency_budget_s

    records = load_jsonl(args.trace)
    run = build_run(records)
    pipeline_name = args.pipeline or run.pipeline_name
    graph = load_pipeline(args.agent_dir, pipeline_name)

    card = build_scorecard(run, graph, config)

    if args.output_dir:
        write_scorecard(card, run, args.output_dir)

    if args.json:
        print(json.dumps(scorecard_to_safe_dict(card), indent=2))
    else:
        render_scorecard(card)

    fail_on = Severity.parse(args.fail_on)
    return 0 if card.max_severity < fail_on else 1


def cmd_lint(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    graph = load_pipeline(args.agent_dir, args.pipeline)
    empty_run = Run(
        trace_id="static",
        pipeline_name=graph.name,
        pipeline_hash=graph.pipeline_hash,
        spans=[],
    )
    card = build_scorecard(empty_run, graph, config)
    render_scorecard(card)
    fail_on = Severity.parse(args.fail_on)
    return 0 if card.max_severity < fail_on else 1


def cmd_graph(args: argparse.Namespace) -> int:
    graph = load_pipeline(args.agent_dir, args.pipeline)

    def _print(node, depth: int) -> None:
        marker = KIND_MARKERS.get(node.kind, "?")
        suffix = f" → {node.output_key}" if node.output_key else ""
        print("  " * depth + f"{marker} {node.name} [{node.agent_class}]{suffix}")
        for child in node.children:
            _print(child, depth + 1)

    _print(graph.root, 0)

    total = len(graph.nodes)
    deterministic = len(graph.deterministic())
    non_deterministic = len(graph.non_deterministic())
    print()
    print(f"nodes: {total} total, {deterministic} deterministic, {non_deterministic} non-deterministic")
    print(f"pipeline_hash: {graph.pipeline_hash}")
    return 0


def cmd_checks(args: argparse.Namespace) -> int:
    metas = all_checks()
    by_stage: dict[str, list] = {"static": [], "runtime": [], "cross_trace": []}
    for meta in metas:
        by_stage[meta.stage].append(meta)

    for stage in ("static", "runtime", "cross_trace"):
        items = by_stage[stage]
        if not items:
            continue
        print(f"-- {stage.upper()} --")
        for meta in items:
            judge_marker = " [judge]" if meta.needs_judge else ""
            print(f"{meta.id:8s} {meta.axis:16s} {meta.description}{judge_marker}")
        print()

    print(f"{len(metas)} checks registered")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="o2a-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score")
    p_score.add_argument("trace")
    _add_common_score_args(p_score)
    p_score.add_argument("--pipeline")
    p_score.add_argument("--output-dir")
    p_score.add_argument("--json", action="store_true")
    p_score.add_argument("--no-judge", action="store_true")
    p_score.add_argument("--latency-budget-s", type=float, default=None)
    p_score.set_defaults(func=cmd_score)

    p_lint = sub.add_parser("lint")
    _add_common_score_args(p_lint)
    p_lint.add_argument("--pipeline", required=True)
    p_lint.set_defaults(func=cmd_lint)

    p_graph = sub.add_parser("graph")
    p_graph.add_argument("--agent-dir", required=True)
    p_graph.add_argument("--pipeline", required=True)
    p_graph.set_defaults(func=cmd_graph)

    p_checks = sub.add_parser("checks")
    p_checks.set_defaults(func=cmd_checks)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
