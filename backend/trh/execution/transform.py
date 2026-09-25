"""Evaluates the fixture pipeline's declared `transform` specs (`$cond` /
`$fn`) against the executor's session state.

This is intentionally small and pipeline-scoped: it implements exactly the
two transform shapes the bundled agent YAMLs declare, not a general
transform-DSL interpreter. A new shape in a new YAML would need a new case
here, the same way a real target's transformation runtime would need one.
"""
from __future__ import annotations

from typing import Any, Callable

from trh.execution.template import render_double_brace

Args = dict[str, Any]


def _fn_extract_field(args: Args) -> Any:
    return args.get("source")


def _fn_score_investor(args: Args) -> Any:
    source = args.get("source")
    if source is None:
        return None
    # Deterministic synthetic score, not a real underwriting calculation.
    return 50 + (sum(ord(c) for c in str(source)) % 50)


def _fn_build_icmp_result(args: Args) -> Any:
    return {"status": "required", "classification": args.get("source")}


def _fn_build_skip_result(args: Args) -> Any:
    return {"status": "skipped", "classification": args.get("source")}


def _fn_note_review(args: Args) -> Any:
    return f"Investor review flag: {args.get('source')}"


_FUNCTIONS: dict[str, Callable[[Args], Any]] = {
    "extract_field": _fn_extract_field,
    "score_investor": _fn_score_investor,
    "build_icmp_result": _fn_build_icmp_result,
    "build_skip_result": _fn_build_skip_result,
    "note_review": _fn_note_review,
}

_OPERATORS: dict[str, Callable[[Any], bool]] = {
    "not_null": lambda v: v is not None,
    "is_null": lambda v: v is None,
}


def _resolve_arg(template: Any, session: dict[str, Any]) -> Any:
    if isinstance(template, str):
        return render_double_brace(template, session)
    return template


def evaluate_transform(transform_spec: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    """Returns {output_key: value} for every key in the node's transform dict."""
    return {output_key: _evaluate_one(spec, session) for output_key, spec in transform_spec.items()}


def _evaluate_one(spec: dict[str, Any], session: dict[str, Any]) -> Any:
    if "$cond" in spec:
        condition = spec["$cond"]["if"]
        left = _resolve_arg(condition["left"], session)
        operator = _OPERATORS.get(condition["op"])
        if operator is None:
            raise ValueError(f"unsupported transform operator: {condition['op']!r}")
        return spec["$cond"]["then"] if operator(left) else spec["$cond"]["else"]
    if "$fn" in spec:
        fn = _FUNCTIONS.get(spec["$fn"])
        if fn is None:
            raise ValueError(f"unsupported transform function: {spec['$fn']!r}")
        args = {k: _resolve_arg(v, session) for k, v in (spec.get("args") or {}).items()}
        return fn(args)
    raise ValueError(f"unsupported transform spec: {spec!r}")
