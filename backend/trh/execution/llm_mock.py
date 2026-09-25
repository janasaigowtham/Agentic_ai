"""Synthetic LlmAgent behavior for execution.

Renders the node's instruction template against session state exactly like
a real call would -- including any key the instruction references whether
or not it's in the node's declared input_keys, since that's what a real
prompt render does (and exactly how an undeclared reference actually
reaches a live prompt). Produces a plausible structured output for
whatever JSON contract the instruction asks for at its end
("Return strict JSON: {...}"), without hand-authoring per-pipeline logic:
field names are matched against a few generic heuristics, not this specific
pipeline's schema.
"""
from __future__ import annotations

import re
from typing import Any

from trh.execution.template import render_single_brace

_JSON_CONTRACT_RE = re.compile(r"Return strict JSON:\s*\{(.*?)\}\s*$", re.IGNORECASE | re.DOTALL)
_FIELD_NAME_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:')


def _synthesize_field(field_name: str) -> Any:
    lowered = field_name.lower()
    if "answer" in lowered:
        return "Y"
    if "verdict" in lowered:
        return "approved"
    if "rationale" in lowered or "reason" in lowered or "note" in lowered:
        return f"Synthetic {field_name} generated from the structured session fields available at this step."
    return f"synthetic-{field_name}"


def synthesize_llm_call(instruction: str, session: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Returns (rendered_prompt, structured_output)."""
    rendered_prompt = render_single_brace(instruction, session)
    contract_match = _JSON_CONTRACT_RE.search(instruction)
    output: dict[str, Any] = {}
    if contract_match:
        for field_match in _FIELD_NAME_RE.finditer(contract_match.group(1)):
            field_name = field_match.group(1)
            output[field_name] = _synthesize_field(field_name)
    return rendered_prompt, output
