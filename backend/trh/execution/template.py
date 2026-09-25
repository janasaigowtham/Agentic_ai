"""Minimal template resolver for the fixture pipeline's two template
dialects: `{{ key[0].field }}` (transform args, with an optional [index] on
the first segment and dotted field access) and `{key}` (single-brace,
LLM instruction prompts). Resolves against the executor's session state --
the same dict database/transformation/LLM steps read and write while a
trajectory executes.
"""
from __future__ import annotations

import re
from typing import Any

_DOUBLE_BRACE_RE = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}"
)
_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_PATH_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[(\d+)\])?")


def resolve_path(path: str, session: dict[str, Any]) -> Any:
    """Resolves 'key', 'key.field', or 'key[0].field' against session."""
    value: Any = None
    for i, segment in enumerate(path.split(".")):
        match = _PATH_TOKEN_RE.match(segment)
        if not match:
            return None
        name, _, index = match.groups()
        if i == 0:
            if name not in session:
                return None
            value = session[name]
        else:
            if not isinstance(value, dict) or name not in value:
                return None
            value = value[name]
        if index is not None:
            if not isinstance(value, list) or int(index) >= len(value):
                return None
            value = value[int(index)]
    return value


def render_double_brace(text: str, session: dict[str, Any]) -> Any:
    """For a template that is *only* a single `{{ path }}` expression (the
    shape every $cond/$fn arg in this fixture uses), returns the resolved
    value directly rather than a stringified substitution -- callers need
    the real type (dict, list, None, ...), not text.
    """
    match = _DOUBLE_BRACE_RE.fullmatch(text.strip())
    if not match:
        return None
    return resolve_path(match.group(1), session)


def render_single_brace(text: str, session: dict[str, Any]) -> str:
    """Substitutes every `{key}` in `text` with its session value.

    Deliberately renders whatever the instruction references, declared in
    the node's input_keys or not -- that's what a real prompt template
    would do, and is exactly how an undeclared reference (spec section
    8.3 / the harness's N3-class finding) actually reaches a live prompt.
    """

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        value = session.get(key)
        return str(value) if value is not None else f"{{{key}:missing}}"

    return _SINGLE_BRACE_RE.sub(_sub, text)
