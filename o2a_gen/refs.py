"""Finds session-key references in instructions, transforms and SQL."""

from __future__ import annotations

import re
from typing import Any

_DOUBLE = re.compile(r"\{\{\s*([A-Za-z_]\w*)")
_SINGLE = re.compile(r"(?<!\{)\{([A-Za-z_]\w*)(?:[\[.][^{}]*)?\}(?!\})")
_SQL_STRING = re.compile(r"'(?:[^']|'')*'")
_SQL_PARAM = re.compile(r"(?<![:\w]):([A-Za-z_]\w*)")
_ENV = re.compile(r"^\$\{ENV:[A-Za-z_]\w*\}$")


def double_brace_refs(value: Any) -> set[str]:
    """Root keys of every {{ key... }} anywhere in a nested value."""
    return {m.group(1) for s in _strings(value) for m in _DOUBLE.finditer(s)}


def single_brace_refs(text: str) -> set[str]:
    """Root keys of {key} / {key[0].field} placeholders in an LlmAgent instruction."""
    return {m.group(1) for m in _SINGLE.finditer(text or "")}


def sql_params(query: str) -> set[str]:
    """:param bind names, ignoring string literals and ::casts."""
    return {m.group(1) for m in _SQL_PARAM.finditer(_SQL_STRING.sub("''", query or ""))}


def is_env_ref(url: str) -> bool:
    return bool(_ENV.match(str(url or "").strip()))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)
