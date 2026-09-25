"""Mock source-system database: synthetic tables plus a best-effort query
executor good enough to answer the fixture pipeline's declared SQL (a flat
SELECT column list, one FROM table, one bind param in WHERE). It is not a
SQL engine -- it exists so a database_agent node can actually run against
realistic-shaped synthetic data instead of only replaying a hand-authored
trace.
"""
from __future__ import annotations

import re
from typing import Any

_SELECT_FROM_RE = re.compile(r"SELECT\s+(.*?)\s+FROM\s+([A-Za-z0-9_\.]+)", re.IGNORECASE | re.DOTALL)
_AS_ALIAS_RE = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)
_TRAILING_IDENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\)?\s*$")


def _parse_select_columns(query: str) -> list[str]:
    match = _SELECT_FROM_RE.search(query)
    if not match:
        return []
    columns: list[str] = []
    for part in match.group(1).split(","):
        part = part.strip()
        alias_match = _AS_ALIAS_RE.search(part)
        if alias_match:
            columns.append(alias_match.group(1).lower())
            continue
        tail_match = _TRAILING_IDENT_RE.search(part)
        if tail_match:
            columns.append(tail_match.group(1).lower())
    return columns


def _table_name(query: str) -> str | None:
    match = _SELECT_FROM_RE.search(query)
    if not match:
        return None
    return match.group(2).split(".")[-1].lower()


class MockDatabase:
    """An in-memory stand-in for whatever source system(s) database_agent
    nodes declare a query against.

    `execute` projects matched rows onto the query's own SELECT column
    list, so the mock respects each node's declared contract rather than
    leaking columns (e.g. a seeded PII field) the query never asked for.
    """

    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._key_columns: dict[str, str] = {}

    def seed_table(self, table_name: str, rows: list[dict[str, Any]], key_column: str) -> None:
        self._tables[table_name.lower()] = rows
        self._key_columns[table_name.lower()] = key_column

    def execute(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        table = _table_name(query)
        if table is None or table not in self._tables:
            return []
        key_column = self._key_columns[table]
        # Every query in this fixture filters on exactly one bind param
        # against one key column -- the only shape this mock needs to support.
        filter_value = next(iter(params.values()), None)
        matched = [row for row in self._tables[table] if str(row.get(key_column)) == str(filter_value)]
        columns = _parse_select_columns(query)
        if not columns:
            return matched
        return [{col: row.get(col) for col in columns if col in row} for row in matched]
