"""Database checks: S1, S2, S3, S4, S5, S6, S7, S8, R1, R5.

Uses sqlglot when importable; degrades to regex when it isn't. Never raises
because sqlglot is absent.
"""
from __future__ import annotations

import re

from ..core.graph import extract_sql_params, extract_template_keys
from ..core.models import Flaw, Severity
from ..core.registry import register

try:
    import sqlglot
    from sqlglot import exp

    HAS_SQLGLOT = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    sqlglot = None
    exp = None
    HAS_SQLGLOT = False

DB_CLASS = "database_agent"
REFERENCE_TABLE_HINTS = ("_lookup", "_config", "_ref", "_dim", "_codes")
MUTATING_RE = re.compile(r"\b(DELETE|UPDATE|INSERT|TRUNCATE|DROP|ALTER)\b", re.I)
SELECT_STAR_RE = re.compile(r"SELECT\s+\*", re.I)


def _db_nodes(graph):
    return [n for n in graph.nodes.values() if n.agent_class == DB_CLASS and n.query]


@register("S1", axis="structural", stage="static")
def sql_fails_to_parse(run, graph, config):
    """SQL fails to parse."""
    if not HAS_SQLGLOT:
        return []
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        dialect = node.db_type
        try:
            sqlglot.parse_one(node.query, dialect=dialect)
        except Exception as e:  # noqa: BLE001
            flaws.append(
                Flaw(
                    type="sql_parse_error",
                    severity=Severity.SEV_1,
                    node=node.name,
                    description=f"{node.name}'s query fails to parse as {dialect}.",
                    fix="Fix the SQL syntax, or correct the declared dialect.",
                    evidence={"error": str(e)[:300], "dialect": dialect},
                    agent_class=node.agent_class,
                )
            )
    return flaws


@register("S2", axis="structural", stage="static")
def sql_param_mismatch(run, graph, config):
    """:params vs input_keys mismatch."""
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        bound = extract_sql_params(node.query)
        declared = set(node.input_keys)
        undeclared = sorted(bound - declared)
        if undeclared:
            flaws.append(
                Flaw(
                    type="query_param_undeclared",
                    severity=Severity.SEV_1,
                    node=node.name,
                    description=f"{node.name} binds {undeclared} but doesn't declare them in input_keys.",
                    fix="Add the bound params to input_keys.",
                    evidence={
                        "bound_params": sorted(bound),
                        "declared_input_keys": sorted(declared),
                        "undeclared": undeclared,
                    },
                    agent_class=node.agent_class,
                )
            )
        unused = sorted(declared - bound - extract_template_keys(node.query))
        if unused:
            flaws.append(
                Flaw(
                    type="query_input_key_unused",
                    severity=Severity.SEV_3,
                    node=node.name,
                    description=f"{node.name} declares input_keys {unused} that the query never references.",
                    fix="Remove unused input_keys, or bind them in the query.",
                    evidence={"bound_params": sorted(bound), "declared_input_keys": sorted(declared), "unused": unused},
                    agent_class=node.agent_class,
                )
            )
    return flaws


@register("S5", axis="structural", stage="static")
def sql_injection_risk(run, graph, config):
    """{{ }} interpolated into SQL (injection risk)."""
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        refs = sorted(extract_template_keys(node.query))
        if not refs:
            continue
        flaws.append(
            Flaw(
                type="sql_injection_risk",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name} interpolates {refs} directly into SQL instead of binding.",
                fix="Use :param bound parameters instead of {{ }} interpolation.",
                evidence={"interpolated_refs": refs},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("S6", axis="structural", stage="static")
def sql_inlined_credentials(run, graph, config):
    """Connection URL inlined (not ${ENV:...})."""
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        if not node.db_url:
            continue
        if "${ENV:" in node.db_url or "${" in node.db_url:
            continue
        flaws.append(
            Flaw(
                type="sql_inlined_credentials",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name}'s connection URL is not sourced from an env reference.",
                fix="Reference the connection string via ${ENV:VAR_NAME}.",
                evidence={"has_env_ref": False},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("S8", axis="structural", stage="static")
def sql_mutating_statement(run, graph, config):
    """database_agent issues a mutating statement."""
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        m = MUTATING_RE.search(node.query)
        if not m:
            continue
        flaws.append(
            Flaw(
                type="sql_mutating_statement",
                severity=Severity.SEV_1,
                node=node.name,
                description=f"{node.name} issues a {m.group(1).upper()} statement.",
                fix="database_agent should be read-only; move mutations to a dedicated writer.",
                evidence={"statement": m.group(1).upper()},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("S3", axis="efficiency", stage="static")
def sql_select_star(run, graph, config):
    """SELECT *."""
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        if not SELECT_STAR_RE.search(node.query):
            continue
        flaws.append(
            Flaw(
                type="sql_select_star",
                severity=Severity.SEV_2,
                node=node.name,
                description=f"{node.name} uses SELECT * instead of naming columns.",
                fix="Name only the columns actually needed downstream.",
                evidence={"output_key": node.output_key},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("S4", axis="efficiency", stage="static")
def sql_unbounded_scan(run, graph, config):
    """No WHERE / LIMIT / QUALIFY against a likely fact table."""
    if not HAS_SQLGLOT:
        return []
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        try:
            tree = sqlglot.parse_one(node.query, dialect=node.db_type)
        except Exception:  # noqa: BLE001
            continue
        tables = [t.name for t in tree.find_all(exp.Table)]
        if any(hint in t.lower() for t in tables for hint in REFERENCE_TABLE_HINTS):
            continue
        if tree.find(exp.Where) or tree.find(exp.Limit) or tree.find(exp.Qualify):
            continue
        flaws.append(
            Flaw(
                type="sql_unbounded_scan",
                severity=Severity.SEV_2,
                node=node.name,
                description=f"{node.name} has no WHERE, LIMIT, or QUALIFY against {tables}.",
                fix="Add a WHERE clause or LIMIT to bound the scan.",
                evidence={"tables": tables},
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("S7", axis="efficiency", stage="static")
def sql_high_complexity(run, graph, config):
    """High structural complexity."""
    if not HAS_SQLGLOT:
        return []
    threshold = config.get("query_complexity_threshold", 6)
    flaws: list[Flaw] = []
    for node in _db_nodes(graph):
        try:
            tree = sqlglot.parse_one(node.query, dialect=node.db_type)
        except Exception:  # noqa: BLE001
            continue
        ctes = list(tree.find_all(exp.CTE))
        joins = list(tree.find_all(exp.Join))
        windows = list(tree.find_all(exp.Window))
        subqueries = list(tree.find_all(exp.Subquery))
        score = len(ctes) + len(joins) + len(windows) + len(subqueries)
        if score <= threshold:
            continue
        flaws.append(
            Flaw(
                type="sql_high_complexity",
                severity=Severity.SEV_3,
                node=node.name,
                description=f"{node.name}'s query has structural complexity {score} (threshold {threshold}).",
                fix="Split into simpler queries or precompute part of this as a view.",
                evidence={
                    "score": score,
                    "ctes": len(ctes),
                    "joins": len(joins),
                    "windows": len(windows),
                    "subqueries": len(subqueries),
                    "threshold": threshold,
                },
                agent_class=node.agent_class,
            )
        )
    return flaws


@register("R1", axis="efficiency", stage="runtime")
def zero_rows_indexed(run, graph, config):
    """Query returned 0 rows while a downstream node indexes [0]."""
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != DB_CLASS or span.row_count != 0:
            continue
        output_key = span.output_key
        if not output_key:
            continue
        needle = f"{output_key}[0]"
        indexers = []
        for other in graph.nodes.values():
            haystacks = [str(other.transform or ""), other.instruction or ""]
            if any(needle in h for h in haystacks):
                indexers.append(other.name)
        severity = Severity.SEV_1 if indexers else Severity.SEV_2
        flaws.append(
            Flaw(
                type="zero_rows_indexed",
                severity=severity,
                node=span.agent_name,
                description=f"{span.agent_name} returned 0 rows for {output_key}.",
                fix="Guard downstream [0] indexing against an empty result set.",
                evidence={
                    "output_key": output_key,
                    "downstream_indexers": indexers,
                    "row_count": 0,
                },
                agent_class=span.agent_class,
            )
        )
    return flaws


@register("R5", axis="efficiency", stage="runtime")
def row_count_over_threshold(run, graph, config):
    """Result set over row threshold."""
    threshold = config.get("db_row_count_threshold", 1000)
    flaws: list[Flaw] = []
    for span in run.spans:
        if span.agent_class != DB_CLASS or span.row_count is None:
            continue
        if span.row_count <= threshold:
            continue
        flaws.append(
            Flaw(
                type="row_count_over_threshold",
                severity=Severity.SEV_2,
                node=span.agent_name,
                description=f"{span.agent_name} returned {span.row_count} rows (threshold {threshold}).",
                fix="Add filtering, pagination, or a LIMIT to bound the result set.",
                evidence={"row_count": span.row_count, "threshold": threshold},
                agent_class=span.agent_class,
            )
        )
    return flaws
