"""Writes every planned agent's fields from its class's agent syntax and the inputs.

Nothing here knows any agent class. For each agent, the navigator browses the
compiled skill tree (procedure sections, tools, metadata, reference) for what the
agent needs; the model then writes the agent's fields following the agent-syntax
section for its class, taking every value from the documents found or the
procedure. A value the inputs do not provide is left as an empty placeholder and
reported. The only checks are class-independent: fields must be named in the
syntax, session-key references must exist when the agent runs, and a router must
send each branch to its agent. A failing answer is sent back with the error.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from o2a_gen.config import GenConfig
from o2a_gen.llm import LLMClient, extract_json
from o2a_gen.navigator import NavResult, navigate
from o2a_gen.plan import AgentNode, Plan
from o2a_gen.refs import double_brace_refs, single_brace_refs, sql_params

PLAN_FIELDS = {"name", "agent_class", "output_key", "sub_agents"}

GROUND_SYSTEM = """You write the fields of one agent YAML in an O2A workflow.

- The agent syntax for the agent's class is the only authority on which fields exist, which are
  required, and how each value is shaped (including fields written as an embedded YAML block,
  `field: |`, which you give as a string holding that YAML).
- Every value (tables, columns, SQL, URLs, methods, headers, auth, tool names, environment
  variables, models, prompts) comes from the documents provided: the procedure, the tools, the
  metadata and the reference material. Never invent one.
- When the syntax needs a value that the documents do not give, set it to an empty string ""
  and name it in `gaps`.
- Agents share data through session keys. Reference only the session keys listed as available,
  in the reference style the syntax shows for that field."""

GROUND_FORMAT = """Return one JSON object:
{{
  "fields": {{every field this agent needs per its syntax, except {plan_fields}, which the
             workflow plan sets}},
  "description": "one sentence",
  "gaps": ["each value the inputs did not provide, and where it belongs"]
}}"""


@dataclass
class Grounding:
    node: str
    step_id: str
    nav: NavResult | None
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    procedure_section: str = ""


def ground_plan(client: LLMClient, plan: Plan, cfg: GenConfig, catalog,
                max_workers: int = 4, proc_tree=None, procedure_text: str = "") -> list[Grounding]:
    """``catalog`` is a Catalog or CombinedCatalog (procedure tree + inputs).
    ``proc_tree`` (a proctree.ProcTree) locates each step's procedure section;
    ``procedure_text`` supplies the exact lines each step cites."""
    work = plan.nodes()
    lines = procedure_text.splitlines()
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(
            lambda n: ground_node(client, n, cfg, catalog, proc_tree, lines), work))


def cited_lines(lines: list[str], source_lines: str, limit: int = 150) -> str:
    """The procedure lines a step cites ("12" or "12-15"), numbered, verbatim."""
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", source_lines or "")
    if not m or not lines:
        return ""
    a = max(1, int(m.group(1)))
    b = min(len(lines), int(m.group(2) or a), a + limit - 1)
    return "\n".join(f"{i}| {lines[i - 1]}" for i in range(a, b + 1))


def ground_node(client: LLMClient, node: AgentNode, cfg: GenConfig, catalog,
                proc_tree=None, lines: list[str] | None = None) -> Grounding:
    step = node.step
    section, section_path = "", ""
    m = re.match(r"\d+", (step.source_lines if step else "") or "")
    if proc_tree is not None and m:
        section = proc_tree.context_for_line(int(m.group()), catalog.index)
        section_path = proc_tree.section_path_for_line(int(m.group()))
    syntax = catalog.syntax_for(node.agent_class)

    nav = None
    if step is not None:
        question = (f"Procedure step: {step.title}\n{step.text}\n"
                    f"It will be an O2A {node.agent_class}. Find the tools, tables and fields, "
                    "and reference material needed to write this agent's fields."
                    + (f"\n\n{section}" if section else ""))
        nav = navigate(client, catalog, question, model=cfg.model("navigate"),
                       max_turns=cfg.navigate_max_turns)

    prompt = "\n".join([
        f"Agent syntax for {node.agent_class}:\n{syntax or '(the syntax has no section for it)'}",
        "",
        f"Agent to write: `{node.name}` ({node.agent_class})",
        _what(node, cited_lines(lines or [], step.source_lines) if step else ""),
        section,
        f"output_key (set by the plan): {node.output_key or '(none)'}",
        f"Session keys available when it runs: {node.available_keys}",
        (f"Keys the procedure says it needs: {node.required_keys}" if node.required_keys else ""),
        _children(node),
        f"LLM model name for any model field: \"{cfg.models.get('llm_agent', '')}\"",
        "",
        f"Documents found in the inputs:\n{_docs_block(nav)}",
        "",
        GROUND_FORMAT.format(plan_fields=", ".join(sorted(PLAN_FIELDS))),
    ])

    def check(d):
        if not isinstance(d, dict) or not isinstance(d.get("fields"), dict):
            raise ValueError('reply with a JSON object holding a "fields" object')
        _check(node, d["fields"], syntax)

    data, err = _ask(client, cfg.model("ground"), prompt, check)
    warnings = [f"{node.name}: {err}"] if err else []
    gaps = data.get("gaps") or []
    gaps = [str(g) for g in (gaps if isinstance(gaps, list) else [gaps]) if str(g).strip()]
    warnings += _apply(node, data)
    return Grounding(node.name, step.id if step else "", nav, gaps, warnings, section_path)


def _ask(client: LLMClient, model: str, prompt: str, check, retries: int = 2):
    """Ask, check, and send failures back. After the last try the answer is kept with the
    problem reported, so one hard agent never stops the whole workflow."""
    messages = [{"role": "user", "content": prompt}]
    last: dict = {}
    err = ""
    for _ in range(retries + 1):
        try:
            reply = client.complete(model=model, system=GROUND_SYSTEM, messages=messages,
                                    max_tokens=8000)
        except ValueError as e:        # e.g. the model declined
            err = str(e)
            break
        try:
            data = extract_json(reply.text)
            if isinstance(data, dict):
                last = data
            check(data)
            return data, ""
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            err = str(e)
            messages = messages + [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": f"That output has a problem: {e}. "
                                            "Reply again with only the corrected JSON."}]
    if not isinstance(last.get("fields"), dict):
        last = {"fields": {}, "gaps": [f"no valid answer: {err}"]}
    return last, f"kept after {retries + 1} tries: {err}"


def _check(node: AgentNode, fields: dict, syntax: str) -> None:
    problems = []
    if syntax:
        unknown = [k for k in fields
                   if k not in PLAN_FIELDS and not re.search(rf"\b{re.escape(str(k))}\b", syntax)]
        if unknown:
            problems.append(f"fields {unknown} are not in the agent syntax for {node.agent_class}")
    refs = double_brace_refs(fields) | sql_params_all(fields)
    refs |= {r for s in _strings(fields) for r in single_brace_refs(s)}
    missing = refs - set(node.available_keys) - {node.output_key}
    if missing:
        problems.append(f"references {sorted(missing)}, which are not session keys available "
                        f"when it runs; available: {node.available_keys}")
    if node.branch_targets:
        text = json.dumps(fields)
        absent = [t.name for _, t in node.branch_targets if t.name not in text]
        if absent:
            problems.append(f"every branch must route to its agent; not routed: {absent}")
    if problems:
        raise ValueError("; ".join(problems))


def _apply(node: AgentNode, d: dict) -> list[str]:
    """Write the answer into node.fields; return warnings."""
    fields = {k: v for k, v in d["fields"].items() if k not in PLAN_FIELDS}
    if d.get("description"):
        node.description = str(d["description"]).strip()
    node.fields = fields
    warnings = []
    empty = [k for k, v in _leaves(fields) if v in ("", None)]
    if empty:
        warnings.append(f"{node.name}: empty placeholders (not in the inputs): {empty}")
    used = double_brace_refs(fields) | sql_params_all(fields)
    used |= {r for s in _strings(fields) for r in single_brace_refs(s)}
    used |= {v for _, v in _leaves(fields) if isinstance(v, str)}
    unread = [k for k in node.required_keys if k and k not in used]
    if unread:
        warnings.append(f"{node.name}: procedure says it needs {unread} but the agent does not "
                        "read them")
    return warnings


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def sql_params_all(value) -> set[str]:
    return {p for s in _strings(value) for p in sql_params(s)}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _leaves(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, value


def _what(node: AgentNode, cited: str = "") -> str:
    step = node.step
    if step is not None:
        lines = [f"Procedure step {step.id} (lines {step.source_lines}): {step.title}", step.text]
        if cited:
            lines.append(f"The procedure's own wording for this step:\n{cited}")
        if node.branch_targets:
            lines.append("Branches (condition in words -> agent to route to):")
            for (label, target), b in zip(node.branch_targets, step.branches):
                lines.append(f"- {label}: when {b.when} -> {target.name}")
        return "\n".join(lines)
    role = {"orchestrator": "It runs the whole workflow.",
            "group": "It runs its sub-agents one after another."}.get(node.role, "")
    return f"{node.description}\n{role}".strip()


def _children(node: AgentNode) -> str:
    if not node.children:
        return ""
    return ("sub_agents (set by the plan, in order): "
            + ", ".join(f"{c.name} ({c.agent_class})" for c in node.children))


def _docs_block(nav: NavResult | None) -> str:
    if nav is None:
        return "(not searched for this agent)"
    if not nav.docs:
        return f"(nothing relevant found: {nav.notes})"
    parts = [f"--- doc {did} ---\n{text[:6000]}" for did, text in nav.docs.items()]
    return "\n\n".join(parts) + (f"\n\nNavigator notes: {nav.notes}" if nav.notes else "")
