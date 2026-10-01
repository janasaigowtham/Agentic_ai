"""Playbook-driven generation: procedure + tools + metadata + agent syntax + playbook
-> workflow YAMLs.

The playbook says HOW to build a workflow (how to check the logic, how to orchestrate,
how to map data from tools). The procedure says WHAT to check, the tools and metadata
supply the real values, and the agent syntax gives each YAML its shape. This module
knows nothing about any particular procedure, tool or agent class: every stage is a
model call that applies the playbook to the inputs, and code only checks what each
stage returns and feeds the problems back.

Stages (each saves its result next to the output folder):
  recipe       the playbook as a build checklist: roles, conditions, order, checks
  logic        the procedure's checks, rules and outcomes          -> <out>_flow.json/.md
  datamap      every value the checks read, mapped to tools/metadata -> <out>_data_map.json/.md
  orchestrate  the agents, classes, keys, order and branches        -> <out>_plan.json/.md
  write        one YAML per agent, values copied from the sources   -> <out>/*.yaml
  review       code checks + a fresh model review; blocking problems are sent back to
               the writer (generation.review_rounds, default 2)     -> <out>_review.json

The playbook and the agent syntax are the system prompt of every call, so they are
cached once and read cheaply by every stage and every writer.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from o2a_gen.config import GenConfig
from o2a_gen.emit import dump_yaml
from o2a_gen.llm import LLMClient, complete_json
from o2a_gen.refs import double_brace_refs, single_brace_refs, sql_params
from o2a_gen.validate import validate_dir

OUTCOME_STATUSES = ("available", "derived", "missing")
_DEFAULT_EFFORT = {"recipe": "high", "logic": "high", "datamap": "high",
                   "orchestrate": "high", "write": "medium", "review": "high"}

COMMON_SYSTEM = """You build O2A agent workflows (one YAML file per agent) for an operations
review. Four kinds of input decide everything:

- The PLAYBOOK (below) says how to build the workflow: how to check the logic, how to
  orchestrate the agents and how to map data from tools. Follow it exactly.
- The AGENT SYNTAX (below) is the only authority on which agent classes exist, which fields
  each class has and how each value is written.
- The PROCEDURE (given in the task) says what the review checks.
- The TOOLS and METADATA (given in the task) supply every concrete value: tool names,
  URLs, methods, headers, auth, SQL, tables, columns, field names, environment variables.

Never invent a value. When a value the syntax needs is not in the inputs, leave it empty
and report it as a gap. Reply with JSON only, exactly in the shape the task asks for.

==================== PLAYBOOK ====================
{playbook}

==================== AGENT SYNTAX ====================
{syntax}
"""

RECIPE_TASK = """TASK: recipe

Turn the PLAYBOOK into a build checklist that the later stages will follow and that the
final review will check against. Use only what the playbook says.

Agent classes the syntax defines: {classes}

Return:
{{
  "roles": [                      every agent role the playbook describes
    {{"role": "role name as the playbook writes it (with its placeholders)",
      "agent_class": "class from the syntax (or several, joined by ' or ')",
      "when": "always | the condition under which the playbook adds it",
      "purpose": "what it does",
      "output_key": "the output key pattern the playbook gives, or empty",
      "count": "one per workflow | one per attribute | as needed (how many the playbook allows)"}}
  ],
  "precedence": ["which input wins when inputs disagree (on business rules, package shape,
                  names and integration values), as the playbook states it"],
  "shared_keys": ["session keys the playbook says several agents may write"],
  "orchestration": ["rules for ordering and nesting agents, branches, gates"],
  "logic_rules": ["rules for how the review logic is checked and by which agents"],
  "data_rules": ["rules for mapping data from tools and metadata"],
  "naming_rules": ["rules for agent names, keys and prefixes"],
  "checks": ["every requirement the finished workflow must meet, one per item"]
}}"""

LOGIC_TASK = """TASK: logic

Apply the playbook's logic rules (recipe below) to the PROCEDURE. Find the parts of the
procedure that define what the review checks (for example attribute or testing
instructions) and turn each into its checks, in the procedure's order. Ignore parts that
do not define checks (status, sampling, certification, revision history), unless the
playbook says otherwise.

Recipe:
{recipe}

Procedure section outline:
{outline}

Procedure (each line prefixed with its number):
{procedure}

Return:
{{
  "attributes": [
    {{"id": "the attribute / question id from the procedure (e.g. Q12345)",
      "number": "two-digit sequence number: 01, 02, ...",
      "question": "the question the attribute answers",
      "scope": "what is in scope",
      "source_lines": "first-last line of the attribute",
      "outcomes": ["every final outcome the attribute can reach"],
      "checks": [
        {{"id": "1", "title": "short title", "source_lines": "a-b",
          "reads": [{{"what": "value the check reads", "where": "screen / system / document"}}],
          "rules": [{{"if": "condition", "then": "outcome, or 'next' / 'go to <check id>'"}}],
          "external": ["outside systems or documents the check needs, e.g. a document repository"]}}
      ]}}
  ],
  "unclear": ["anything in the procedure that is ambiguous, and how you read it"]
}}"""

DATAMAP_TASK = """TASK: datamap

Apply the playbook's data rules (recipe below) to the flow. For every value the checks read
and every outside system they need, find where it comes from in the TOOLS and METADATA.
Use the exact tool, field and column names that appear there.

Recipe:
{recipe}

Flow:
{flow}

TOOLS:
{tools}

METADATA:
{metadata}

Return:
{{
  "fields": [
    {{"need": "the value, as the flow names it", "checks": ["<attribute id>:<check id>", ...],
      "status": "available | derived | missing",
      "tool": "tool that returns it (available)", "field": "exact field / column name (available)",
      "derivation": "how it is computed and from which fields (derived)",
      "evidence": "where in the tools/metadata this is stated"}}
  ],
  "external": [
    {{"system": "outside system the flow needs", "tools": ["exact tool names that serve it"],
      "status": "available | missing", "notes": "order, async steps, gates the tools/metadata describe"}}
  ]
}}"""

ORCHESTRATE_TASK = """TASK: orchestrate

Apply the playbook's roles and orchestration rules (recipe below) to the flow and the data
map: decide every agent of the workflow. Use agent and tool names from the TOOLS and
METADATA where they name them; otherwise follow the playbook's naming rules with the
prefix "{prefix}". Every check in the flow must be covered by at least one agent.

Rules for the plan:
- Every agent fills exactly one recipe role: copy the role name as the recipe writes it.
  A step that only the TOOLS or METADATA define (not a playbook role) uses
  "tool: <step name as the tools/metadata write it>".
- When inputs disagree, follow the recipe's precedence rules. When the TOOLS or METADATA
  name an agent or stage for a role, use that name for the role; never add a second agent
  for the same role.
- Respect each role's count (one per workflow / one per attribute).
- Two agents may write the same session key only if they sit on different branches of the
  same router (only one of them runs), or the key is one of the recipe's shared keys.
{feedback}
Agent classes the syntax defines: {classes}
Pipeline inputs (supplied by the caller): {inputs}

Recipe:
{recipe}

Flow:
{flow}

Data map:
{datamap}

TOOLS:
{tools}

METADATA:
{metadata}

Return:
{{
  "name": "workflow name",
  "pipeline_inputs": ["session keys the caller supplies"],
  "root": "name of the top-level agent",
  "agents": [
    {{"name": "agent name", "agent_class": "one class from the syntax",
      "role": "the recipe role it fills (or 'tool: <step>')", "purpose": "one sentence",
      "input_keys": ["session keys it reads"], "output_key": "session key it writes, or empty",
      "sub_agents": ["children in execution order (containers and routers)"],
      "routes": [{{"target": "child name", "when": "condition in words"}}],
      "covers": ["<attribute id>:<check id> handled by this agent"],
      "sources": ["exact tool / metadata names its values come from"],
      "procedure_lines": "a-b, if it implements procedure text",
      "playbook_rule": "the playbook rule that creates it"}}
  ]
}}"""

WRITE_TASK = """TASK: write

Write the fields of ONE agent of the workflow, following its class's syntax and the
playbook. Copy values from the source documents exactly. Reference only the session keys
listed as available, in the reference style the syntax shows for that field. Where the
syntax needs a value the sources do not give, set it to "" and name it in gaps.

Agent (from the plan):
{agent}

Whole plan (name, class, output key):
{plan}

Session keys available when it runs: {available}

Syntax for {agent_class}:
{class_syntax}

Checks it implements:
{checks}

Data map:
{datamap}

Procedure text it implements:
{procedure_lines}

Source documents:
{sources}
{feedback}
Return:
{{
  "fields": {{every field this agent needs per its syntax, except {plan_fields}, which the
             plan sets}},
  "description": "one sentence",
  "gaps": ["each value for THIS agent that the inputs did not provide, and where it belongs"],
  "input_gaps": ["gaps in the input documents themselves that affect many agents, e.g. the
                  syntax does not document a function's argument names"],
  "plan_issues": ["problems with the plan that you cannot fix in this agent's fields
                   (wrong children, order, keys, duplicate roles); leave the fields as the
                   plan says and report them here"]
}}"""

REVIEW_TASK = """TASK: review

Review the generated workflow as a fresh reviewer. Check it against: the recipe's checks
(the playbook), the flow (every check and rule implemented, in order, with every outcome
reachable), the data map (values come from the mapped tools and fields; missing data is
handled the way the playbook says), and the agent syntax. Also consider the code checks
below. Report only real problems; say exactly how to fix each one.

Recipe:
{recipe}

Flow:
{flow}

Data map:
{datamap}

Code checks:
{code_findings}

Workflow YAMLs:
{yamls}

Return:
{{
  "findings": [
    {{"agent": "agent name (or '*' for the whole workflow)",
      "severity": "blocking | minor",
      "level": "plan (which agents exist, their classes, children, order, routes targets,
                input/output keys) | agent (the fields inside one agent's YAML)",
      "problem": "what is wrong", "evidence": "playbook rule / flow check / syntax rule",
      "fix": "exactly what to change"}}
  ]
}}"""

PLAN_FIELDS = ("name", "agent_class", "input_keys", "output_key", "sub_agents")


@dataclass
class Inputs:
    procedure: str
    playbook: str
    syntax: str
    tools: str
    metadata: str
    classes: list[str]
    outline: str = ""


@dataclass
class AgentResult:
    name: str
    fields: dict = field(default_factory=dict)
    description: str = ""
    gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    input_gaps: list[str] = field(default_factory=list)
    plan_issues: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- inputs

def raw_inputs(catalog, kind: str) -> str:
    """The original files of one input kind, whole, each under its file name. Falls back to
    the compiled documents when the original files are gone."""
    out, seen = [], set()
    for did, meta in catalog.index.items():
        if meta.get("kind") != kind:
            continue
        src = Path(str(meta.get("source", "")))
        if src.is_file() and src not in seen:
            seen.add(src)
            out.append(f"----- {src.name} -----\n{src.read_text(encoding='utf-8', errors='replace')}")
        elif not src.is_file():
            out.append(f"----- {meta.get('name', did)} -----\n{catalog.get(did) or ''}")
    return "\n\n".join(out).strip()


def has_playbook(catalog) -> bool:
    return any(m.get("kind") == "playbook" for m in catalog.index.values())


def load_inputs(catalog, procedure_text: str, outline: str = "") -> Inputs:
    return Inputs(procedure=procedure_text, playbook=raw_inputs(catalog, "playbook"),
                  syntax=raw_inputs(catalog, "schema") or catalog.syntax_document(),
                  tools=raw_inputs(catalog, "tool"), metadata=raw_inputs(catalog, "metadata"),
                  classes=catalog.classes(), outline=outline)


# --------------------------------------------------------------------------- checks

def _lines_ok(s: str, n: int) -> bool:
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(s or ""))
    return bool(m) and 1 <= int(m.group(1)) <= int(m.group(2) or m.group(1)) <= n


def _class_tokens(value) -> list[str]:
    vals = value if isinstance(value, list) else [value]
    return [t for v in vals for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(v))]


def check_recipe(data, classes: set[str]) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("roles"), list) or not data["roles"]:
        raise ValueError('expected an object with a non-empty "roles" list')
    bad = []
    for r in data["roles"]:
        if not isinstance(r, dict) or not r.get("role"):
            raise ValueError('every role needs a "role" name')
        tokens = _class_tokens(r.get("agent_class", ""))
        if tokens and not any(t in classes for t in tokens):
            bad.append(f"{r['role']}: {r.get('agent_class')!r}")
    if bad:
        raise ValueError(f"agent_class not in the syntax ({sorted(classes)}): " + "; ".join(bad[:8]))
    for k in ("orchestration", "logic_rules", "data_rules", "checks"):
        if not isinstance(data.get(k, []), list):
            raise ValueError(f'"{k}" must be a list')


def check_flow(data, n_lines: int) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("attributes"), list) \
            or not data["attributes"]:
        raise ValueError('expected an object with a non-empty "attributes" list')
    problems = []
    for a in data["attributes"]:
        aid = a.get("id") if isinstance(a, dict) else None
        if not aid or not isinstance(a.get("checks"), list) or not a["checks"]:
            problems.append(f"attribute {aid!r} needs an id and a non-empty checks list")
            continue
        if not _lines_ok(a.get("source_lines"), n_lines):
            problems.append(f"{aid}: source_lines {a.get('source_lines')!r} not in 1-{n_lines}")
        ids = set()
        for c in a["checks"]:
            cid = str(c.get("id", "")) if isinstance(c, dict) else ""
            if not cid or cid in ids:
                problems.append(f"{aid}: check ids must be present and unique ({cid!r})")
            ids.add(cid)
            if not isinstance(c.get("rules"), list) or not c["rules"]:
                problems.append(f"{aid}:{cid} has no rules")
            if not _lines_ok(c.get("source_lines"), n_lines):
                problems.append(f"{aid}:{cid} source_lines {c.get('source_lines')!r} not in 1-{n_lines}")
    if problems:
        raise ValueError("; ".join(problems[:12]))


def check_datamap(data, flow: dict, sources_text: str) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("fields"), list):
        raise ValueError('expected an object with a "fields" list')
    text = sources_text.lower()
    problems = []
    for f in data["fields"]:
        if not isinstance(f, dict):
            problems.append("each field must be an object")
            continue
        st = f.get("status")
        if st not in OUTCOME_STATUSES:
            problems.append(f"{f.get('need')!r}: status must be one of {OUTCOME_STATUSES}")
        elif st == "available":
            for k in ("tool", "field"):
                v = str(f.get(k) or "").strip()
                if not v:
                    problems.append(f"{f.get('need')!r}: available but no {k}")
                elif v.lower() not in text:
                    problems.append(f"{f.get('need')!r}: {k} {v!r} does not appear in the tools "
                                    "or metadata; use the exact name, or mark it missing")
        elif st == "derived" and not str(f.get("derivation") or "").strip():
            problems.append(f"{f.get('need')!r}: derived but no derivation")
    for e in data.get("external") or []:
        for t in (e.get("tools") or []) if isinstance(e, dict) else []:
            if str(t).lower() not in text:
                problems.append(f"external {e.get('system')!r}: tool {t!r} is not in the tools "
                                "or metadata")
    needed = sum(len(c.get("reads") or []) for a in flow.get("attributes", [])
                 for c in a.get("checks", []))
    if needed and not data["fields"]:
        problems.append(f"the flow reads {needed} values but no field is mapped")
    if problems:
        raise ValueError("; ".join(problems[:12]))


def _norm_role(r) -> str:
    return re.sub(r"\s+", " ", str(r or "")).strip().strip("`").lower()


def _branches(agents: list[dict], root: str) -> dict[str, dict[str, str]]:
    """agent -> {router name: the child (branch) of that router it sits under}. An agent
    with routes is a router; its children are mutually exclusive branches."""
    return _walk_tree(agents, root)[0]


def _ancestors(agents: list[dict], root: str) -> dict[str, set[str]]:
    """agent -> the containers above it."""
    return _walk_tree(agents, root)[1]


def _walk_tree(agents: list[dict], root: str):
    by = {a.get("name"): a for a in agents}
    branches: dict[str, dict[str, str]] = {}
    above: dict[str, set[str]] = {}

    def walk(n, path, up, seen):
        if n in seen or n not in by:
            return
        branches.setdefault(n, dict(path))
        above.setdefault(n, set(up))
        a = by[n]
        for c in a.get("sub_agents") or []:
            sub = {**path, n: str(c)} if a.get("routes") else path
            walk(str(c), sub, up | {n}, seen | {n})

    walk(root, {}, set(), set())
    return branches, above


def check_plan(data, classes: set[str], flow: dict, inputs: list[str],
               recipe: dict | None = None) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("agents"), list) or not data["agents"]:
        raise ValueError('expected an object with a non-empty "agents" list')
    agents = data["agents"]
    names = [str(a.get("name", "")) for a in agents if isinstance(a, dict)]
    problems = []
    if len(names) != len(agents) or not all(re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names):
        problems.append("every agent needs a name made of letters, digits and underscores")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"duplicate agent names {dupes}")
    known = set(names)
    referenced = set()
    writers = {str(a.get("output_key")) for a in agents if a.get("output_key")}
    pipeline_inputs = set(inputs) | set(data.get("pipeline_inputs") or [])
    for a in agents:
        n = a.get("name")
        if a.get("agent_class") not in classes:
            problems.append(f"{n}: agent_class {a.get('agent_class')!r} is not in the syntax "
                            f"({sorted(classes)})")
        subs = [str(s) for s in a.get("sub_agents") or []]
        referenced |= set(subs)
        for s in subs:
            if s not in known:
                problems.append(f"{n}: sub_agent {s!r} is not a planned agent")
        for r in a.get("routes") or []:
            t = str(r.get("target", "")) if isinstance(r, dict) else ""
            referenced.add(t)
            if t not in subs:
                problems.append(f"{n}: route target {t!r} must also be in its sub_agents")
        for k in a.get("input_keys") or []:
            if k not in writers and k not in pipeline_inputs:
                problems.append(f"{n}: reads {k!r}, which no agent writes and is not a pipeline input")
    roots = [x for x in names if x not in referenced]
    if data.get("root") not in known:
        problems.append(f"root {data.get('root')!r} is not a planned agent")
    elif roots != [data["root"]]:
        problems.append(f"exactly one agent may be unreferenced (the root); found {roots}")
    if recipe:
        problems += _role_problems(agents, recipe, flow)
        problems += _writer_problems(agents, data.get("root"), recipe)
    covered = {str(c) for a in agents for c in a.get("covers") or []}
    missing = [f"{a['id']}:{c['id']}" for a in flow.get("attributes", [])
               for c in a.get("checks", []) if f"{a['id']}:{c['id']}" not in covered]
    if missing:
        problems.append(f"checks not covered by any agent: {missing[:10]}")
    if problems:
        raise ValueError("; ".join(problems[:14]))


def _role_problems(agents: list[dict], recipe: dict, flow: dict) -> list[str]:
    roles = {_norm_role(r.get("role")): r for r in recipe.get("roles") or [] if isinstance(r, dict)}
    n_attr = max(1, len(flow.get("attributes") or []))
    problems, count = [], {}
    for a in agents:
        role = _norm_role(a.get("role"))
        if not role:
            problems.append(f"{a.get('name')}: no role; copy a recipe role name or use "
                            "'tool: <step>'")
        elif role not in roles and not role.startswith("tool:"):
            problems.append(f"{a.get('name')}: role {a.get('role')!r} is not a recipe role "
                            f"({sorted(roles)}) nor 'tool: <step>'")
        count.setdefault(role, []).append(a.get("name"))
    for role, names in count.items():
        limit = str((roles.get(role) or {}).get("count", "")).lower()
        allowed = 1 if "workflow" in limit else n_attr if "attribute" in limit else None
        if allowed is not None and len(names) > allowed:
            problems.append(f"role {role!r} is filled {len(names)} times ({names}); the playbook "
                            f"allows {limit}")
    return problems


def _pattern(p: str) -> re.Pattern:
    """A key pattern such as '<prefix>_answer_q<NN>' as a regex; placeholders match a name."""
    parts = re.split(r"<[^>]+>", str(p).strip().strip("`"))
    return re.compile("^" + "[A-Za-z0-9_]+".join(re.escape(x) for x in parts) + "$")


def _shared_patterns(recipe: dict) -> list[re.Pattern]:
    """Keys several agents may write: the recipe's shared keys, plus any output-key pattern
    the playbook gives to more than one role (e.g. a first-pass and a final evaluator)."""
    pats = [str(k) for k in recipe.get("shared_keys") or [] if str(k).strip()]
    outs = [str(r.get("output_key") or "").strip() for r in recipe.get("roles") or []
            if isinstance(r, dict)]
    pats += [o for o in set(outs) if o and outs.count(o) > 1]
    return [_pattern(x) for x in pats]


def _writer_problems(agents: list[dict], root, recipe: dict) -> list[str]:
    shared = _shared_patterns(recipe)
    branch_of = _branches(agents, str(root))
    above = _ancestors(agents, str(root))
    writers: dict[str, list[str]] = {}
    for a in agents:
        k = a.get("output_key")
        if k and not a.get("routes") and not any(p.match(str(k)) for p in shared):
            writers.setdefault(str(k), []).append(a.get("name"))
    problems = []
    for key, ws in writers.items():
        for i, x in enumerate(ws):
            for y in ws[i + 1:]:
                if x in above.get(y, set()) or y in above.get(x, set()):
                    continue      # a container reporting its own child's key
                px, py = branch_of.get(x, {}), branch_of.get(y, {})
                if not any(r in py and px[r] != py[r] for r in px):
                    problems.append(f"{x} and {y} both write {key!r} and can both run; give one "
                                    "of them another key, or drop the duplicate")
    return problems[:6]


# --------------------------------------------------------------------------- plan helpers

def execution_order(plan: dict) -> list[str]:
    by = {a["name"]: a for a in plan["agents"]}
    order: list[str] = []

    def walk(n):
        if n in order or n not in by:
            return
        order.append(n)
        for c in by[n].get("sub_agents") or []:
            walk(str(c))

    walk(plan["root"])
    order += [n for n in by if n not in order]
    return order


def available_keys(plan: dict, name: str, inputs: list[str]) -> list[str]:
    """Keys written by agents that run before ``name`` (in execution order), plus the inputs."""
    by = {a["name"]: a for a in plan["agents"]}
    keys = list(dict.fromkeys(list(inputs) + list(plan.get("pipeline_inputs") or [])))
    for n in execution_order(plan):
        if n == name:
            break
        k = by[n].get("output_key")
        if k and k not in keys:
            keys.append(k)
    return keys


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{3,}")


def _find_sources(catalog, refs: list[str], limit_chars: int = 120000) -> tuple[str, list[str]]:
    """Catalog documents (tools, metadata) that define what an agent's plan entry names.

    ``refs`` are free text (plan ``sources``, the agent's name and role): every identifier
    in them that names a catalog document, or a ``tool_name``/``name`` entry inside one,
    pulls in that document and all its parts (e.g. a tool's overview and its definition)."""
    names: dict[str, list[str]] = {}
    for did, meta in catalog.index.items():
        if meta.get("kind") in ("tool", "metadata"):
            head = re.split(r"\s+[/(]", str(meta.get("name", "")))[0].strip().lower()
            names.setdefault(head, []).append(did)
    tokens = list(dict.fromkeys(t.lower().strip(".") for r in refs for t in _IDENT.findall(str(r))))
    found: list[str] = []
    for t in tokens:
        hits = names.get(t, [])
        if not hits:
            pat = re.compile(rf"(?m)^\s*-?\s*(tool_name|name|id):\s*['\"]?{re.escape(t)}['\"]?\s*$",
                             re.I)
            hits = [d for d, txt in catalog.documents.items()
                    if catalog.index.get(d, {}).get("kind") in ("tool", "metadata") and pat.search(txt)]
        found += [d for d in hits if d not in found]
    blocks = [f"[{catalog.index.get(d, {}).get('kind')}] {catalog.index.get(d, {}).get('name')}"
              f"\n{catalog.get(d) or ''}" for d in found]
    text = "\n\n".join(blocks)
    return (text[:limit_chars] if text else ""), found


def _checks_for(flow: dict, covers: list[str]) -> str:
    want = {str(c) for c in covers}
    out = []
    for a in flow.get("attributes", []):
        for c in a.get("checks", []):
            if f"{a['id']}:{c['id']}" in want:
                out.append({"attribute": a["id"], "question": a.get("question"), **c})
    return json.dumps(out, indent=1) if out else "(none)"


def _cited(procedure: str, lines: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", str(lines or ""))
    if not m:
        return "(none)"
    rows = procedure.splitlines()
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return "\n".join(f"{i}| {rows[i - 1]}" for i in range(max(1, a), min(len(rows), b) + 1)
                     if i - 1 < len(rows))[:40000] or "(none)"


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _strings(x)


_RUNTIME = re.compile(r"\$\{[^}]*\}")      # ${ENV:VAR}, ${UUID}: filled by the runtime, not state


def _sql_texts(fields) -> list[str]:
    """SQL queries in the fields: a ``query`` value, also inside embedded YAML strings."""
    out = []

    def visit(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "query" and isinstance(x, str):
                    out.append(x)
                else:
                    visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, str) and "\n" in v:
            try:
                parsed = yaml.safe_load(v)
            except yaml.YAMLError:
                return
            if isinstance(parsed, (dict, list)):
                visit(parsed)

    visit(fields)
    return out


def check_fields(agent: dict, fields: dict, class_syntax: str, available: list[str]) -> None:
    problems = []
    if class_syntax:
        unknown = [k for k in fields if k not in PLAN_FIELDS
                   and not re.search(rf"\b{re.escape(str(k))}\b", class_syntax)]
        if unknown:
            problems.append(f"fields {unknown} are not in the syntax for {agent['agent_class']}")
    clean = _RUNTIME.sub("", json.dumps(fields))   # runtime values are not session keys
    refs = double_brace_refs(json.loads(clean))
    refs |= {p for q in _sql_texts(fields) for p in sql_params(_RUNTIME.sub("", q))}
    instruction = fields.get("instruction")
    if isinstance(instruction, str):               # {key} placeholders live in instructions
        refs |= single_brace_refs(_RUNTIME.sub("", instruction))
    missing = refs - set(available) - {agent.get("output_key")} - set(agent.get("input_keys") or [])
    if missing:
        problems.append(f"references {sorted(missing)}, which are not available session keys "
                        f"(available: {available})")
    subs = set(agent.get("sub_agents") or [])
    for r in fields.get("routes") or [] if isinstance(fields.get("routes"), list) else []:
        t = r.get("target_agent") if isinstance(r, dict) else None
        if t is not None and subs and t not in subs:
            problems.append(f"route target_agent {t!r} is not one of its sub_agents {sorted(subs)}")
    if problems:
        raise ValueError("; ".join(problems))


def _strs(v) -> list[str]:
    v = v if isinstance(v, list) else [v] if v else []
    return [str(x) for x in v if str(x).strip()]


def _comment(label: str, text) -> list[str]:
    """A note as YAML comment lines: every line of a multi-line note gets its own '#'."""
    lines = [ln.rstrip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    return [f"# {label}: {lines[0]}"] + [f"#   {ln}" for ln in lines[1:]]


def _key_fields(agent: dict, fields: dict, class_syntax: str) -> tuple[dict, str]:
    """The plan's input keys in the form the class's syntax defines: ``input_keys`` (list)
    where the syntax has it, else ``input_key`` (one key); nothing if it has neither."""
    keys = [str(k) for k in agent.get("input_keys") or []]
    if not keys or "input_key" in fields or "input_keys" in fields:
        return {}, ""
    if not class_syntax or re.search(r"\binput_keys\b", class_syntax):
        return {"input_keys": keys}, ""
    if re.search(r"\binput_key\b", class_syntax):
        note = (f"{agent['agent_class']} takes one input_key; the agent also reads "
                f"{keys[1:]} through its fields") if len(keys) > 1 else ""
        return {"input_key": keys[0]}, note
    return {}, (f"{agent['agent_class']} has no input_key(s) in the syntax; it reads {keys} "
                "through its fields")


# --------------------------------------------------------------------------- rendering

def flow_md(flow: dict) -> str:
    out = []
    for a in flow.get("attributes", []):
        out += [f"# {a.get('id')} (q{a.get('number', '')}): {a.get('question', '')}",
                f"Lines {a.get('source_lines')}. Scope: {a.get('scope', '')}",
                f"Outcomes: {', '.join(a.get('outcomes') or [])}", ""]
        for c in a.get("checks", []):
            out.append(f"## {c.get('id')}. {c.get('title')} (lines {c.get('source_lines')})")
            for r in c.get("reads") or []:
                out.append(f"- reads: {r.get('what')} ({r.get('where', '')})")
            for x in c.get("external") or []:
                out.append(f"- needs: {x}")
            for r in c.get("rules") or []:
                out.append(f"- if {r.get('if')} → {r.get('then')}")
            out.append("")
    if flow.get("unclear"):
        out += ["# Unclear in the procedure", *[f"- {u}" for u in flow["unclear"]]]
    return "\n".join(out)


def datamap_md(dm: dict) -> str:
    out = ["| Value | Checks | Status | Source |", "|---|---|---|---|"]
    for f in dm.get("fields", []):
        src = (f"{f.get('tool')} · {f.get('field')}" if f.get("status") == "available"
               else f.get("derivation") or "—")
        out.append(f"| {f.get('need')} | {', '.join(f.get('checks') or [])} | {f.get('status')} | {src} |")
    for e in dm.get("external") or []:
        out.append(f"\n**{e.get('system')}** ({e.get('status')}): {', '.join(e.get('tools') or [])}. "
                   f"{e.get('notes', '')}")
    return "\n".join(out)


def plan_md(plan: dict) -> str:
    by = {a["name"]: a for a in plan["agents"]}
    out = [f"# {plan.get('name', 'workflow')}", f"Pipeline inputs: {plan.get('pipeline_inputs')}", ""]

    def walk(n, d, seen):
        if n in seen or n not in by:
            return
        seen.add(n)
        a = by[n]
        out.append(f"{'  ' * d}- **{n}** ({a.get('agent_class')}) → {a.get('output_key') or '—'}"
                   f"  {a.get('purpose', '')}")
        for c in a.get("sub_agents") or []:
            walk(str(c), d + 1, seen)

    seen: set[str] = set()
    walk(plan["root"], 0, seen)
    for n in by:
        walk(n, 0, seen)
    return "\n".join(out)


# --------------------------------------------------------------------------- stages

class Harness:
    def __init__(self, client: LLMClient, cfg: GenConfig, catalog, inputs: Inputs,
                 out_dir: Path, workers: int = 4):
        self.client, self.cfg, self.catalog, self.inp = client, cfg, catalog, inputs
        self.out_dir = Path(out_dir)
        self.workers = workers
        self.system = COMMON_SYSTEM.format(playbook=inputs.playbook or "(no playbook given)",
                                           syntax=inputs.syntax)
        self.classes = set(inputs.classes)
        self.pipeline_inputs = list(cfg.pipeline_inputs)
        efforts = dict(_DEFAULT_EFFORT)
        efforts.update(cfg.generation.get("stage_effort") or {})
        self.effort = efforts

    def _model(self, stage: str) -> str:
        return self.cfg.model("ground" if stage == "write" else "extract")

    def _ask(self, stage: str, prompt: str, validate, max_tokens: int = 32000):
        return complete_json(self.client, model=self._model(stage), system=self.system,
                             prompt=prompt, max_tokens=max_tokens, retries=3,
                             validate=validate, effort=self.effort.get(stage))

    def _save(self, suffix: str, data, md: str | None = None) -> None:
        base = self.out_dir.parent / f"{self.out_dir.name}_{suffix}"
        base.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        if md is not None:
            base.with_suffix(".md").write_text(md, encoding="utf-8")

    @staticmethod
    def _j(x) -> str:
        return json.dumps(x, indent=1, ensure_ascii=False)

    def recipe(self) -> dict:
        r = self._ask("recipe", RECIPE_TASK.format(classes=", ".join(sorted(self.classes))),
                      lambda d: check_recipe(d, self.classes))
        self._save("recipe", r)
        return r

    def logic(self, recipe: dict) -> dict:
        numbered = "\n".join(f"{i}| {ln}" for i, ln in enumerate(self.inp.procedure.splitlines(), 1))
        n = len(self.inp.procedure.splitlines())
        flow = self._ask("logic", LOGIC_TASK.format(
            recipe=self._j({k: recipe.get(k) for k in ("roles", "logic_rules", "checks")}),
            outline=self.inp.outline or "(none)", procedure=numbered),
            lambda d: check_flow(d, n), max_tokens=64000)
        self._save("flow", flow, flow_md(flow))
        return flow

    def datamap(self, recipe: dict, flow: dict) -> dict:
        src = self.inp.tools + "\n" + self.inp.metadata
        dm = self._ask("datamap", DATAMAP_TASK.format(
            recipe=self._j({k: recipe.get(k) for k in ("data_rules", "checks")}),
            flow=self._j(flow), tools=self.inp.tools, metadata=self.inp.metadata),
            lambda d: check_datamap(d, flow, src), max_tokens=64000)
        self._save("data_map", dm, datamap_md(dm))
        return dm

    def orchestrate(self, recipe: dict, flow: dict, dm: dict, previous: dict | None = None,
                    problems: str = "") -> dict:
        feedback = ""
        if previous is not None:
            feedback = (f"\nYour previous plan:\n{self._j(previous)}\n\nProblems the review found in "
                        f"the workflow built from it (fix them all, change nothing else):\n{problems}\n")
        plan = self._ask("orchestrate", ORCHESTRATE_TASK.format(
            prefix=self.cfg.prefix or "(from the procedure / tools)",
            classes=", ".join(sorted(self.classes)), inputs=self.pipeline_inputs or "(decide)",
            recipe=self._j(recipe), flow=self._j(flow), datamap=self._j(dm),
            tools=self.inp.tools, metadata=self.inp.metadata, feedback=feedback),
            lambda d: check_plan(d, self.classes, flow, self.pipeline_inputs, recipe),
            max_tokens=64000)
        if not self.pipeline_inputs:
            self.pipeline_inputs = list(plan.get("pipeline_inputs") or [])
        self._save("plan", plan, plan_md(plan))
        return plan

    def write_agent(self, plan: dict, flow: dict, dm: dict, agent: dict,
                    feedback: str = "") -> AgentResult:
        cls = agent["agent_class"]
        class_syntax = self.catalog.syntax_for(cls)
        available = available_keys(plan, agent["name"], self.pipeline_inputs)
        srcs, src_ids = _find_sources(self.catalog, list(agent.get("sources") or [])
                                      + [agent["name"], str(agent.get("role", ""))])
        if not src_ids:
            srcs = ("No catalog document matched this agent's sources, so here are the full "
                    f"inputs.\n\nTOOLS:\n{self.inp.tools}\n\nMETADATA:\n{self.inp.metadata}")
        summary = [{"name": a["name"], "agent_class": a["agent_class"],
                    "output_key": a.get("output_key", "")} for a in plan["agents"]]
        prompt = WRITE_TASK.format(
            agent=self._j(agent), plan=self._j(summary), available=available,
            agent_class=cls, class_syntax=class_syntax or "(see the agent syntax above)",
            checks=_checks_for(flow, agent.get("covers") or []), datamap=self._j(dm),
            procedure_lines=_cited(self.inp.procedure, agent.get("procedure_lines", "")),
            sources=srcs, plan_fields=", ".join(PLAN_FIELDS),
            feedback=f"\nProblems found in your previous version (fix them all):\n{feedback}\n"
            if feedback else "")

        def validate(d):
            if not isinstance(d, dict) or not isinstance(d.get("fields"), dict):
                raise ValueError('reply with a JSON object holding a "fields" object')
            check_fields(agent, d["fields"], class_syntax, available)

        res = AgentResult(agent["name"], sources=src_ids)
        try:
            d = self._ask("write", prompt, validate, max_tokens=24000)
        except ValueError as e:      # one hard agent never stops the whole workflow
            res.warnings.append(f"{agent['name']}: no valid answer ({e})")
            res.gaps.append(f"all fields: the writer could not produce a valid answer ({e})")
            return res
        res.fields = {k: v for k, v in d["fields"].items() if k not in PLAN_FIELDS}
        res.description = str(d.get("description") or agent.get("purpose") or "")
        res.gaps = _strs(d.get("gaps"))
        res.input_gaps = _strs(d.get("input_gaps"))
        res.plan_issues = _strs(d.get("plan_issues"))
        return res

    def emit(self, plan: dict, results: dict[str, AgentResult]) -> list[str]:
        """Write one YAML per planned agent. Returns the agents whose file did not parse."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob("*.yaml"):
            old.unlink()
        broken = []
        for a in plan["agents"]:
            r = results.get(a["name"]) or AgentResult(a["name"])
            d: dict = {"name": a["name"], "agent_class": a["agent_class"]}
            if r.description or a.get("purpose"):
                d["description"] = r.description or a.get("purpose")
            keys, key_note = _key_fields(a, r.fields, self.catalog.syntax_for(a["agent_class"]))
            d.update(keys)
            if a.get("output_key"):
                d["output_key"] = a["output_key"]
            d.update(r.fields)
            for k, v in (self.cfg.agent_templates.get(a["agent_class"]) or {}).items():
                d.setdefault(k, v)
            if a.get("sub_agents"):
                d["sub_agents"] = [{"name": s} for s in a["sub_agents"]]
            head = ["# Generated by o2a_gen (playbook mode). Review before use."]
            head += _comment("role", a.get("role", ""))
            head += _comment("playbook", a.get("playbook_rule", ""))
            head += _comment("checks", ", ".join(map(str, a.get("covers") or [])))
            head += _comment("procedure lines", a.get("procedure_lines", ""))
            head += _comment("sources", ", ".join(map(str, a.get("sources") or [])))
            head += _comment("note", key_note)
            for g in r.gaps:
                head += _comment("TODO", g)
            text = "\n".join(head) + "\n" + dump_yaml(d)
            try:
                yaml.safe_load(text)
            except yaml.YAMLError as e:
                broken.append(a["name"])
                r.warnings.append(f"{a['name']}: written YAML does not parse ({e})")
            (self.out_dir / f"{a['name']}.yaml").write_text(text, encoding="utf-8")
        return broken

    def code_findings(self, plan: dict):
        syntax = {a["agent_class"]: self.catalog.syntax_for(a["agent_class"]) for a in plan["agents"]}
        return validate_dir(self.out_dir, self.pipeline_inputs, syntax)

    def review(self, recipe: dict, flow: dict, dm: dict, findings) -> list[dict]:
        names = {p.stem for p in self.out_dir.glob("*.yaml")}
        yamls = "\n\n".join(f"### {p.name}\n{p.read_text(encoding='utf-8')}"
                            for p in sorted(self.out_dir.glob("*.yaml")))

        def validate(d):
            if not isinstance(d, dict) or not isinstance(d.get("findings"), list):
                raise ValueError('expected an object with a "findings" list')
            for f in d["findings"]:
                if not isinstance(f, dict) or f.get("severity") not in ("blocking", "minor"):
                    raise ValueError('each finding needs severity "blocking" or "minor"')
                if f.get("agent") not in names | {"*"}:
                    raise ValueError(f"finding names unknown agent {f.get('agent')!r}; "
                                     f"use one of the YAML names or '*'")

        d = self._ask("review", REVIEW_TASK.format(
            recipe=self._j(recipe), flow=self._j(flow), datamap=self._j(dm),
            code_findings="\n".join(str(f) for f in findings) or "(none)", yamls=yamls),
            validate, max_tokens=32000)
        return d["findings"]


def run_playbook(client: LLMClient, cfg: GenConfig, catalog, procedure_text: str,
                 out_dir: Path, *, outline: str = "", workers: int = 4) -> dict:
    """Run every stage; returns the report (also written as <out>_report.json by the caller)."""
    inp = load_inputs(catalog, procedure_text, outline)
    h = Harness(client, cfg, catalog, inp, out_dir, workers)

    print("[2/7] Reading the playbook into a build recipe ...", flush=True)
    recipe = h.recipe()
    print(f"      {len(recipe['roles'])} roles, {len(recipe.get('checks') or [])} checks")

    print("[3/7] Understanding the review logic in the procedure ...", flush=True)
    flow = h.logic(recipe)
    n_checks = sum(len(a["checks"]) for a in flow["attributes"])
    print(f"      {len(flow['attributes'])} attribute(s), {n_checks} checks")

    print("[4/7] Mapping the data the checks need to tools and metadata ...", flush=True)
    dm = h.datamap(recipe, flow)
    counts = {s: sum(f.get("status") == s for f in dm["fields"]) for s in OUTCOME_STATUSES}
    print(f"      {counts}")

    print("[5/7] Planning the agents from the playbook ...", flush=True)
    plan = h.orchestrate(recipe, flow, dm)
    print(f"      {len(plan['agents'])} agents, root {plan['root']}")

    print("[6/7] Writing each agent ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = {r.name: r for r in pool.map(
            lambda a: h.write_agent(plan, flow, dm, a), plan["agents"])}
    h.emit(plan, results)

    print("[7/7] Reviewing ...", flush=True)
    rounds = int(cfg.generation.get("review_rounds", 2))
    review: list[dict] = []
    findings = h.code_findings(plan)
    replans = 0
    for rnd in range(rounds + 1):
        try:
            review = h.review(recipe, flow, dm, findings)
        except ValueError as e:
            print(f"      review failed: {e}")
            review = []
        by = {a["name"]: a for a in plan["agents"]}
        plan_todo, agent_todo = _sort_findings(findings, review, results, by, plan["root"])
        blocking = sum(f["severity"] == "blocking" for f in review)
        print(f"      round {rnd + 1}: {blocking} blocking finding(s), "
              f"{sum(f.severity == 'ERROR' for f in findings)} code error(s), "
              f"{len(plan_todo)} plan problem(s)")
        if (not plan_todo and not agent_todo) or rnd == rounds:
            break
        rewrite = set(agent_todo)
        if plan_todo:
            try:
                new_plan = h.orchestrate(recipe, flow, dm, previous=plan,
                                         problems="\n".join(f"- {x}" for x in plan_todo))
                old = {a["name"]: a for a in plan["agents"]}
                rewrite |= {a["name"] for a in new_plan["agents"] if old.get(a["name"]) != a}
                plan, replans = new_plan, replans + 1
                results = {n: r for n, r in results.items()
                           if n in {a["name"] for a in plan["agents"]}}
                print(f"      re-planned: {len(plan['agents'])} agents, "
                      f"{len(rewrite)} to (re)write")
            except ValueError as e:
                print(f"      re-plan failed, keeping the plan: {e}")
        by = {a["name"]: a for a in plan["agents"]}
        names = [n for n in rewrite if n in by]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            fixed = list(pool.map(lambda n: h.write_agent(
                plan, flow, dm, by[n], feedback="\n".join(agent_todo.get(n, []))), names))
        for r in fixed:
            results[r.name] = r
        h.emit(plan, results)
        findings = h.code_findings(plan)

    by = {a["name"]: a for a in plan["agents"]}
    h._save("review", {"findings": review, "code_findings": [asdict(f) for f in findings],
                       "replans": replans})
    gaps = [{"agent": n, "gap": g} for n, r in results.items() for g in r.gaps]
    input_gaps = list(dict.fromkeys(g for r in results.values() for g in r.input_gaps))
    open_plan_issues = [f"{n}: {x}" for n, r in results.items() for x in r.plan_issues]
    (out_dir.parent / f"{out_dir.name}_placeholders.md").write_text(
        "# Gaps in the inputs\n\n"
        + ("\n".join(f"- {g}" for g in input_gaps) or "(none)")
        + "\n\n# Values left empty\n\n| Agent | Missing value |\n|---|---|\n"
        + "\n".join(f"| {g['agent']} | {' '.join(g['gap'].split())} |" for g in gaps),
        encoding="utf-8")
    covered = {str(c): a["name"] for a in plan["agents"] for c in a.get("covers") or []}
    return {
        "mode": "playbook",
        "pipeline": plan["root"],
        "pipeline_inputs": h.pipeline_inputs,
        "agents": len(plan["agents"]),
        "coverage": [{"check": f"{a['id']}:{c['id']}", "title": c.get("title"),
                      "lines": c.get("source_lines"), "agent": covered.get(f"{a['id']}:{c['id']}")}
                     for a in flow["attributes"] for c in a["checks"]],
        "grounding": [{"agent": n, "gaps": r.gaps, "warnings": r.warnings,
                       "catalog_docs": [{"id": d, **{k: catalog.index.get(d, {}).get(k)
                                                     for k in ("kind", "name")}}
                                        for d in r.sources],
                       "procedure_section": by[n].get("procedure_lines", "")}
                      for n, r in results.items()],
        "review": review,
        "findings": [asdict(f) for f in findings],
        "errors": sum(f.severity == "ERROR" for f in findings),
        "warnings": sum(f.severity == "WARN" for f in findings)
                    + sum(f["severity"] == "minor" for f in review),
        "blocking": sum(f["severity"] == "blocking" for f in review),
        "replans": replans,
        "input_gaps": input_gaps,
        "open_plan_issues": open_plan_issues,
    }


_PLAN_CODES = {"missing-agent", "root", "order", "never-written", "maybe-missing"}


def _sort_findings(findings, review: list[dict], results: dict, by: dict, root: str):
    """Split problems into plan-level ones (fixed by re-planning) and agent-level ones
    (fixed by that agent's writer)."""
    plan_todo: list[str] = []
    agent_todo: dict[str, list[str]] = {}
    for f in findings:
        if f.severity != "ERROR":
            continue
        if f.code in _PLAN_CODES:
            plan_todo.append(f"{f.agent}: {f.code}: {f.message}")
            continue
        for n in str(f.agent).split(","):
            if n in by:
                agent_todo.setdefault(n, []).append(f"{f.code}: {f.message}")
    for f in review:
        if f["severity"] != "blocking":
            continue
        text = f"{f['problem']} Fix: {f.get('fix', '')}"
        if f.get("level") == "plan" or f["agent"] == "*" or f["agent"] not in by:
            plan_todo.append(f"{f['agent']}: {text}")
        else:
            agent_todo.setdefault(f["agent"], []).append(text)
    for n, r in results.items():
        plan_todo += [f"{n} (writer): {x}" for x in r.plan_issues]
    return list(dict.fromkeys(plan_todo)), agent_todo
