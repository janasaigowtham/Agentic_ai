"""Fills each planned agent's fields using the catalog: SQL, transforms, prompts, routes.

For every step, the navigator browses the compiled catalog for the metadata,
tools and reference documents it needs. A second call then writes the agent's
fields from what was found, following the agent-syntax section for the agent's
class, which is always included. Each answer is checked
(session keys exist, SQL uses bind parameters, credentials come from ENV, ...)
and sent back to the model with the error if it fails.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from o2a_gen.config import GenConfig
from o2a_gen.llm import LLMClient, complete_json
from o2a_gen.navigator import Catalog, NavResult, navigate
from o2a_gen.plan import AgentNode, Plan
from o2a_gen.refs import double_brace_refs, single_brace_refs, sql_params

ROUTER_OPERATORS = {"eq", "neq", "not_null", "is_null", "gt", "lt", "gte", "lte", "in", "not_in"}
RESERVED_FIELDS = {"name", "agent_class", "sub_agents", "routes", "output_key", "input_keys",
                   "input_key", "instruction", "transform", "default_db_yaml", "model",
                   "description", "tools"}

CONVENTIONS = """General O2A rules (the agent syntax given with each request is authoritative
where it says more):
- Agents share state through a session dict. An agent reads `input_keys` and writes `output_key`.
- Transforms and SQL reference session values as {{ key }} or {{ key[0].field }}.
- LlmAgent instructions reference session values as {key} or {key[0].field}.
- SQL binds values as :param, where param is a session key. Never paste literal values or
  credentials into SQL. Connection URLs are always ${ENV:VAR_NAME}."""

GROUND_SYSTEM = ("You write the fields of one agent in an O2A pipeline. Follow the O2A agent "
                 "syntax provided for its agent_class exactly. Use real table, column, tool and "
                 "connection names from the catalog documents provided; never invent them. If "
                 "something you need is missing, write your best attempt and explain the gap in "
                 "`gaps`.\n\n" + CONVENTIONS)


@dataclass
class Grounding:
    node: str
    step_id: str
    nav: NavResult | None
    gaps: str = ""
    warnings: list[str] = field(default_factory=list)


def ground_plan(client: LLMClient, plan: Plan, cfg: GenConfig, catalog: Catalog | None,
                max_workers: int = 4) -> list[Grounding]:
    work = [n for n in plan.nodes() if n.step is not None]
    tools = _tool_names(catalog)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return list(pool.map(lambda n: ground_node(client, n, cfg, catalog, tools), work))


def ground_node(client: LLMClient, node: AgentNode, cfg: GenConfig,
                catalog: Catalog | None, tools: set[str] | None = None) -> Grounding:
    step = node.step
    assert step is not None
    nav = None
    if catalog is not None and node.agent_class != "decision_router_agent":
        nav = navigate(client, catalog, _nav_question(node), model=cfg.model("navigate"),
                       max_turns=cfg.navigate_max_turns)
    docs_block = _docs_block(nav)
    spec, check = _SPECS[node.agent_class]
    syntax = catalog.syntax_for(node.agent_class) if catalog is not None else ""
    prompt = (
        f"O2A agent syntax for {node.agent_class}:\n"
        f"{syntax or '(not provided: follow the general rules)'}\n\n"
        f"Agent to write: `{node.name}` ({node.agent_class})\n"
        f"Procedure step {step.id} (lines {step.source_lines}): {step.title}\n{step.text}\n\n"
        f"output_key: {node.output_key}\n"
        f"Session keys available when it runs: {node.available_keys}\n"
        f"Keys the procedure says it needs: {node.required_keys}\n"
        + _router_context(node)
        + f"\nCatalog documents found:\n{docs_block}\n\n{spec}"
    )
    ctx = {"node": node, "cfg": cfg, "tools": tools or set()}

    def validate(d):
        if not isinstance(d, dict):
            raise ValueError("reply with a JSON object")
        check(d, ctx)

    data = complete_json(client, model=cfg.model("ground"), system=GROUND_SYSTEM,
                         prompt=prompt, max_tokens=4000, validate=validate)
    warnings = _apply(node, data, cfg, tools or set())
    if not syntax:
        warnings.append(f"{node.name}: no agent syntax found for {node.agent_class}")
    return Grounding(node.name, step.id, nav, str(data.get("gaps", "") or ""), warnings)


# --------------------------------------------------------------------------
# per-class output specs and checks
# --------------------------------------------------------------------------

_COMMON = ('"description": "one sentence", "gaps": "what the catalog lacked, or empty", '
           '"extra": {other fields the agent syntax requires for this class, with values}')

LOOKUP_SPEC = ('Return JSON: {"db_type": "teradata|postgres|...", "connection_env": "ENV_VAR_NAME", '
               '"query": "SQL using :params", "params": ["session keys bound in the query"], '
               + _COMMON + "}")
COMPUTE_SPEC = ('Return JSON: {"transform": {"<output_key>": <expression using $cond/$fn/$each '
                'and {{ key }} references>}, "input_keys": [...], "strict": true|false, '
                + _COMMON + "}")
REVIEW_SPEC = ('Return JSON: {"instruction": "the full LlmAgent prompt, referencing session '
               'values as {key}; say exactly what to check, what evidence to cite, and the output '
               'format", "input_keys": [...], "tools": ["tool names from the catalog, if the '
               'agent must call tools"], ' + _COMMON + "}")
DECISION_SPEC = ('Return JSON: {"routes": [{"branch": "<branch label>", "conditions": '
                 '[{"context_key": "<session key>", "operator": "eq|neq|not_null|is_null|gt|lt|gte|lte|in|not_in", '
                 '"value": <value>}], "priority": 10}], "description": "one sentence", "gaps": ""} '
                 "with exactly one route per branch; lower priority number wins.")
APPROVAL_SPEC = 'Return JSON: {' + _COMMON + '}'


def _check_lookup(d, ctx):
    node = ctx["node"]
    query = str(d.get("query") or "").strip()
    if not query:
        raise ValueError("query is empty")
    params = set(d.get("params") or [])
    used = sql_params(query) | double_brace_refs(query)
    missing = used - set(node.available_keys)
    if missing:
        raise ValueError(f"query references {sorted(missing)}, which are not available session "
                         f"keys; available: {node.available_keys}")
    if used - params:
        raise ValueError(f"params must list every bound key; missing {sorted(used - params)}")
    env = str(d.get("connection_env") or ctx["cfg"].default_connection_env or "")
    if not env or not env.replace("_", "").isalnum():
        raise ValueError("connection_env must be an environment variable name")


def _check_compute(d, ctx):
    node = ctx["node"]
    t = d.get("transform")
    if not isinstance(t, dict) or node.output_key not in t:
        raise ValueError(f"transform must be an object whose key is {node.output_key!r}")
    missing = double_brace_refs(t) - set(node.available_keys)
    if missing:
        raise ValueError(f"transform references {sorted(missing)}, not available session keys; "
                         f"available: {node.available_keys}")


def _check_review(d, ctx):
    node = ctx["node"]
    instr = str(d.get("instruction") or "")
    if len(instr) < 40:
        raise ValueError("instruction is missing or too short")
    missing = single_brace_refs(instr) - set(node.available_keys)
    if missing:
        raise ValueError(f"instruction references {sorted(missing)}, not available session keys; "
                         f"available: {node.available_keys}. Use only available keys.")
    unknown = set(d.get("tools") or []) - ctx["tools"]
    if unknown:
        raise ValueError(f"tools {sorted(unknown)} are not in the catalog "
                         f"(known: {sorted(ctx['tools']) or 'none'})")


def _check_decision(d, ctx):
    node = ctx["node"]
    routes = d.get("routes")
    labels = [b for b, _ in node.branch_targets]
    if not isinstance(routes, list):
        raise ValueError("routes must be a list")
    got = [r.get("branch") for r in routes]
    if sorted(got) != sorted(labels):
        raise ValueError(f"need exactly one route per branch {labels}; got {got}")
    for r in routes:
        if not isinstance(r.get("priority", 100), int):
            raise ValueError(f"priority must be an integer, got {r.get('priority')!r}")
        for c in r.get("conditions") or []:
            if c.get("context_key") not in node.available_keys:
                raise ValueError(f"context_key {c.get('context_key')!r} is not an available key; "
                                 f"available: {node.available_keys}")
            if c.get("operator") not in ROUTER_OPERATORS:
                raise ValueError(f"operator {c.get('operator')!r} not in {sorted(ROUTER_OPERATORS)}")


def _check_approval(d, ctx):
    if not isinstance(d, dict):
        raise ValueError("expected an object")


_SPECS = {
    "database_agent": (LOOKUP_SPEC, _check_lookup),
    "slv_transformation_agent": (COMPUTE_SPEC, _check_compute),
    "transformation_agent": (COMPUTE_SPEC, _check_compute),
    "LlmAgent": (REVIEW_SPEC, _check_review),
    "decision_router_agent": (DECISION_SPEC, _check_decision),
    "agent_gate": (APPROVAL_SPEC, _check_approval),
}


def _apply(node: AgentNode, d: dict, cfg: GenConfig, tools: set[str]) -> list[str]:
    """Write validated model output into node.fields; return warnings."""
    warnings: list[str] = []
    f: dict = {}
    if d.get("description"):
        node.description = str(d["description"]).strip()
    cls = node.agent_class
    if cls == "database_agent":
        query = str(d["query"]).strip()
        env = str(d.get("connection_env") or cfg.default_connection_env)
        f["input_keys"] = sorted(sql_params(query) | double_brace_refs(query))
        f["default_db_yaml"] = {
            "type": str(d.get("db_type") or cfg.default_db_type),
            "connection": {"url": f"${{ENV:{env}}}"},
            "query": query,
        }
        if not d.get("connection_env"):
            warnings.append(f"{node.name}: no connection found in catalog; used default {env}")
    elif cls in ("slv_transformation_agent", "transformation_agent"):
        refs = double_brace_refs(d["transform"])
        f["input_keys"] = sorted(refs | (set(d.get("input_keys") or []) & set(node.available_keys)))
        if "strict" in d:
            f["strict"] = bool(d["strict"])
        f["transform"] = d["transform"]
    elif cls == "LlmAgent":
        instr = str(d["instruction"]).strip()
        refs = single_brace_refs(instr)
        f["input_keys"] = sorted(refs | (set(d.get("input_keys") or []) & set(node.available_keys)))
        f["model"] = cfg.model("llm_agent")
        f["instruction"] = instr
        if d.get("tools"):
            f["tools"] = [t for t in d["tools"] if t in tools]
    elif cls == "decision_router_agent":
        targets = dict(node.branch_targets)
        routes = []
        keys: set[str] = set()
        for r in sorted(d["routes"], key=lambda r: int(r.get("priority", 100))):
            conds = [{"context_key": c["context_key"], "operator": c["operator"],
                      **({"value": c["value"]} if "value" in c else {})}
                     for c in r.get("conditions") or []]
            keys |= {c["context_key"] for c in conds}
            routes.append({"conditions": conds, "target_agent": targets[r["branch"]].name,
                           "priority": int(r.get("priority", 100))})
        f["input_keys"] = sorted(keys)
        f["routes"] = routes
    extra = d.get("extra") if isinstance(d.get("extra"), dict) else {}
    for k, v in extra.items():
        if k in RESERVED_FIELDS:
            continue
        f.setdefault(k, v)
    missing_required = set(node.required_keys) - set(f.get("input_keys", []))
    if missing_required and cls != "agent_gate":
        warnings.append(f"{node.name}: procedure says it needs {sorted(missing_required)} "
                        "but the generated agent does not read them")
    node.fields = f
    return warnings


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _nav_question(node: AgentNode) -> str:
    step = node.step
    what = {
        "database_agent": "the table(s) and columns holding this data and the connection "
                          "(env var) used to reach them",
        "slv_transformation_agent": "the fields of the input data and any transform functions "
                                    "the agent syntax documents",
        "LlmAgent": "policies or guidelines this review must apply and the tools it may need",
        "agent_gate": "how approvals or external events are defined for gates",
    }.get(node.agent_class, "related material")
    return (f"Procedure step: {step.title}\n{step.text}\n"
            f"It will be an O2A {node.agent_class}. Find {what}.")


def _docs_block(nav: NavResult | None) -> str:
    if nav is None:
        return "(no catalog; rely on the conventions)"
    if not nav.docs:
        return f"(nothing relevant found: {nav.notes})"
    parts = [f"--- doc {did} ---\n{text[:5000]}" for did, text in nav.docs.items()]
    return "\n\n".join(parts) + (f"\n\nNavigator notes: {nav.notes}" if nav.notes else "")


def _router_context(node: AgentNode) -> str:
    if node.agent_class != "decision_router_agent" or node.step is None:
        return ""
    lines = ["Branches (condition in words -> target agent):"]
    for (label, target), b in zip(node.branch_targets, node.step.branches):
        lines.append(f"- {label}: when {b.when} -> {target.name}")
    return "\n".join(lines) + "\n"


def _tool_names(catalog: Catalog | None) -> set[str]:
    if catalog is None:
        return set()
    return {m["name"] for m in catalog.index.values() if m.get("kind") == "tool"}
