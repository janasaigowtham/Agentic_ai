"""Static checks on a directory of O2A agent YAMLs (generated or hand-written).

A lightweight subset of the evaluator's static checks, so generated output can
be verified without running the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from o2a_gen.refs import double_brace_refs, is_env_ref, single_brace_refs, sql_params

# Fallback only: when a catalog is given, the classes come from its agent syntax document.
KNOWN_CLASSES = {"LlmAgent", "database_agent", "rest_api_agent", "slv_transformation_agent",
                 "transformation_agent", "decision_router_agent", "SequentialAgent",
                 "ParallelAgent", "LoopAgent", "FailFastLoopAgent", "agent_gate",
                 "resumable_orchestrator"}
CONTAINERS = {"SequentialAgent", "resumable_orchestrator"}


@dataclass
class Finding:
    severity: str   # ERROR | WARN
    code: str
    agent: str
    message: str

    def __str__(self):
        return f"{self.severity:5} {self.code:14} {self.agent}: {self.message}"


def load_agents(folder: Path) -> tuple[dict[str, dict], list[Finding]]:
    agents: dict[str, dict] = {}
    findings: list[Finding] = []
    for p in sorted(Path(folder).glob("*.y*ml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            findings.append(Finding("ERROR", "yaml", p.stem, f"cannot parse: {e}"))
            continue
        if not isinstance(data, dict):
            findings.append(Finding("ERROR", "yaml", p.stem, "not a mapping"))
            continue
        name = str(data.get("name") or "")
        if name != p.stem:
            findings.append(Finding("ERROR", "name-file", p.stem,
                                    f"name {name!r} does not match file name"))
        agents[name or p.stem] = data
    return agents, findings


def input_keys(a: dict) -> list[str]:
    keys = list(a.get("input_keys") or [])
    if a.get("input_key"):
        keys.append(a["input_key"])
    return keys


def db_block(a: dict) -> dict:
    raw = a.get("default_db_yaml")
    if isinstance(raw, str):
        try:
            raw = yaml.safe_load(raw)
        except yaml.YAMLError:
            return {}
    return raw if isinstance(raw, dict) else {}


def children(a: dict) -> list[str]:
    return [str(s.get("name") if isinstance(s, dict) else s) for s in a.get("sub_agents") or []]


ALWAYS_ALLOWED = {"name", "agent_class", "description", "version", "sub_agents"}


def validate_dir(folder: Path, pipeline_inputs: list[str] | None = None,
                 syntax: dict[str, str] | None = None) -> list[Finding]:
    """``syntax`` maps agent_class -> its agent-syntax text; when given, fields the
    syntax never mentions are flagged."""
    agents, findings = load_agents(folder)
    inputs = set(pipeline_inputs or [])
    F = findings.append
    for name, a in agents.items():
        text = (syntax or {}).get(a.get("agent_class"), "")
        if not text:
            continue
        for fld in a:
            if fld not in ALWAYS_ALLOWED and not re.search(rf"\b{re.escape(str(fld))}\b", text):
                F(Finding("WARN", "unknown-field", name,
                          f"field {fld!r} is not in the agent syntax for {a.get('agent_class')}"))

    referenced = {c for a in agents.values() for c in children(a)}
    referenced |= {r.get("target_agent") for a in agents.values() for r in a.get("routes") or []}
    for name, a in agents.items():
        cls = a.get("agent_class")
        if cls not in KNOWN_CLASSES:
            F(Finding("WARN", "class", name, f"unknown agent_class {cls!r}"))
        for c in children(a):
            if c not in agents:
                F(Finding("ERROR", "missing-agent", name, f"sub_agent {c!r} has no YAML"))
        if cls == "decision_router_agent":
            _check_router(name, a, F)
        if cls == "database_agent":
            _check_db(name, a, F)
        if cls == "LlmAgent":
            refs = single_brace_refs(str(a.get("instruction") or ""))
            undeclared = refs - set(input_keys(a))
            if undeclared:
                F(Finding("ERROR", "undeclared-ref", name,
                          f"instruction uses {sorted(undeclared)} but input_keys does not declare them"))
            if not a.get("model"):
                F(Finding("WARN", "model", name, "LlmAgent has no model"))
        if cls in ("slv_transformation_agent", "transformation_agent"):
            undeclared = double_brace_refs(a.get("transform") or {}) - set(input_keys(a))
            if undeclared:
                F(Finding("ERROR", "undeclared-ref", name,
                          f"transform uses {sorted(undeclared)} but input_keys does not declare them"))

    roots = [n for n in agents if n not in referenced]
    if not roots:
        F(Finding("ERROR", "root", "-", "no root agent (every agent is referenced by another)"))
    elif len(roots) > 1:
        F(Finding("WARN", "root", "-", f"several unreferenced agents: {sorted(roots)}"))

    writers: dict[str, list[str]] = {}
    for n, a in agents.items():
        if a.get("output_key") and a.get("agent_class") != "decision_router_agent":
            writers.setdefault(a["output_key"], []).append(n)
    branch_of = _branch_paths(agents)
    for key, ws in writers.items():
        if len(ws) > 1 and not _mutually_exclusive(ws, branch_of):
            F(Finding("WARN", "multi-writer", ",".join(sorted(ws)),
                      f"{len(ws)} agents write {key!r}"))

    produced_anywhere = set(writers) | inputs
    for root in roots:
        _flow(root, agents, set(inputs), set(), produced_anywhere, F, set())
    return findings


def _branch_paths(agents: dict[str, dict]) -> dict[str, dict[str, str]]:
    """agent -> {router name: the branch (router child) it sits under}."""
    out: dict[str, dict[str, str]] = {}

    def walk(name, path, seen):
        if name in seen or name not in agents:
            return
        out.setdefault(name, dict(path))
        a = agents[name]
        for c in children(a):
            sub = {**path, name: c} if a.get("agent_class") == "decision_router_agent" else path
            walk(c, sub, seen | {name})

    referenced = {c for a in agents.values() for c in children(a)}
    for root in (n for n in agents if n not in referenced):
        walk(root, {}, set())
    return out


def _mutually_exclusive(names: list[str], branch_of: dict[str, dict[str, str]]) -> bool:
    """True if every pair sits in different branches of some shared router."""
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = branch_of.get(a, {}), branch_of.get(b, {})
            if not any(r in pb and pa[r] != pb[r] for r in pa):
                return False
    return True


def _flow(name, agents, definite, maybe, anywhere, F, seen) -> tuple[set, set]:
    """Walk execution order; check each agent's inputs exist when it runs."""
    a = agents.get(name)
    if a is None or name in seen:
        return definite, maybe
    seen = seen | {name}
    needed = input_keys(a) + [c.get("context_key") for r in a.get("routes") or []
                              for c in r.get("conditions") or [] if c.get("context_key")]
    for k in dict.fromkeys(needed):
        if k in definite:
            continue
        if k in maybe:
            F(Finding("WARN", "maybe-missing", name,
                      f"reads {k!r}, which only some router branches produce"))
        elif k in anywhere:
            F(Finding("ERROR", "order", name, f"reads {k!r} before any agent writes it"))
        else:
            F(Finding("ERROR", "never-written", name,
                      f"reads {k!r}, which no agent writes and is not a pipeline input"))
    cls = a.get("agent_class")
    if cls in CONTAINERS:
        for c in children(a):
            definite, maybe = _flow(c, agents, definite, maybe, anywhere, F, seen)
    elif cls == "decision_router_agent":
        outs = [_flow(c, agents, set(definite), set(maybe), anywhere, F, seen) for c in children(a)]
        if outs:
            common = set.intersection(*(d for d, _ in outs))
            union = set.union(*(d | m for d, m in outs))
            definite, maybe = common, (maybe | union) - common
    if a.get("output_key") and cls not in ("decision_router_agent",):
        definite = definite | {a["output_key"]}
    return definite, maybe


def _check_router(name, a, F):
    routes = a.get("routes") or []
    subs = set(children(a))
    if not routes:
        F(Finding("ERROR", "router", name, "decision_router_agent has no routes"))
    prios = [r.get("priority") for r in routes]
    if len(prios) != len(set(prios)):
        F(Finding("WARN", "router", name, "two routes share a priority"))
    for r in routes:
        if r.get("target_agent") not in subs:
            F(Finding("ERROR", "router", name,
                      f"route target {r.get('target_agent')!r} is not in sub_agents"))
    if routes and all(r.get("conditions") for r in routes):
        F(Finding("WARN", "router-default", name,
                  "no default route: if no condition matches, nothing runs"))
    undeclared = {c.get("context_key") for r in routes for c in r.get("conditions") or []} \
        - set(input_keys(a))
    if undeclared and input_keys(a):
        F(Finding("WARN", "undeclared-ref", name,
                  f"routes test {sorted(undeclared)} not listed in input_keys"))


def _check_db(name, a, F):
    db = db_block(a)
    if not db:
        F(Finding("ERROR", "db", name, "default_db_yaml missing or unparseable"))
        return
    url = (db.get("connection") or {}).get("url")
    if not is_env_ref(url):
        F(Finding("ERROR", "db-credential", name,
                  "connection url must be ${ENV:VAR}, not a literal"))
    params = sql_params(str(db.get("query") or ""))
    undeclared = params - set(input_keys(a))
    if undeclared:
        F(Finding("ERROR", "undeclared-ref", name,
                  f"query binds {sorted(undeclared)} but input_keys does not declare them"))
