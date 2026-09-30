"""Procedure document -> structured, validated steps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from o2a_gen.llm import LLMClient, complete_json

@dataclass
class Branch:
    label: str
    when: str
    steps: list["Step"]


@dataclass
class Step:
    id: str
    title: str
    text: str
    agent_class: str                       # one of the classes the agent syntax defines
    source_lines: str = ""
    produces: str = ""
    uses: list[str] = field(default_factory=list)
    depends_on: str = ""                   # branching steps only: step whose output is tested
    branches: list[Branch] = field(default_factory=list)


@dataclass
class Phase:
    id: str
    title: str
    steps: list[Step]
    direct: bool = False                   # run by the orchestrator itself, not in a group


@dataclass
class Procedure:
    name: str
    description: str
    inputs: list[str]
    phases: list[Phase]
    source: str = ""
    orchestrator_class: str = ""           # top-level agent_class, chosen from the agent syntax
    group_class: str = ""                  # agent_class that runs a phase's steps in order

    def all_steps(self) -> list[Step]:
        out: list[Step] = []

        def walk(steps: list[Step]):
            for s in steps:
                out.append(s)
                for b in s.branches:
                    walk(b.steps)

        for ph in self.phases:
            walk(ph.steps)
        return out


def load_procedure_text(path: Path) -> str:
    """Read .md/.txt directly; .docx and .pdf need python-docx / pypdf."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            raise ImportError("pip install python-docx to read .docx procedures") from None
        lines = []
        for para in docx.Document(str(path)).paragraphs:
            style = (para.style.name or "").lower() if para.style is not None else ""
            m = re.match(r"heading (\d)", style)
            text = para.text.strip()
            if text:
                lines.append(("#" * int(m.group(1)) + " " if m else "") + text)
        return "\n".join(lines)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pip install pypdf to read .pdf procedures") from None
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


EXTRACT_SYSTEM = """You convert an operations review procedure into a workflow of agents for the
O2A runtime. The agent syntax document you are given is the only source of which agent classes
exist and what each one does. You do not invent steps: every step must come from the procedure
text, and you cite the line numbers it came from. Keep the procedure's order."""

EXTRACT_PROMPT = """Agent syntax (the complete list of agent classes and what each is for):

{syntax}

Procedure (each line is prefixed with its line number):

{numbered}

Return one JSON object:
{{
  "name": "snake_case name for the whole procedure",
  "description": "one or two sentences",
  "inputs": ["data the review starts with, as snake_case session keys"],
  "orchestrator_class": "the agent_class that runs the whole workflow",
  "group_class": "the agent_class that runs a phase's steps one after another",
  "phases": [
    {{"id": "P1", "title": "short phase title", "direct": false, "steps": [STEP, ...]}}
  ]
}}

STEP = {{
  "id": "S1",                          (unique across the whole document: S1, S2, ...)
  "title": "short imperative title",
  "text": "one short sentence on what this step does (its full wording is read later
           from source_lines, so do not copy it here)",
  "source_lines": "12-15",
  "agent_class": one of {classes},
  "produces": "short noun phrase for the data this step creates",
  "uses": ["ids of earlier steps whose output this step needs, or names from inputs"],
  "depends_on": "branching steps only: id of the earlier step whose output is tested",
  "branches": [                        (branching steps only, at least two)
    {{"label": "short branch name", "when": "the condition in words", "steps": [STEP, ...]}}
  ]
}}

Rules:
- Choose each step's agent_class from what the agent syntax says each class is for, and from
  what the tools and data the step needs are (a query, an API call, a deterministic
  transformation, judgement by an LLM, waiting for an external event, ...). Prefer
  deterministic classes; use an LLM class only where judgement or writing is needed.
- Where the procedure branches ("if ... then ... otherwise ..."), make one step with `branches`,
  whose agent_class is the class the syntax gives for routing. It must test ONE simple value:
  if the condition is complex, add a step just before it that produces that value (e.g. a Y/N
  flag) and set depends_on to it.
- Every branch has at least one step. If a branch just continues or ends the review, add one
  step that records that outcome.
- Steps inside a branch only run on that branch. When each branch ends by producing the same
  kind of outcome (e.g. "review outcome"), use the identical `produces` text in each branch
  so later steps can read that outcome whichever branch ran.
- Set "direct": true on a phase with a single step that the orchestrator should run itself
  rather than inside a group (the agent syntax says which classes belong directly under the
  orchestrator)."""


def extract_procedure(client: LLMClient, text: str, *, model: str, classes: list[str],
                      syntax: str, source: str = "", outline: str = "") -> Procedure:
    """``classes``: the agent classes the syntax document defines; ``syntax``: that document's
    text. ``outline``: the compiled procedure tree's section summaries, given as a map."""
    if not classes:
        raise ValueError("no agent classes found: the agent syntax document is required")
    numbered = "\n".join(f"{i}| {line}" for i, line in enumerate(text.splitlines(), 1))
    prompt = EXTRACT_PROMPT.format(numbered=numbered, syntax=syntax,
                                   classes=" | ".join(sorted(classes)))
    if outline:
        prompt = f"Section outline (summaries of the procedure's sections):\n{outline}\n\n{prompt}"
    n_lines = len(text.splitlines())

    def validate(data):
        _validate_raw(data, set(classes))
        _check_lines(_parse(data), n_lines)

    data = complete_json(client, model=model, system=EXTRACT_SYSTEM, prompt=prompt,
                         max_tokens=64000, validate=validate)
    proc = _parse(data)
    proc.source = source
    return proc


def _validate_raw(data, classes: set[str]) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list) or not data["phases"]:
        raise ValueError("expected an object with a non-empty 'phases' list")
    proc = _parse(data)
    ids: set[str] = set()
    known = set(proc.inputs)
    problems: list[str] = []
    for key in ("orchestrator_class", "group_class"):
        if getattr(proc, key) not in classes:
            problems.append(f"{key} {getattr(proc, key)!r} is not an agent class in the syntax")

    def walk(steps: list[Step], visible: set[str]):
        visible = set(visible)
        for s in steps:
            if s.id in ids:
                problems.append(f"duplicate step id {s.id}")
            ids.add(s.id)
            if s.agent_class not in classes:
                problems.append(f"{s.id}: agent_class {s.agent_class!r} is not in the agent "
                                f"syntax; use one of {sorted(classes)}")
            for u in s.uses:
                if u not in visible and u not in known:
                    problems.append(f"{s.id}: uses {u!r}, which is not an earlier step or input")
            if s.branches:
                if len(s.branches) < 2:
                    problems.append(f"{s.id}: a decision needs at least two branches")
                if s.depends_on not in visible:
                    problems.append(f"{s.id}: depends_on {s.depends_on!r} must be an earlier step")
                # Later steps may use a branch's output (the validator warns if
                # not every branch produces what they need).
                before = visible | {s.id}
                for b in s.branches:
                    if not b.steps:
                        problems.append(f"{s.id}: branch {b.label!r} has no steps")
                    visible |= walk(b.steps, before)
            visible.add(s.id)
        return visible

    visible: set[str] = set()
    for ph in proc.phases:
        visible = walk(ph.steps, visible)
    if problems:
        raise ValueError("; ".join(problems[:12]))


def _check_lines(proc: Procedure, n_lines: int) -> None:
    """Every step must cite lines that exist, as "a" or "a-b"."""
    bad = []
    for s in proc.all_steps():
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", s.source_lines or "")
        if not m or not (1 <= int(m.group(1)) <= int(m.group(2) or m.group(1)) <= n_lines):
            bad.append(f"{s.id}: source_lines {s.source_lines!r}")
    if bad:
        raise ValueError(f"source_lines must be line numbers between 1 and {n_lines}: "
                         + "; ".join(bad[:8]))


def _parse(data: dict) -> Procedure:
    def step(d: dict) -> Step:
        return Step(
            id=str(d.get("id", "")).strip(),
            title=str(d.get("title", "")).strip(),
            text=str(d.get("text", "")).strip(),
            agent_class=str(d.get("agent_class", "")).strip(),
            source_lines=str(d.get("source_lines", "")),
            produces=str(d.get("produces", "")).strip(),
            uses=[str(u) for u in d.get("uses") or []],
            depends_on=str(d.get("depends_on") or ""),
            branches=[Branch(str(b.get("label", "")), str(b.get("when", "")),
                             [step(x) for x in b.get("steps") or []])
                      for b in d.get("branches") or []],
        )

    return Procedure(
        name=str(data.get("name", "procedure")),
        description=str(data.get("description", "")),
        inputs=[str(i) for i in data.get("inputs") or []],
        phases=[Phase(str(p.get("id", f"P{i}")), str(p.get("title", f"phase {i}")),
                      [step(s) for s in p.get("steps") or []], bool(p.get("direct", False)))
                for i, p in enumerate(data["phases"], 1)],
        orchestrator_class=str(data.get("orchestrator_class", "")).strip(),
        group_class=str(data.get("group_class", "")).strip(),
    )
