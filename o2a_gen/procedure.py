"""Procedure document -> structured, validated steps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from o2a_gen.llm import LLMClient, complete_json

STEP_KINDS = ("lookup", "compute", "review", "decision", "approval")


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
    kind: str
    source_lines: str = ""
    produces: str = ""
    uses: list[str] = field(default_factory=list)
    depends_on: str = ""                   # decision only: step whose output is tested
    branches: list[Branch] = field(default_factory=list)


@dataclass
class Phase:
    id: str
    title: str
    steps: list[Step]


@dataclass
class Procedure:
    name: str
    description: str
    inputs: list[str]
    phases: list[Phase]
    source: str = ""

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


EXTRACT_SYSTEM = """You convert an operations review procedure into a structured workflow for an
agent pipeline. You do not invent steps: every step must come from the procedure text, and you
cite the line numbers it came from. Keep the procedure's order."""

EXTRACT_PROMPT = """Procedure (each line is prefixed with its line number):

{numbered}

Return one JSON object:
{{
  "name": "snake_case name for the whole procedure",
  "description": "one or two sentences",
  "inputs": ["data the review starts with, as snake_case session keys, e.g. loan_number"],
  "phases": [
    {{"id": "P1", "title": "short phase title", "steps": [STEP, ...]}}
  ]
}}

STEP = {{
  "id": "S1",                          (unique across the whole document: S1, S2, ...)
  "title": "short imperative title",
  "text": "what the procedure says to do, close to its wording",
  "source_lines": "12-15",
  "kind": one of {kinds},
  "produces": "short noun phrase for the data this step creates",
  "uses": ["ids of earlier steps whose output this step needs, or names from inputs"],
  "depends_on": "decision only: id of the earlier step whose output the decision tests",
  "branches": [                        (decision only, at least two)
    {{"label": "short branch name", "when": "the condition in words", "steps": [STEP, ...]}}
  ]
}}

Kinds:
- lookup: fetch data from a system or database
- compute: derive, compare, format or flag values from data already fetched (no judgement)
- review: needs reading, judgement, or writing text (a person would think about it)
- decision: the procedure branches ("if ... then ... otherwise ...")
- approval: wait for a person to approve, sign off or respond

Rules:
- A decision must test ONE simple value. If the procedure's condition is complex, add a compute
  step just before the decision that produces that value (e.g. a Y/N flag) and set depends_on to it.
- Every branch has at least one step. If a branch just continues or ends the review, add one
  compute step that records that outcome.
- Steps inside a branch only run on that branch. When each branch ends by producing the same
  kind of outcome (e.g. "ICMP review outcome"), use the identical `produces` text in each branch
  so later steps can read that outcome whichever branch ran."""


def extract_procedure(client: LLMClient, text: str, *, model: str, source: str = "") -> Procedure:
    numbered = "\n".join(f"{i}| {line}" for i, line in enumerate(text.splitlines(), 1))
    prompt = EXTRACT_PROMPT.format(numbered=numbered, kinds=" | ".join(STEP_KINDS))
    data = complete_json(client, model=model, system=EXTRACT_SYSTEM, prompt=prompt,
                         max_tokens=8000, validate=_validate_raw)
    proc = _parse(data)
    proc.source = source
    return proc


def _validate_raw(data) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list) or not data["phases"]:
        raise ValueError("expected an object with a non-empty 'phases' list")
    proc = _parse(data)
    ids: set[str] = set()
    known = set(proc.inputs)
    problems: list[str] = []

    def walk(steps: list[Step], visible: set[str]):
        visible = set(visible)
        for s in steps:
            if s.id in ids:
                problems.append(f"duplicate step id {s.id}")
            ids.add(s.id)
            if s.kind not in STEP_KINDS:
                problems.append(f"{s.id}: kind {s.kind!r} is not one of {STEP_KINDS}")
            for u in s.uses:
                if u not in visible and u not in known:
                    problems.append(f"{s.id}: uses {u!r}, which is not an earlier step or input")
            if s.kind == "decision":
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


def _parse(data: dict) -> Procedure:
    def step(d: dict) -> Step:
        return Step(
            id=str(d.get("id", "")).strip(),
            title=str(d.get("title", "")).strip(),
            text=str(d.get("text", "")).strip(),
            kind=str(d.get("kind", "")).strip().lower(),
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
                      [step(s) for s in p.get("steps") or []])
                for i, p in enumerate(data["phases"], 1)],
    )
