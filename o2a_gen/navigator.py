"""Browses a compiled catalog skill tree to find the documents a step needs.

This replaces Corpus2Skill's serve.py for our use: no Skills API upload and no
server-side code execution. The model drives navigation by replying with one
JSON action per turn, so any chat model works, including models without
native tool calling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from o2a_gen.llm import LLMClient, extract_json

_READ_LIMIT = 6000

NAV_SYSTEM = """You are looking up reference material in a catalog organised as a folder tree.
It holds the procedure being implemented (folder `procedure/`, sections in document order,
each with related sections and likely tools & data), the O2A agent syntax, data metadata
(tables, columns, connections), tool definitions and reference documents. Each folder has a SKILL.md (top level) or INDEX.md (deeper) describing
what it contains and listing document IDs.

Reply with exactly ONE JSON object per turn, no prose:
  {"action": "read", "path": "<folder>/SKILL.md"}      read a navigation file
  {"action": "ls", "path": "<folder>"}                 list a folder
  {"action": "find", "term": "<name or phrase>"}       look up a table/agent/tool/entity by name
  {"action": "get_document", "doc_id": "<id>"}         read a full document
  {"action": "done", "doc_ids": ["<id>", ...], "notes": "<what you found and what is missing>"}

Method:
1. Start from the step's procedure section if you are given one; check its related sections
   and its likely tools & data. Otherwise scan the top-level skills.
2. Drill down through INDEX.md files, or use find when you know a name.
3. Read the full documents you intend to rely on with get_document.
4. Finish with done, listing only the doc_ids that are actually relevant: the table
   definitions and connections a query needs, the tools a review needs, the guidelines it applies.
If nothing relevant exists, finish with done, an empty list, and say so in notes."""


@dataclass
class NavResult:
    doc_ids: list[str]
    docs: dict[str, str]
    notes: str
    turns: int
    trace: list[dict] = field(default_factory=list)


class Catalog:
    """A compiled catalog directory (output of ``o2a-gen compile-catalog``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.skills_dir = self.root / ".claude" / "skills"
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(f"no compiled skill tree at {self.skills_dir}")
        self.documents: dict[str, str] = _load_json(self.root / "documents.json", {})
        self.index: dict[str, dict] = _load_json(self.root / "catalog_index.json", {})
        self.entities: dict[str, dict] = _load_json(self.root / "entity_index.json", {})

    def _safe(self, rel: str) -> Path:
        p = (self.skills_dir / rel.strip().lstrip("/")).resolve()
        if p != self.skills_dir.resolve() and self.skills_dir.resolve() not in p.parents:
            raise ValueError("path is outside the catalog")
        return p

    def top_level(self) -> str:
        rows = []
        for d in sorted(p for p in self.skills_dir.iterdir() if p.is_dir()):
            desc = ""
            skill_md = d / "SKILL.md"
            if skill_md.exists():
                m = re.search(r"description:\s*>?\s*\n?\s*(.+)", skill_md.read_text(encoding="utf-8"))
                desc = m.group(1).strip() if m else ""
            rows.append(f"- {d.name}/ : {desc[:300]}")
        return "\n".join(rows)

    def ls(self, rel: str) -> str:
        p = self._safe(rel)
        if not p.is_dir():
            return f"not a folder: {rel}"
        return "\n".join(f"{c.name}{'/' if c.is_dir() else ''}" for c in sorted(p.iterdir())) or "(empty)"

    def read(self, rel: str) -> str:
        p = self._safe(rel)
        if p.is_dir():
            for name in ("SKILL.md", "INDEX.md"):
                if (p / name).exists():
                    p = p / name
                    break
        if not p.is_file():
            return f"not found: {rel}"
        return _clip(p.read_text(encoding="utf-8", errors="replace"))

    def syntax_for(self, agent_class: str) -> str:
        """The agent-syntax sections for this class plus the general sections."""
        ids = [i for i, m in self.index.items() if m.get("kind") == "schema"
               and (m.get("name") == agent_class
                    or str(m.get("name", "")).startswith((f"{agent_class} (part ", "general")))]
        ids.sort(key=lambda i: self.index[i]["name"] != agent_class)  # class section first
        return "\n\n".join(self.documents.get(i, "") for i in ids).strip()

    def classes(self) -> list[str]:
        """The agent classes the agent syntax document defines."""
        names = [m["name"] for m in self.index.values() if m.get("kind") == "schema"]
        return sorted({n for n in names if not n.startswith("general") and " (part " not in n})

    def syntax_document(self) -> str:
        """Every agent-syntax section, general ones first."""
        ids = [i for i, m in self.index.items() if m.get("kind") == "schema"]
        ids.sort(key=lambda i: not str(self.index[i]["name"]).startswith("general"))
        return "\n\n".join(self.documents.get(i, "") for i in ids).strip()

    def get(self, doc_id: str) -> str | None:
        return self.documents.get(doc_id)

    def find(self, term: str, limit: int = 15) -> str:
        t = term.strip().lower()
        if not t:
            return "empty term"
        hits = [f"- doc `{did}` [{meta.get('kind')}] {meta.get('name')}"
                for did, meta in self.index.items() if t in str(meta.get("name", "")).lower()]
        for ent, rec in self.entities.items():
            if t in ent.lower():
                paths = ", ".join(rec.get("skill_paths", [])[:4])
                hits.append(f"- entity '{ent}' appears in: {paths}")
        if not hits:  # fall back to a plain text search over document bodies
            hits = [f"- doc `{did}` [{self.index.get(did, {}).get('kind', '?')}] "
                    f"{self.index.get(did, {}).get('name', did)}"
                    for did, text in self.documents.items() if t in text.lower()]
        return "\n".join(hits[:limit]) or f"no matches for {term!r}"


class CombinedCatalog:
    """Several compiled trees browsed as one (e.g. the procedure tree + the catalog).
    Top-level folder names must not collide; the first tree wins if they do."""

    def __init__(self, *catalogs: Catalog):
        self.parts = [c for c in catalogs if c is not None]
        self.documents: dict[str, str] = {}
        self.index: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        for c in reversed(self.parts):
            self.documents.update(c.documents)
            self.index.update(c.index)
            self.entities.update(c.entities)

    def _owner(self, rel: str) -> Catalog | None:
        first = rel.strip().lstrip("/").split("/", 1)[0]
        return next((c for c in self.parts if first and (c.skills_dir / first).exists()), None)

    def top_level(self) -> str:
        return "\n".join(c.top_level() for c in self.parts)

    def ls(self, rel: str) -> str:
        if not rel.strip().strip("/"):
            return "\n".join(c.ls("") for c in self.parts)
        owner = self._owner(rel)
        return owner.ls(rel) if owner else f"not a folder: {rel}"

    def read(self, rel: str) -> str:
        owner = self._owner(rel)
        return owner.read(rel) if owner else f"not found: {rel}"

    def get(self, doc_id: str) -> str | None:
        return self.documents.get(doc_id)

    def find(self, term: str, limit: int = 15) -> str:
        hits = [h for c in self.parts for h in c.find(term, limit).splitlines()
                if not h.startswith("no matches")]
        return "\n".join(hits[:limit]) or f"no matches for {term!r}"

    def syntax_for(self, agent_class: str) -> str:
        return next((t for c in self.parts if (t := c.syntax_for(agent_class))), "")

    def classes(self) -> list[str]:
        return sorted({c for part in self.parts for c in part.classes()})

    def syntax_document(self) -> str:
        return next((t for c in self.parts if (t := c.syntax_document())), "")


def navigate(client: LLMClient, catalog: Catalog | CombinedCatalog, question: str, *,
             model: str, max_turns: int = 12) -> NavResult:
    messages = [{"role": "user", "content": (
        f"What I need:\n{question}\n\nTop-level skills:\n{catalog.top_level()}"
    )}]
    fetched: dict[str, str] = {}
    trace: list[dict] = []
    for turn in range(1, max_turns + 1):
        reply = client.complete(model=model, system=NAV_SYSTEM, messages=messages,
                                max_tokens=600, temperature=0.0).text
        try:
            action = extract_json(reply)
            if not isinstance(action, dict):
                raise ValueError("expected a JSON object")
            kind = action.get("action")
        except ValueError as e:
            action, kind = {}, None
            observation = f"Invalid reply ({e}). Reply with one JSON action."
        trace.append(action)
        try:
            if kind == "done":
                ids = [i for i in action.get("doc_ids", []) if i in catalog.documents]
                for i in ids:
                    fetched.setdefault(i, catalog.documents[i])
                return NavResult(ids, {i: fetched[i] for i in ids},
                                 str(action.get("notes", "")), turn, trace)
            elif kind == "read":
                observation = catalog.read(str(action.get("path", "")))
            elif kind == "ls":
                observation = catalog.ls(str(action.get("path", "")))
            elif kind == "find":
                observation = catalog.find(str(action.get("term", "")))
            elif kind == "get_document":
                did = str(action.get("doc_id", ""))
                text = catalog.get(did)
                if text is None:
                    observation = f"unknown doc_id {did!r}"
                else:
                    fetched[did] = text
                    observation = _clip(text)
            elif kind is not None:
                observation = f"unknown action {kind!r}"
        except ValueError as e:
            observation = f"error: {e}"
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": observation}]
    # Out of turns: hand back whatever was read in full.
    return NavResult(list(fetched), fetched, "stopped at max_turns", max_turns, trace)


def _clip(text: str) -> str:
    return text if len(text) <= _READ_LIMIT else text[:_READ_LIMIT] + "\n…[truncated]"


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
