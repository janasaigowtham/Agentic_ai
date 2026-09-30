"""Compiles the procedure itself into a skill tree (the Corpus2Skill treatment).

The procedure is the corpus: it is split into sections, each section is
summarised by the LLM and embedded, and the result is written as a browsable
tree next to the catalog. Unlike the catalog, the hierarchy comes from the
procedure's own headings, in order, because the order of steps is the workflow.
Embeddings are used for what headings can't give:

  * Related sections: other parts of the procedure most similar to this one
    (definitions, appendices, rules stated elsewhere).
  * Likely tools & data: the catalog documents (metadata, tools, reference)
    most similar to this section, from the catalog's saved vectors.

Layout (under <out>/.claude/skills/procedure/):

    SKILL.md                         whole-procedure summary + ordered sections
    01-pre-process/INDEX.md          section summary, lines, doc rows, related, likely tools & data
    01-pre-process/01-.../INDEX.md   sub-sections, in document order
    documents.json                   section text by doc ID ("p" + hash)
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from o2a_gen.llm import LLMClient, complete_json
from o2a_gen.skilltree import _slug, embed_texts, load_vectors

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PART_CHARS = 3000


@dataclass
class Section:
    title: str
    level: int
    start: int                     # 1-based line of the heading (or first line)
    end: int = 0                   # last line, including sub-sections
    own: list[str] = field(default_factory=list)   # lines before the first sub-section
    own_start: int = 1             # line number of own[0]
    children: list["Section"] = field(default_factory=list)
    summary: str = ""
    path: str = ""
    doc_ids: list[str] = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class ProcDoc:
    id: str
    section: Section
    start: int
    end: int
    text: str


def parse_sections(text: str, title: str) -> Section:
    """Heading tree in document order. Text before the first heading is 'Preamble'."""
    lines = text.splitlines()
    root = Section(title=title, level=0, start=1, end=len(lines))
    stack = [root]
    current = root
    for i, line in enumerate(lines, 1):
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack[-1].level >= level:
                stack.pop()
            sec = Section(title=m.group(2).strip(), level=level, start=i, own_start=i + 1)
            stack[-1].children.append(sec)
            stack.append(sec)
            current = sec
        else:
            current.own.append(line)
    # A single top-level heading is the document title: fold it into the root.
    if len(root.children) == 1 and not "".join(root.own).strip():
        only = root.children[0]
        root.title, root.own, root.children = only.title, only.own, only.children
        root.own_start = only.own_start
    if "".join(root.own).strip() and root.children:
        pre = Section("Preamble", 1, root.own_start, own=root.own, own_start=root.own_start)
        root.own = []
        root.children.insert(0, pre)
    _set_ends(root, len(lines))
    return root


def _set_ends(sec: Section, doc_end: int) -> None:
    for i, c in enumerate(sec.children):
        nxt = sec.children[i + 1].start - 1 if i + 1 < len(sec.children) else doc_end
        _set_ends(c, nxt)
    sec.end = doc_end if sec.level == 0 else max(doc_end, sec.start)


def _doc_id(start: int, part: int, text: str) -> str:
    return "p" + hashlib.sha1(f"{start}:{part}:{text[:200]}".encode()).hexdigest()[:13]


def section_docs(root: Section) -> list[ProcDoc]:
    """One document per section's own text; long text is split on blank lines."""
    docs: list[ProcDoc] = []
    for sec in root.walk():
        body = "\n".join(sec.own).strip()
        if not body:
            continue
        own_start = sec.own_start
        parts, buf = [], ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > _PART_CHARS:
                parts.append(buf)
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        parts.append(buf)
        own_end = own_start + len(sec.own) - 1
        for n, part in enumerate(parts):
            d = ProcDoc(_doc_id(sec.start, n, part), sec, own_start, own_end,
                        f"[procedure] {sec.title}"
                        + (f" (part {n + 1})" if len(parts) > 1 else "")
                        + f"\nLines {own_start}-{own_end}\n\n{part}")
            sec.doc_ids.append(d.id)
            docs.append(d)
    return docs


SECTION_SYSTEM = ("You summarise sections of an operations review procedure so an agent can "
                  "find the right part quickly. Be concrete: name the data, systems, decisions "
                  "and approvals involved.")
SECTION_PROMPT = """Section: {title}
{body}

Return JSON: {{"summary": "2-3 sentences: what this part of the procedure does or defines"}}"""


def summarise(client: LLMClient, root: Section, model: str, workers: int) -> None:
    levels: list[list[Section]] = []

    def collect(s: Section, d: int):
        while len(levels) <= d:
            levels.append([])
        levels[d].append(s)
        for c in s.children:
            collect(c, d + 1)

    collect(root, 0)

    def one(s: Section):
        body = "\n".join(s.own).strip()[:4000]
        if s.children:
            body += "\n\nSub-sections:\n" + "\n".join(f"- {c.title}: {c.summary}" for c in s.children)
        try:
            out = complete_json(client, model=model, system=SECTION_SYSTEM,
                                prompt=SECTION_PROMPT.format(title=s.title, body=body),
                                max_tokens=300)
            s.summary = str(out.get("summary") or "").strip()
        except ValueError:
            s.summary = ""
        if not s.summary:
            s.summary = body.splitlines()[0][:200] if body else s.title

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for level in reversed(levels):
            list(pool.map(one, level))


@dataclass
class ProcTree:
    root: Section
    docs: list[ProcDoc]
    related: dict[str, list[tuple[str, float]]]
    hints: dict[str, list[tuple[str, float]]]
    out_dir: Path

    def doc_for_line(self, line: int) -> ProcDoc | None:
        """The most specific section document covering a procedure line."""
        best = None
        for d in self.docs:
            if d.section.start <= line <= d.section.end:
                if best is None or d.section.level >= best.section.level:
                    best = d
        return best

    def section_path_for_line(self, line: int) -> str:
        d = self.doc_for_line(line)
        return f"procedure/{d.section.path}".rstrip("/") if d else ""

    def context_for_line(self, line: int, index: dict[str, dict]) -> str:
        d = self.doc_for_line(line)
        if d is None:
            return ""
        lines = [f"This step is in procedure section `{self.section_path_for_line(line)}` "
                 f"(lines {d.section.start}-{d.section.end}), doc `{d.id}`."]
        if self.related.get(d.id):
            lines.append("Related procedure sections (by similarity):")
            by_id = {x.id: x for x in self.docs}
            for rid, score in self.related[d.id]:
                lines.append(f"- `{rid}` {by_id[rid].section.title} ({score:.2f})")
        if self.hints.get(d.id):
            lines.append("Likely tools & data for this section (by similarity; verify before use):")
            for cid, score in self.hints[d.id]:
                meta = index.get(cid, {})
                lines.append(f"- `{cid}` [{meta.get('kind', '?')}] {meta.get('name', cid)} ({score:.2f})")
        return "\n".join(lines)


def build_procedure_tree(client: LLMClient, text: str, title: str, out_dir: Path, *,
                         models: dict, embedding: dict, catalog_root: Path | None,
                         catalog_index: dict[str, dict] | None = None, workers: int = 8,
                         related_k: int = 3, hint_k: int = 5) -> ProcTree:
    out_dir = Path(out_dir)
    root = parse_sections(text, title)
    docs = section_docs(root)
    if not docs:
        raise ValueError("the procedure has no text")

    summarise(client, root, models["catalog_summary"], workers)

    texts = [f"{d.section.title}. {d.section.summary}\n{d.text[:2000]}" for d in docs]
    vecs = embed_texts(client, texts, embedding)

    related: dict[str, list[tuple[str, float]]] = {}
    sims = vecs @ vecs.T
    for i, d in enumerate(docs):
        order = [j for j in np.argsort(-sims[i]) if docs[j].section is not d.section]
        related[d.id] = [(docs[j].id, float(sims[i, j])) for j in order[:related_k]]

    hints: dict[str, list[tuple[str, float]]] = {}
    saved = load_vectors(catalog_root, embedding) if catalog_root else None
    if saved is not None:
        ids, mat = saved
        # Section -> catalog is a search: embed sections as queries if a query_template is set.
        qvecs = embed_texts(client, texts, embedding, as_query=True) \
            if embedding.get("query_template") else vecs
        kinds = catalog_index or {}
        keep = [k for k, cid in enumerate(ids) if kinds.get(cid, {}).get("kind") != "schema"]
        if keep:
            sub = mat[keep]
            csims = qvecs @ sub.T
            for i, d in enumerate(docs):
                top = np.argsort(-csims[i])[:hint_k]
                hints[d.id] = [(ids[keep[j]], float(csims[i, j])) for j in top]
    elif catalog_root:
        print("  (catalog has no saved vectors for this embedding model/settings: "
              "rebuild it with compile-catalog to get tool & data hints)")

    tree = ProcTree(root, docs, related, hints, out_dir)
    _write(tree, catalog_index or {})
    return tree


def _write(tree: ProcTree, catalog_index: dict[str, dict]) -> None:
    import shutil

    skills = tree.out_dir / ".claude" / "skills"
    if skills.exists():
        shutil.rmtree(skills)
    base = skills / "procedure"
    by_id = {d.id: d for d in tree.docs}

    def assign(sec: Section, parent: str):
        for n, c in enumerate(sec.children, 1):
            c.path = f"{parent}/{n:02d}-{_slug(c.title) or 'section'}".lstrip("/")
            assign(c, c.path)

    tree.root.path = ""
    assign(tree.root, "")

    for sec in tree.root.walk():
        folder = base / sec.path if sec.path else base
        folder.mkdir(parents=True, exist_ok=True)
        desc = " ".join(sec.summary.split())[:400] or sec.title
        out = ["---", f"name: {sec.path.split('/')[-1] if sec.path else 'procedure'}",
               "description: >", f"  {desc}", "---", "", f"# {sec.title}", "",
               f"Lines {sec.start}-{sec.end}.", "", "## Overview", "", sec.summary or sec.title, ""]
        if sec.doc_ids:
            out += ["## Text", "", "Read with get_document(doc_id).", ""]
            out += [f"- `{i}` - lines {by_id[i].start}-{by_id[i].end}" for i in sec.doc_ids]
            out.append("")
        if sec.children:
            out += ["## Sub-sections (in procedure order)", ""]
            out += [f"- **{c.path.split('/')[-1]}/** (lines {c.start}-{c.end}): {c.summary}"
                    for c in sec.children]
            out.append("")
        rel = [r for i in sec.doc_ids for r in tree.related.get(i, [])]
        if rel:
            out += ["## Related procedure sections", ""]
            out += [f"- `{rid}` procedure/{by_id[rid].section.path} ({s:.2f})" for rid, s in rel]
            out.append("")
        hin = [h for i in sec.doc_ids for h in tree.hints.get(i, [])]
        if hin:
            out += ["## Likely tools & data", ""]
            out += [f"- `{cid}` [{catalog_index.get(cid, {}).get('kind', '?')}] "
                    f"{catalog_index.get(cid, {}).get('name', cid)} ({s:.2f})" for cid, s in hin]
            out.append("")
        (folder / ("SKILL.md" if not sec.path else "INDEX.md")).write_text("\n".join(out), encoding="utf-8")

    (tree.out_dir / "documents.json").write_text(
        json.dumps({d.id: d.text for d in tree.docs}, ensure_ascii=False), encoding="utf-8")
    (tree.out_dir / "catalog_index.json").write_text(json.dumps(
        {d.id: {"id": d.id, "kind": "procedure", "name": d.section.title,
                "source": f"lines {d.start}-{d.end}"} for d in tree.docs}, indent=2))
    (tree.out_dir / "procedure_links.json").write_text(json.dumps(
        {"related": tree.related, "hints": tree.hints}, indent=2))
