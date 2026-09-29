"""Builds the grounding catalog: existing agents, tables, tools and reference docs.

Source layout (every folder is optional):

    catalog_src/
      agents/     existing O2A agent YAMLs, from any pipeline (best examples to copy)
      tables/     data dictionary: .sql DDL, .md/.txt notes, .csv (table,column,type,description),
                  or .yaml/.json ({name, columns, connection_env, ...} or a list of them)
      tools/      tool definitions: .yaml/.json ({name, ...} or list) or .md/.txt
      reference/  policies, guidelines, SOP excerpts (.md/.txt)

The catalog is compiled into a navigable skill tree by o2a_gen.skilltree (or,
optionally, the vendored Corpus2Skill). ``catalog_index.json`` maps each short doc ID back to
its kind and name.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

KINDS = ("agents", "tables", "tools", "reference")
_MAX_CHARS = 7500  # stay under Corpus2Skill's default max_doc_chars (8000)


@dataclass
class CatalogDoc:
    id: str
    kind: str      # agent | table | tool | reference
    name: str
    source: str
    text: str


def _doc_id(kind: str, name: str, part: int = 0) -> str:
    # Corpus2Skill truncates IDs to 16 chars, so keep them short and unique.
    h = hashlib.sha1(f"{kind}:{name}:{part}".encode()).hexdigest()[:13]
    return f"{kind[0]}{h}"


def _mk(kind: str, name: str, source: Path, body: str, part: int = 0) -> CatalogDoc:
    header = f"[{kind}] {name}\n\n"
    return CatalogDoc(_doc_id(kind, name, part), kind, name, str(source), header + body.strip())


def _load_structured(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _agents(folder: Path) -> list[CatalogDoc]:
    docs = []
    for p in sorted(folder.glob("*.y*ml")):
        raw = p.read_text(encoding="utf-8", errors="replace")
        try:
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or p.stem)
        cls = data.get("agent_class", "?")
        desc = str(data.get("description") or "").strip()
        body = (f"Existing O2A agent `{name}` (agent_class: {cls}).\n"
                + (f"Purpose: {desc}\n" if desc else "")
                + f"\n```yaml\n{raw.strip()}\n```")
        docs.append(_mk("agent", name, p, body[:_MAX_CHARS]))
    return docs


_CREATE_TABLE = re.compile(r"create\s+(?:multiset\s+|set\s+)?table\s+([\w.\"]+)", re.I)


def _tables(folder: Path) -> list[CatalogDoc]:
    docs = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".csv":
            by_table: dict[str, list[dict]] = {}
            with p.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                    if row.get("table"):
                        by_table.setdefault(row["table"], []).append(row)
            for table, rows in by_table.items():
                lines = [f"Table `{table}` columns:"]
                for r in rows:
                    lines.append(f"- {r.get('column', '?')} ({r.get('type', '?')}): "
                                 f"{r.get('description', '')}".rstrip(": "))
                docs.append(_mk("table", table, p, "\n".join(lines)[:_MAX_CHARS]))
        elif suffix == ".sql":
            text = p.read_text(encoding="utf-8", errors="replace")
            starts = [m.start() for m in _CREATE_TABLE.finditer(text)]
            if not starts:
                docs.append(_mk("table", p.stem, p, f"```sql\n{text[:_MAX_CHARS]}\n```"))
                continue
            for i, s in enumerate(starts):
                chunk = text[s:starts[i + 1] if i + 1 < len(starts) else len(text)]
                name = _CREATE_TABLE.match(chunk).group(1).strip('"')
                docs.append(_mk("table", name, p, f"```sql\n{chunk.strip()[:_MAX_CHARS]}\n```"))
        elif suffix in (".yaml", ".yml", ".json"):
            data = _load_structured(p)
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("table") or p.stem)
                    docs.append(_mk("table", name, p, yaml.safe_dump(item, sort_keys=False)[:_MAX_CHARS]))
        elif suffix in (".md", ".txt"):
            docs.extend(_split_text("table", p.stem, p))
    return docs


def _tools(folder: Path) -> list[CatalogDoc]:
    docs = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".yaml", ".yml", ".json"):
            data = _load_structured(p)
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict):
                    name = str(item.get("name") or p.stem)
                    docs.append(_mk("tool", name, p, yaml.safe_dump(item, sort_keys=False)[:_MAX_CHARS]))
        elif p.suffix.lower() in (".md", ".txt"):
            docs.extend(_split_text("tool", p.stem, p))
    return docs


def _split_text(kind: str, name: str, path: Path) -> list[CatalogDoc]:
    """One doc per file, split on markdown headings if the file is long."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= _MAX_CHARS:
        return [_mk(kind, name, path, text)]
    sections = re.split(r"(?m)^(?=#{1,3} )", text)
    docs, buf, part = [], "", 0
    for sec in sections:
        if buf and len(buf) + len(sec) > _MAX_CHARS:
            docs.append(_mk(kind, f"{name} (part {part + 1})", path, buf, part))
            buf, part = "", part + 1
        while len(sec) > _MAX_CHARS:  # a single oversized section
            docs.append(_mk(kind, f"{name} (part {part + 1})", path, sec[:_MAX_CHARS], part))
            sec, part = sec[_MAX_CHARS:], part + 1
        buf += sec
    if buf.strip():
        docs.append(_mk(kind, f"{name} (part {part + 1})", path, buf, part))
    return docs


def collect_catalog(src: Path) -> list[CatalogDoc]:
    src = Path(src)
    loaders = {"agents": _agents, "tables": _tables, "tools": _tools,
               "reference": lambda f: [d for p in sorted(f.glob("*")) if p.suffix.lower() in (".md", ".txt")
                                       for d in _split_text("reference", p.stem, p)]}
    docs: list[CatalogDoc] = []
    for kind in KINDS:
        folder = src / kind
        if folder.is_dir():
            docs.extend(loaders[kind](folder))
    seen: set[str] = set()
    unique = []
    for d in docs:
        if d.id not in seen:
            seen.add(d.id)
            unique.append(d)
    return unique


def write_corpus(docs: list[CatalogDoc], out_dir: Path) -> Path:
    """Write the JSONL corpus Corpus2Skill reads, plus catalog_index.json."""
    corpus_dir = out_dir / "catalog_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with (corpus_dir / "catalog.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d.id, "contents": d.text}, ensure_ascii=False) + "\n")
    index = {d.id: {k: v for k, v in asdict(d).items() if k != "text"} for d in docs}
    (out_dir / "catalog_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return corpus_dir


def compile_catalog(src: Path, out_dir: Path, client, cfg) -> Path:
    """Collect the catalog and compile it into a skill tree. Returns the skills dir.

    catalog.engine: native (default, o2a_gen.skilltree) or corpus2skill (vendored copy).
    """
    out_dir = Path(out_dir)
    docs = collect_catalog(Path(src))
    if not docs:
        raise ValueError(f"no catalog documents found under {src} (expected {', '.join(KINDS)}/)")
    engine = cfg.catalog.get("engine", "native")
    if engine == "native":
        from o2a_gen.skilltree import build_skill_tree
        write_corpus(docs, out_dir)  # catalog_index.json for the navigator; JSONL for reference
        return build_skill_tree(client, docs, out_dir, models=cfg.models,
                                embedding=cfg.embedding, catalog=cfg.catalog)
    if engine == "corpus2skill":
        from o2a_gen.c2s_bridge import compile_with_corpus2skill
        corpus_dir = write_corpus(docs, out_dir)
        return compile_with_corpus2skill(
            client, corpus_dir=corpus_dir, output_dir=out_dir,
            models=cfg.models, embedding=cfg.embedding, catalog=cfg.catalog,
        )
    raise ValueError(f"unknown catalog.engine {engine!r} (use 'native' or 'corpus2skill')")
