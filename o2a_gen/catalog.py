"""Builds the generation catalog: agent syntax, metadata, tools and reference docs.

Source layout:

    catalog_src/
      schema/     REQUIRED. The O2A agent syntax: the runtime schema of each agent_class
                  (.md/.txt split on headings that name a class, or .yaml/.json keyed by class)
      metadata/   data dictionary: .sql DDL, .md/.txt notes (e.g. which connection env var a
                  table uses), .csv (table,column,type,description), or .yaml/.json table specs.
                  (`tables/` is accepted as an alias.)
      tools/      tool definitions: .yaml/.json ({name, ...} or list) or .md/.txt
      reference/  policies, guidelines, SOP excerpts (.md/.txt)

Existing agent YAMLs are deliberately NOT an input: generation must work from the
procedure, tools, metadata and syntax alone. Existing YAMLs are only used afterwards,
by `compare`. An `agents/` folder here is ignored with a warning.

The catalog is compiled into a navigable skill tree by o2a_gen.skilltree (or,
optionally, the vendored Corpus2Skill). ``catalog_index.json`` maps each short doc ID
back to its kind and name.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

KINDS = ("schema", "metadata", "tools", "reference")
_MAX_CHARS = 7500  # stay under Corpus2Skill's default max_doc_chars (8000)


@dataclass
class CatalogDoc:
    id: str
    kind: str      # schema | metadata | tool | reference
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


def _schema(folder: Path, known_classes: set[str]) -> list[CatalogDoc]:
    """One doc per agent_class section of the syntax document(s); the rest is 'general'."""
    docs: list[CatalogDoc] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".yaml", ".yml", ".json"):
            data = _load_structured(p)
            if isinstance(data, dict):
                for key, spec in data.items():
                    name = str(key) if str(key) in known_classes else f"general: {key}"
                    docs.append(_mk("schema", name, p,
                                    yaml.safe_dump({key: spec}, sort_keys=False)[:_MAX_CHARS]))
            continue
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r"(?m)^(?=#{1,4} )", text)
        general: list[str] = []
        for sec in sections:
            heading = sec.split("\n", 1)[0]
            cls = next((c for c in known_classes if re.search(rf"\b{re.escape(c)}\b", heading)), None)
            if cls:
                docs.append(_mk("schema", cls, p, sec[:_MAX_CHARS]))
            elif sec.strip():
                general.append(sec)
        if general:
            docs.extend(_split_text("schema", f"general ({p.stem})", p, "\n".join(general)))
    return docs


_CREATE_TABLE = re.compile(r"create\s+(?:multiset\s+|set\s+)?table\s+([\w.\"]+)", re.I)


def _metadata(folder: Path) -> list[CatalogDoc]:
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
                docs.append(_mk("metadata", table, p, "\n".join(lines)[:_MAX_CHARS]))
        elif suffix == ".sql":
            text = p.read_text(encoding="utf-8", errors="replace")
            starts = [m.start() for m in _CREATE_TABLE.finditer(text)]
            if not starts:
                docs.append(_mk("metadata", p.stem, p, f"```sql\n{text[:_MAX_CHARS]}\n```"))
                continue
            for i, s in enumerate(starts):
                chunk = text[s:starts[i + 1] if i + 1 < len(starts) else len(text)]
                name = _CREATE_TABLE.match(chunk).group(1).strip('"')
                docs.append(_mk("metadata", name, p, f"```sql\n{chunk.strip()[:_MAX_CHARS]}\n```"))
        elif suffix in (".yaml", ".yml", ".json"):
            data = _load_structured(p)
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("table") or p.stem)
                    docs.append(_mk("metadata", name, p, yaml.safe_dump(item, sort_keys=False)[:_MAX_CHARS]))
        elif suffix in (".md", ".txt"):
            docs.extend(_split_text("metadata", p.stem, p))
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


def _split_text(kind: str, name: str, path: Path, text: str | None = None) -> list[CatalogDoc]:
    """One doc per file, split on markdown headings if the file is long."""
    if text is None:
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
    from o2a_gen.validate import KNOWN_CLASSES

    src = Path(src)
    if (src / "agents").is_dir():
        print(f"WARNING: ignoring {src / 'agents'}: existing agent YAMLs are not a generation "
              "input. Pass them to `compare` (or generate --compare-with) instead.")
    metadata_dir = src / "metadata" if (src / "metadata").is_dir() else src / "tables"
    sources = [
        (src / "schema", lambda f: _schema(f, KNOWN_CLASSES)),
        (metadata_dir, _metadata),
        (src / "tools", _tools),
        (src / "reference", lambda f: [d for p in sorted(f.glob("*"))
                                       if p.suffix.lower() in (".md", ".txt")
                                       for d in _split_text("reference", p.stem, p)]),
    ]
    docs: list[CatalogDoc] = []
    for folder, load in sources:
        if folder.is_dir():
            docs.extend(load(folder))
    if not any(d.kind == "schema" for d in docs):
        raise ValueError(f"{src / 'schema'} is required: add the O2A agent syntax document")
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
