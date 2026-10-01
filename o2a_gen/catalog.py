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
      playbook/   how to build the workflow: logic checks, orchestration, data mapping
                  (.md/.txt). When present, generation runs the playbook harness (harness.py).

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

KINDS = ("schema", "metadata", "tools", "reference", "playbook")
_MAX_CHARS = 7500  # stay under Corpus2Skill's default max_doc_chars (8000)


@dataclass
class CatalogDoc:
    id: str
    kind: str      # schema | metadata | tool | reference | playbook
    name: str
    source: str
    text: str


def _doc_id(kind: str, name: str, part: int | str = 0) -> str:
    # Corpus2Skill truncates IDs to 16 chars, so keep them short and unique.
    h = hashlib.sha1(f"{kind}:{name}:{part}".encode()).hexdigest()[:13]
    return f"{'b' if kind == 'playbook' else kind[0]}{h}"   # 'p' is the procedure tree's


def _mk(kind: str, name: str, source: Path, body: str, part: int | str = 0) -> CatalogDoc:
    header = f"[{kind}] {name}\n\n"
    return CatalogDoc(_doc_id(kind, name, part), kind, name, str(source), header + body.strip())


def _load_structured(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def discover_classes(text: str) -> set[str]:
    """Agent classes named by a syntax document: the first column of a table whose header
    mentions a class, `agent_class: X` values that a heading also names, and headings that
    are just a class identifier (`## database_agent`, `## 4. LlmAgent syntax`)."""
    classes: set[str] = set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and "class" in line.lower() and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            for row in lines[i + 2:]:
                if not row.lstrip().startswith("|"):
                    break
                cell = row.strip().strip("|").split("|")[0].strip().strip("`* ")
                if re.fullmatch(r"[A-Za-z_]\w*", cell):
                    classes.add(cell)
    for h in lines:
        if not h.startswith("#"):
            continue
        title = re.sub(r"^#+\s*(\d+(\.\d+)*\.?\s*)?", "", h).strip().strip("`*")
        title = re.sub(r"\s+(syntax|class|agent class)$", "", title, flags=re.I).strip("`* ")
        if re.fullmatch(r"[A-Za-z]\w*", title) and ("_" in title or re.search(r"[a-z][A-Z]", title)):
            classes.add(title)
    headings = " ".join(h for h in lines if h.startswith("#"))
    for m in re.finditer(r"agent_class:\s*[\"']?([A-Za-z_]\w*)", text):
        if re.search(rf"\b{re.escape(m.group(1))}\b", headings):
            classes.add(m.group(1))
    return classes


def _schema(folder: Path, known_classes: set[str]) -> list[CatalogDoc]:
    """One doc per agent_class section of the syntax document(s), including its
    sub-headings; everything else is 'general'."""
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
        by_class: dict[str, list[str]] = {}
        general: list[str] = []
        current, level = None, 0
        for sec in re.split(r"(?m)^(?=#{1,6} )", text):
            heading = sec.split("\n", 1)[0]
            hl = len(heading) - len(heading.lstrip("#")) if heading.startswith("#") else 0
            cls = next((c for c in sorted(known_classes, key=len, reverse=True)
                        if hl and re.search(rf"\b{re.escape(c)}\b", heading)), None)
            if cls:
                current, level = cls, hl
                by_class.setdefault(cls, []).append(sec)
            elif current and hl > level:
                by_class[current].append(sec)
            else:
                current = None
                if sec.strip():
                    general.append(sec)
        for cls, parts in by_class.items():
            body = "".join(parts)
            chunks = [body[i:i + _MAX_CHARS] for i in range(0, len(body), _MAX_CHARS)]
            for n, chunk in enumerate(chunks):
                docs.append(_mk("schema", cls if n == 0 else f"{cls} (part {n + 1})", p, chunk, n))
        if general:
            docs.extend(_split_text("schema", f"general ({p.stem})", p, "\n".join(general)))
    return docs


_NAME_KEYS = ("name", "tool_name", "table", "table_name", "operation_id", "operationId",
              "agent_name", "id", "source_system_id", "title")


def _item_name(item) -> str | None:
    if isinstance(item, dict):
        for k in _NAME_KEYS:
            if isinstance(item.get(k), (str, int)) and str(item[k]).strip():
                return str(item[k]).strip()
    return None


def _named_list(v) -> bool:
    return isinstance(v, list) and any(_item_name(x) for x in v)


def _structured_docs(kind: str, path: Path, data, name: str | None = None,
                     trail: str = "") -> list[CatalogDoc]:
    """Split any YAML/JSON into docs no larger than _MAX_CHARS without losing content.
    Lists of named items (tools, tables, ...) always become one doc per item; oversized
    mappings are split by key; the remaining small fields are kept together."""
    name = name or _item_name(data) or path.stem
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    where = f"(from {path.name}{': ' + trail if trail else ''})\n"
    has_named = _named_list(data) or (isinstance(data, dict) and not _item_name(data)
                                      and any(_named_list(v) for v in data.values()))
    if len(dumped) + len(where) <= _MAX_CHARS and not has_named:
        return [_mk(kind, name, path, where + dumped, f"{path.name}:{trail}")]
    docs: list[CatalogDoc] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            t = f"{trail}[{i}]"
            docs += _structured_docs(kind, path, item, _item_name(item) or f"{name}[{i}]", t)
        return docs
    if isinstance(data, dict):
        small: dict = {}
        for k, v in data.items():
            t = f"{trail}.{k}" if trail else str(k)
            piece = yaml.safe_dump({k: v}, sort_keys=False, allow_unicode=True, width=100)
            if _named_list(v):
                docs += _structured_docs(kind, path, v, str(k), t)
            elif len(piece) <= _MAX_CHARS // 4:
                small[k] = v
            else:
                docs += _structured_docs(kind, path, v, f"{name} / {k}", t)
        if small:
            label = name if not docs else f"{name} (overview)"
            dumped_small = yaml.safe_dump(small, sort_keys=False, allow_unicode=True, width=100)
            if len(dumped_small) + len(where) <= _MAX_CHARS:
                docs.append(_mk(kind, label, path, where + dumped_small, f"{path.name}:{trail}:small"))
            else:
                docs += _split_text(kind, label, path, where + dumped_small)
        return docs
    return _split_text(kind, name, path, where + dumped)

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
            docs.extend(_structured_docs("metadata", p, _load_structured(p)))
        elif suffix in (".md", ".txt"):
            docs.extend(_split_text("metadata", p.stem, p))
    return docs


def _tools(folder: Path) -> list[CatalogDoc]:
    docs = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".yaml", ".yml", ".json"):
            docs.extend(_structured_docs("tool", p, _load_structured(p)))
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
    src = Path(src)
    if (src / "agents").is_dir():
        print(f"WARNING: ignoring {src / 'agents'}: existing agent YAMLs are not a generation "
              "input. Pass them to `compare` (or generate --compare-with) instead.")
    classes = set()
    if (src / "schema").is_dir():
        for p in (src / "schema").iterdir():
            if p.suffix.lower() in (".md", ".txt"):
                classes |= discover_classes(p.read_text(encoding="utf-8", errors="replace"))
    metadata_dir = src / "metadata" if (src / "metadata").is_dir() else src / "tables"
    sources = [
        (src / "schema", lambda f: _schema(f, classes)),
        (metadata_dir, _metadata),
        (src / "tools", _tools),
        (src / "reference", lambda f: [d for p in sorted(f.glob("*"))
                                       if p.suffix.lower() in (".md", ".txt")
                                       for d in _split_text("reference", p.stem, p)]),
        # How to build the workflow: logic checks, orchestration, data mapping.
        (src / "playbook", lambda f: [d for p in sorted(f.glob("*"))
                                      if p.suffix.lower() in (".md", ".txt", ".markdown")
                                      for d in _split_text("playbook", p.stem, p)]),
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
