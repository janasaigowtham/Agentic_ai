"""Native skill-tree compiler for the o2a_gen catalog.

Turns catalog documents into a folder tree an agent can browse:

    <out>/.claude/skills/<group>/SKILL.md        top level: what the group covers
    <out>/.claude/skills/<group>/<sub>/INDEX.md  deeper levels, down to document rows
    <out>/documents.json                          full text by doc ID
    <out>/entity_index.json                       names (tables, agent classes, tools, keywords) -> folders
    <out>/cards.json                              per-document cards, cached by content hash

Pipeline:
  1. Card: one LLM call per document -> title, one-line summary, keywords (cached).
  2. Embed: card + opening text, via the configured embedder.
  3. Partition: top level by catalog kind (syntax, metadata, tools, reference),
     then recursive k-means inside each kind until groups are small enough to list.
  4. Describe: bottom-up, one LLM call per folder -> folder name + summary.
  5. Write the tree, the document store and a name index (no LLM).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from o2a_gen.catalog import CatalogDoc
from o2a_gen.llm import LLMClient, complete_json_many

KIND_FOLDERS = {
    "schema": ("agent-syntax", "O2A runtime schema: fields and rules for each agent_class."),
    "metadata": ("data-metadata", "Database tables, columns and connections."),
    "tool": ("tools", "Tools agents can call."),
    "reference": ("reference-docs", "Policies, guidelines and procedure references."),
    "playbook": ("playbook", "How to build the workflow: logic checks, orchestration, data mapping."),
}


@dataclass
class TreeOptions:
    branching: int = 8          # max children per folder
    leaf_max: int = 12          # max documents listed in one folder
    min_group: int = 2          # smaller k-means groups are merged into their nearest neighbour
    max_depth: int = 4
    use_cards: bool = True
    partition_by_kind: bool = True
    workers: int = 8
    seed: int = 7

    @classmethod
    def from_config(cls, catalog: dict) -> "TreeOptions":
        known = cls.__dataclass_fields__
        return cls(**{k: type(known[k].default)(v) for k, v in catalog.items() if k in known})


@dataclass
class Node:
    doc_ids: list[str]
    children: list["Node"] = field(default_factory=list)
    label: str = ""
    summary: str = ""
    kind: str | None = None     # set on top-level kind folders
    path: str = ""

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


# --------------------------------------------------------------------------
# 1. cards
# --------------------------------------------------------------------------

CARD_SYSTEM = "You index catalog documents for a search tree. Be concrete and brief."
CARD_PROMPT = """Document:
{text}

Return JSON: {{"title": "short title (use the exact table, tool or agent_class name if it has one)",
"one_line": "one sentence on what it is and when to use it",
"keywords": ["up to 8 exact names or terms someone would search for: table, column, tool, field, policy names"]}}"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def make_cards(client: LLMClient, docs: list[CatalogDoc], model: str, cache_path: Path,
               workers: int, effort: str | None = None) -> dict[str, dict]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [d for d in docs if cache.get(d.id, {}).get("hash") != _hash(d.text)]
    got = complete_json_many(
        client, [(d.id, CARD_SYSTEM, CARD_PROMPT.format(text=d.text[:4000])) for d in todo],
        model=model, max_tokens=400, effort=effort, workers=workers, validate=_check_card)
    for d in todo:
        card = got.get(d.id) or {}
        if not card:
            print(f"  card for {d.name!r} built from the document itself (no valid model answer)")
        cache[d.id] = {"title": str(card.get("title") or d.name),
                       "one_line": str(card.get("one_line") or ""),
                       "keywords": [str(k) for k in (card.get("keywords") or [])][:8],
                       "hash": _hash(d.text)}
    cards = {d.id: cache[d.id] for d in docs}
    cache_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    return cards


def _check_card(d) -> None:
    if not isinstance(d, dict) or not d.get("title"):
        raise ValueError("need an object with a title")


def plain_card(d: CatalogDoc) -> dict:
    first = next((ln.strip("# ").strip() for ln in d.text.splitlines()[2:] if ln.strip()), "")
    return {"title": d.name, "one_line": first[:160], "keywords": [d.name]}


# --------------------------------------------------------------------------
# 2. embeddings
# --------------------------------------------------------------------------

_MODELS: dict[tuple, object] = {}   # loaded sentence-transformers models, one per config


def _local_model(embedding: dict):
    """Load a sentence-transformers model once per process (an 8B model is ~16 GB)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("embedding.provider 'local' needs: pip install sentence-transformers") from None
    key = (embedding.get("model"), bool(embedding.get("trust_remote_code", False)),
           embedding.get("device"), embedding.get("dtype"))
    if key not in _MODELS:
        kwargs: dict = {"trust_remote_code": key[1]}
        if key[2]:
            kwargs["device"] = key[2]
        if key[3]:
            kwargs["model_kwargs"] = {"torch_dtype": key[3]}
        model = SentenceTransformer(key[0], **kwargs)
        if embedding.get("max_seq_length"):
            model.max_seq_length = int(embedding["max_seq_length"])
        _MODELS[key] = model
    return _MODELS[key]


def _templated(texts: list[str], embedding: dict, as_query: bool) -> list[str]:
    """Apply `query_template` / `document_template` ({text}, {instruction}) if configured.
    With no query_template, queries are embedded exactly like documents (symmetric)."""
    tmpl = embedding.get("query_template") if as_query else None
    tmpl = tmpl or embedding.get("document_template")
    if not tmpl:
        return texts
    instr = embedding.get("query_instruction", "")
    return [tmpl.format(text=t, instruction=instr) for t in texts]


def embed_texts(client: LLMClient, texts: list[str], embedding: dict, batch: int | None = None,
                as_query: bool = False) -> np.ndarray:
    """Unit-length vectors, one per text. ``as_query`` uses `query_template` if set."""
    provider = embedding.get("provider", "local")
    model = embedding.get("model", "")
    batch = int(embedding.get("batch_size", 32)) if batch is None else batch
    texts = _templated(texts, embedding, as_query)
    if provider == "llm":
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch):
            vecs.extend(client.embed(texts[i:i + batch], model=model))
        arr = np.asarray(vecs, dtype=np.float32)
    elif provider == "local":
        arr = np.asarray(_local_model(embedding).encode(texts, batch_size=batch), dtype=np.float32)
    else:
        raise ValueError(f"unknown embedding.provider {provider!r}")
    if embedding.get("dimensions"):          # Matryoshka-style: keep the leading dimensions
        arr = arr[:, :int(embedding["dimensions"])]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)


def embedding_signature(embedding: dict) -> dict:
    """What must match for two sets of vectors to be comparable."""
    return {"provider": embedding.get("provider"), "model": embedding.get("model"),
            "dimensions": embedding.get("dimensions"),
            "document_template": embedding.get("document_template")}


# --------------------------------------------------------------------------
# 3. partition
# --------------------------------------------------------------------------

def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    """Cosine k-means with k-means++ seeding on L2-normalised rows. Returns labels."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        d = 1 - np.max(x @ np.stack(centers).T, axis=1)
        d = np.clip(d, 0, None)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1 / n)
        centers.append(x[rng.choice(n, p=p)])
    c = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if _ and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                m = members.mean(axis=0)
                c[j] = m / (np.linalg.norm(m) or 1)
    return labels


def split(ids: list[str], vecs: dict[str, np.ndarray], opt: TreeOptions, depth: int = 0) -> Node:
    node = Node(list(ids))
    if len(ids) <= opt.leaf_max or depth >= opt.max_depth:
        return node
    k = max(2, min(opt.branching, math.ceil(len(ids) / opt.leaf_max)))
    x = np.stack([vecs[i] for i in ids])
    labels = kmeans(x, k, opt.seed + depth)
    groups = [[ids[i] for i in np.flatnonzero(labels == j)] for j in range(k)]
    groups = _merge_small([g for g in groups if g], vecs, opt.min_group)
    if len(groups) < 2:
        return node
    node.children = [split(g, vecs, opt, depth + 1) for g in groups]
    return node


def _merge_small(groups: list[list[str]], vecs, min_group: int) -> list[list[str]]:
    groups = sorted(groups, key=len, reverse=True)
    while len(groups) > 1 and len(groups[-1]) < min_group:
        small = groups.pop()
        cents = [np.mean([vecs[i] for i in g], axis=0) for g in groups]
        c_small = np.mean([vecs[i] for i in small], axis=0)
        groups[int(np.argmax([float(c @ c_small) for c in cents]))].extend(small)
        groups.sort(key=len, reverse=True)
    return groups


# --------------------------------------------------------------------------
# 4. describe folders
# --------------------------------------------------------------------------

FOLDER_SYSTEM = ("You name and describe folders in a catalog tree so an agent can decide "
                 "which folder to open. Name what is inside, concretely.")
FOLDER_PROMPT = """This folder contains:
{contents}

Return JSON: {{"label": "2-5 word folder name, lowercase-with-hyphens",
"summary": "2-3 sentences: what is in here and which questions it answers; mention key names"}}"""


def describe(client: LLMClient, root: Node, cards: dict[str, dict], model: str, workers: int,
             effort: str | None = None) -> None:
    levels: list[list[Node]] = []

    def collect(n: Node, d: int):
        if len(levels) <= d:
            levels.append([])
        levels[d].append(n)
        for c in n.children:
            collect(c, d + 1)

    for top in root.children:
        collect(top, 0)

    def prompt(n: Node) -> str:
        if n.children:
            contents = "\n".join(f"- sub-folder '{c.label}': {c.summary}" for c in n.children)
        else:
            contents = "\n".join(f"- {cards[i]['title']}: {cards[i]['one_line']}"
                                 for i in n.doc_ids[:40])
        return FOLDER_PROMPT.format(contents=contents)

    def apply(n: Node, out: dict) -> None:
        summary = str(out.get("summary") or "").strip()
        if n.kind is None:
            n.label = _slug(str(out.get("label") or "")) or "group"
            n.summary = summary or f"{len(n.doc_ids)} catalog items."
        else:  # kind folders keep their fixed purpose line first
            n.summary = f"{n.summary} {summary}".strip()

    for level in reversed(levels):   # children first, so parents see their summaries
        got = complete_json_many(client, [(str(i), FOLDER_SYSTEM, prompt(n))
                                          for i, n in enumerate(level)],
                                 model=model, max_tokens=400, effort=effort, workers=workers)
        for i, n in enumerate(level):
            out = got.get(str(i))
            apply(n, out if isinstance(out, dict) else {})


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


# --------------------------------------------------------------------------
# 5. write
# --------------------------------------------------------------------------

def _assign_paths(n: Node, parent: str) -> None:
    used: set[str] = set()
    for c in n.children:
        base = c.label or "group"
        name, i = base, 2
        while name in used:
            name, i = f"{base}-{i}", i + 1
        used.add(name)
        c.path = f"{parent}/{name}" if parent else name
        _assign_paths(c, c.path)


def _render(n: Node, cards: dict[str, dict]) -> str:
    desc = " ".join(n.summary.split())[:400]
    lines = ["---", f"name: {n.path.split('/')[-1]}", "description: >", f"  {desc}",
             f"documents: {len(n.doc_ids)}", "---", "", "## Overview", "", n.summary, ""]
    if n.children:
        lines += ["## Sub-folders", "", "Open a sub-folder's INDEX.md to see what it holds.", ""]
        for c in n.children:
            lines.append(f"- **{c.path.split('/')[-1]}/** ({len(c.doc_ids)} items): {c.summary}")
        lines.append("")
    else:
        lines += [f"## Documents ({len(n.doc_ids)})", "",
                  "Read one with get_document(doc_id). Rows: `id` - title - summary - [keywords]", ""]
        for i in n.doc_ids:
            c = cards[i]
            kw = ", ".join(c.get("keywords") or [])
            lines.append(f"- `{i}` - {c['title']} - {c['one_line']}" + (f" - [{kw}]" if kw else ""))
        lines.append("")
    return "\n".join(lines)


def write_tree(root: Node, out_dir: Path, docs: list[CatalogDoc], cards: dict[str, dict]) -> Path:
    skills = out_dir / ".claude" / "skills"
    if skills.exists():
        import shutil
        shutil.rmtree(skills)
    skills.mkdir(parents=True)
    _assign_paths(root, "")
    for top in root.children:
        for n in top.walk():
            d = skills / n.path
            d.mkdir(parents=True, exist_ok=True)
            fname = "SKILL.md" if n is top else "INDEX.md"
            (d / fname).write_text(_render(n, cards), encoding="utf-8")

    (out_dir / "documents.json").write_text(
        json.dumps({d.id: d.text for d in docs}, ensure_ascii=False), encoding="utf-8")

    leaf_of = {i: n.path for top in root.children for n in top.walk() if not n.children
               for i in n.doc_ids}
    entities: dict[str, dict] = {}
    for d in docs:
        for term in {d.name, *cards[d.id].get("keywords", [])}:
            rec = entities.setdefault(term, {"count": 0, "skill_paths": [], "doc_ids": []})
            rec["count"] += 1
            if leaf_of[d.id] not in rec["skill_paths"]:
                rec["skill_paths"].append(leaf_of[d.id])
            rec["doc_ids"].append(d.id)
    (out_dir / "entity_index.json").write_text(json.dumps(entities, indent=2, ensure_ascii=False))
    return skills


# --------------------------------------------------------------------------
# saved vectors (reused at generate time to link procedure sections to the catalog)
# --------------------------------------------------------------------------

def save_vectors(out_dir: Path, ids: list[str], matrix: np.ndarray, embedding: dict) -> None:
    np.save(out_dir / "vectors.npy", np.asarray(matrix, dtype=np.float32))
    (out_dir / "vectors.json").write_text(json.dumps({"ids": ids, **embedding_signature(embedding)}))


def load_vectors(root: Path, embedding: dict) -> tuple[list[str], np.ndarray] | None:
    """Saved catalog vectors, if they were made with the same embedding model and settings."""
    meta_path, arr_path = Path(root) / "vectors.json", Path(root) / "vectors.npy"
    if not (meta_path.exists() and arr_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if {k: meta.get(k) for k in embedding_signature(embedding)} != embedding_signature(embedding):
        return None
    return meta["ids"], np.load(arr_path)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_skill_tree(client: LLMClient, docs: list[CatalogDoc], out_dir: Path, *,
                     models: dict, embedding: dict, catalog: dict) -> Path:
    opt = TreeOptions.from_config(catalog)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Cards for {len(docs)} documents ...")
    cards = (make_cards(client, docs, models["catalog_cards"], out_dir / "cards.json", opt.workers,
                        effort=catalog.get("effort", "low"))
             if opt.use_cards else {d.id: plain_card(d) for d in docs})

    print("[2/4] Embedding ...")
    texts = [f"{cards[d.id]['title']}. {cards[d.id]['one_line']} "
             f"{' '.join(cards[d.id]['keywords'])}\n{d.text[:2000]}" for d in docs]
    matrix = embed_texts(client, texts, embedding)
    vecs = dict(zip([d.id for d in docs], matrix))
    save_vectors(out_dir, [d.id for d in docs], matrix, embedding)

    print("[3/4] Partitioning ...")
    root = Node([d.id for d in docs])
    if opt.partition_by_kind:
        for kind, (label, blurb) in KIND_FOLDERS.items():
            ids = [d.id for d in docs if d.kind == kind]
            if ids:
                top = split(ids, vecs, opt)
                top.kind, top.label, top.summary = kind, label, blurb
                root.children.append(top)
    else:
        root.children = split(root.doc_ids, vecs, opt).children or [Node(root.doc_ids)]

    print("[4/4] Describing folders ...")
    describe(client, root, cards, models["catalog_summary"], opt.workers,
             effort=catalog.get("effort", "low"))
    skills = write_tree(root, out_dir, docs, cards)
    meta = {"documents": len(docs), "folders": sum(1 for _ in root.walk()) - 1,
            "options": opt.__dict__, "embedding": embedding}
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Skill tree: {skills} ({meta['folders']} folders)")
    return skills
