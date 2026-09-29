"""Runs the vendored Corpus2Skill compiler with o2a_gen's LLM client and embedder.

Corpus2Skill calls ``anthropic.Anthropic().messages.create`` directly. Rather
than edit the vendored code, we swap its module-level clients for shims that
forward to our ``LLMClient`` (and therefore to Tachyon), for the duration of
a compile only.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from o2a_gen.llm import LLMClient

_VENDORED = Path(__file__).resolve().parent.parent / "third_party" / "corpus2skill"


def _stub_anthropic_if_missing() -> None:
    """Corpus2Skill imports ``anthropic`` at module load, but we replace its clients,
    so the package is not needed. Install a stub that fails loudly if ever used."""
    try:
        import anthropic  # noqa: F401
        return
    except ImportError:
        pass
    import types

    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError("anthropic SDK is not used by o2a_gen; calls go through LLMClient")

    stub = types.ModuleType("anthropic")
    stub.Anthropic = stub.AsyncAnthropic = _Unavailable
    sys.modules["anthropic"] = stub


def import_corpus2skill():
    """Import corpus2skill, falling back to the vendored copy if it isn't installed."""
    _stub_anthropic_if_missing()
    try:
        import corpus2skill  # noqa: F401
    except ImportError:
        if not _VENDORED.is_dir():
            raise ImportError(
                "corpus2skill not found: pip install -e third_party/corpus2skill"
            ) from None
        sys.path.insert(0, str(_VENDORED))
        import corpus2skill  # noqa: F401
    from corpus2skill import compile as c2s_compile, config as c2s_config, summarizer
    return c2s_compile, c2s_config, summarizer


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        else:
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts)


def _system_text(system) -> str | None:
    if system is None:
        return None
    return _flatten(system) if not isinstance(system, str) else system


class _Messages:
    def __init__(self, client: LLMClient):
        self._client = client

    def create(self, *, model, max_tokens, messages, system=None, temperature=0.0, **_ignored):
        c = self._client.complete(
            model=model,
            system=_system_text(system),
            messages=[{"role": m["role"], "content": _flatten(m["content"])} for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=c.text)],
            usage=SimpleNamespace(input_tokens=c.input_tokens, output_tokens=c.output_tokens),
            stop_reason="end_turn",
        )


class _AsyncMessages:
    def __init__(self, client: LLMClient):
        self._sync = _Messages(client)

    async def create(self, **kwargs):
        return await asyncio.to_thread(self._sync.create, **kwargs)


class AnthropicShim:
    """Looks enough like ``anthropic.Anthropic`` for Corpus2Skill's summarizer."""

    def __init__(self, client: LLMClient):
        self.messages = _Messages(client)


class AsyncAnthropicShim:
    def __init__(self, client: LLMClient):
        self.messages = _AsyncMessages(client)


@contextmanager
def patched_corpus2skill(client: LLMClient, embedding: dict):
    c2s_compile, _, summarizer = import_corpus2skill()
    saved = (summarizer._client, summarizer._async_client,
             c2s_compile.embed_documents, c2s_compile.embed_single)
    summarizer._client = AnthropicShim(client)
    summarizer._async_client = AsyncAnthropicShim(client)
    if embedding.get("provider", "local") == "llm":
        c2s_compile.embed_documents, c2s_compile.embed_single = _llm_embedders(client)
    elif embedding.get("provider") != "local":
        raise ValueError(f"unknown embedding.provider {embedding.get('provider')!r}")
    try:
        yield c2s_compile
    finally:
        (summarizer._client, summarizer._async_client,
         c2s_compile.embed_documents, c2s_compile.embed_single) = saved


def _llm_embedders(client: LLMClient):
    import numpy as np

    def embed_documents(texts, model_name, batch_size=32, max_chars=12000):
        vecs: list[list[float]] = []
        step = max(1, batch_size)
        for i in range(0, len(texts), step):
            vecs.extend(client.embed([t[:max_chars] for t in texts[i:i + step]], model=model_name))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def embed_single(text, model_name, context=None, max_chars=12000):
        full = text if not context else f"{text}\n---\n{context}"
        return embed_documents([full], model_name, 1, max_chars)[0]

    return embed_documents, embed_single


def compile_with_corpus2skill(client: LLMClient, *, corpus_dir: Path, output_dir: Path,
                              models: dict, embedding: dict, catalog: dict) -> Path:
    """Compile ``corpus_dir`` into ``output_dir/.claude/skills``; returns the skills dir."""
    with patched_corpus2skill(client, embedding) as c2s_compile:
        from corpus2skill.config import CompileConfig
        cfg = CompileConfig(
            input_dir=corpus_dir,
            output_dir=output_dir,
            p=int(catalog.get("p", 10)),
            max_top_clusters=int(catalog.get("max_top", 8)),
            min_cluster_size=int(catalog.get("min_cluster_size", 3)),
            embed_model=embedding.get("model", "Qwen/Qwen3-Embedding-0.6B"),
            llm_model=models["catalog_summary"],
            doc_summary_model=models["catalog_cards"],
            repartition_model=models["catalog_summary"],
            entity_model=models["catalog_summary"],
            use_doc_summaries=bool(catalog.get("use_doc_summaries", True)),
            compact=bool(catalog.get("compact", False)),
        )
        c2s_compile.compile_corpus(cfg)
    skills_dir = output_dir / ".claude" / "skills"
    if not skills_dir.is_dir():
        raise RuntimeError(f"Corpus2Skill produced no skill tree in {skills_dir}")
    return skills_dir
