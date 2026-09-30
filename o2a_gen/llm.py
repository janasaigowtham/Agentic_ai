"""LLM access for o2a_gen.

Every model call in o2a_gen, including the ones Corpus2Skill makes while
compiling the catalog, goes through an ``LLMClient``. Plug your company SDK
(Tachyon) in by pointing the config at two plain functions; see
``o2a_gen/providers/tachyon_provider.py``.
"""

from __future__ import annotations

import importlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Completion: ...

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...


def _load_callable(ref: str) -> Callable:
    """Load ``package.module:function``."""
    if ":" not in ref:
        raise ValueError(f"expected 'module:function', got {ref!r}")
    mod_name, fn_name = ref.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name, None)
    if not callable(fn):
        raise ValueError(f"{ref!r} is not a callable")
    return fn


def _to_completion(result: Any) -> Completion:
    if isinstance(result, Completion):
        return result
    if isinstance(result, str):
        return Completion(text=result)
    if isinstance(result, dict):
        return Completion(
            text=str(result.get("text", "")),
            input_tokens=int(result.get("input_tokens", 0) or 0),
            output_tokens=int(result.get("output_tokens", 0) or 0),
        )
    raise TypeError(f"chat function returned {type(result).__name__}; expected str or dict")


class CallableClient:
    """Wraps user-supplied ``chat`` (and optional ``embed``) functions.

    chat(model, system, messages, max_tokens, temperature) -> str | {"text", "input_tokens", "output_tokens"}
    embed(texts, model) -> list[list[float]]

    ``messages`` is a list of {"role": "user"|"assistant", "content": str}.
    """

    def __init__(self, chat: str | Callable, embed: str | Callable | None = None,
                 max_concurrency: int = 8):
        self._chat = _load_callable(chat) if isinstance(chat, str) else chat
        self._embed = _load_callable(embed) if isinstance(embed, str) else embed
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self.usage = UsageMeter()

    def complete(self, *, model, messages, system=None, max_tokens=1024, temperature=0.0):
        with self._sem:
            c = _to_completion(self._chat(
                model=model, system=system, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            ))
        self.usage.add(model, c)
        return c

    def embed(self, texts, *, model):
        if self._embed is None:
            raise RuntimeError(
                "embedding.provider is 'llm' but no llm.embed function is configured"
            )
        with self._sem:
            return self._embed(texts=texts, model=model)


Rule = tuple[Callable[[str, str], bool], Any]


class FakeLLM:
    """Deterministic stand-in for tests and dry runs.

    ``rules`` is a list of (predicate(system, last_user_text), response). A
    response is a string, a dict/list (serialised as JSON), or a callable
    taking (system, messages) and returning one of those. First match wins.
    """

    def __init__(self, rules: list[Rule] | None = None, default: Any = "ok"):
        self.rules = list(rules or [])
        self.default = default
        self.calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        self.usage = UsageMeter()

    def complete(self, *, model, messages, system=None, max_tokens=1024, temperature=0.0):
        last = messages[-1]["content"] if messages else ""
        self.calls.append({"model": model, "system": system, "messages": messages})
        resp = self.default
        for pred, r in self.rules:
            if pred(system or "", last):
                resp = r
                break
        if callable(resp):
            resp = resp(system or "", messages)
        if not isinstance(resp, str):
            resp = json.dumps(resp)
        c = Completion(text=resp)
        self.usage.add(model, c)
        return c

    def embed(self, texts, *, model):
        # Stable bag-of-words hashing vectors so similar texts land together.
        import hashlib
        self.embed_calls.append(list(texts))
        dim = 64
        out = []
        for t in texts:
            v = [0.0] * dim
            for w in re.findall(r"[a-z0-9_]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, c: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += c.input_tokens
            self.output_tokens += c.output_tokens
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "calls_by_model": dict(self.by_model)}


def build_client(llm_cfg: dict) -> LLMClient:
    provider = llm_cfg.get("provider", "callable")
    if provider == "callable":
        if not llm_cfg.get("chat"):
            raise ValueError("llm.chat must name your chat function, e.g. 'my_glue:chat'")
        return CallableClient(llm_cfg["chat"], llm_cfg.get("embed"),
                              int(llm_cfg.get("max_concurrency", 8)))
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"unknown llm.provider {provider!r} (use 'callable' or 'fake')")


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (tolerates code fences and prose)."""
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            end = cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON found in model output")


def complete_json(
    client: LLMClient,
    *,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 4000,
    retries: int = 2,
    validate: Callable[[Any], None] | None = None,
) -> Any:
    """Ask for JSON; on a parse or validation error, show the model the error and retry."""
    messages = [{"role": "user", "content": prompt}]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        c = client.complete(model=model, system=system, messages=messages,
                            max_tokens=max_tokens, temperature=0.0)
        try:
            data = extract_json(c.text)
            if validate:
                validate(data)
            return data
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": c.text},
                {"role": "user", "content": f"That output was invalid: {e}. "
                                            "Reply again with only the corrected JSON."},
            ]
    raise ValueError(f"model did not return valid JSON after {retries + 1} tries: {last_err}")
