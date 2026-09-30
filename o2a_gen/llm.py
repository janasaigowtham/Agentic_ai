"""LLM access for o2a_gen.

Every model call in o2a_gen, including the ones Corpus2Skill makes while
compiling the catalog, goes through an ``LLMClient``. Plug your company SDK
(Tachyon) in by pointing the config at two plain functions; see
``o2a_gen/providers/tachyon_provider.py``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""
    batch: bool = False


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        effort: str | None = None,
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


def _accepts(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


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
            cache_read_tokens=int(result.get("cache_read_tokens", 0) or 0),
            cache_write_tokens=int(result.get("cache_write_tokens", 0) or 0),
            stop_reason=str(result.get("stop_reason", "") or ""),
        )
    raise TypeError(f"chat function returned {type(result).__name__}; expected str or dict")


class CallableClient:
    """Wraps user-supplied ``chat`` (and optional ``embed``) functions.

    chat(model, system, messages, max_tokens, temperature[, effort]) -> str | {"text", ...}
    embed(texts, model) -> list[list[float]]
    batch(requests) -> list[dict | Exception]     optional, e.g. the Anthropic Batch API

    ``messages`` is a list of {"role": "user"|"assistant", "content": str}. ``effort`` is
    passed only to chat functions that accept it. ``batch`` gets a list of chat keyword
    dicts and returns one result (as chat would) or exception per request, in order.
    """

    def __init__(self, chat: str | Callable, embed: str | Callable | None = None,
                 max_concurrency: int = 8, batch: str | Callable | None = None,
                 batch_min_requests: int = 5):
        self._chat = _load_callable(chat) if isinstance(chat, str) else chat
        self._embed = _load_callable(embed) if isinstance(embed, str) else embed
        self._batch = _load_callable(batch) if isinstance(batch, str) else batch
        self._chat_takes_effort = _accepts(self._chat, "effort")
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self.batch_enabled = self._batch is not None
        self.batch_min = max(1, int(batch_min_requests))
        self.usage = UsageMeter()

    def complete(self, *, model, messages, system=None, max_tokens=1024, temperature=0.0,
                 effort=None):
        extra = {"effort": effort} if effort and self._chat_takes_effort else {}
        with self._sem:
            c = _to_completion(self._chat(
                model=model, system=system, messages=messages,
                max_tokens=max_tokens, temperature=temperature, **extra,
            ))
        self.usage.add(model, c)
        return c

    def complete_many(self, requests: list[dict]) -> list[Completion | Exception]:
        """Send requests (``complete`` keywords) through the batch function."""
        out: list[Completion | Exception] = []
        for req, res in zip(requests, self._batch(requests)):
            if isinstance(res, Exception):
                out.append(res)
                continue
            c = _to_completion(res)
            c.batch = True
            self.usage.add(req["model"], c)
            out.append(c)
        return out

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

    def complete(self, *, model, messages, system=None, max_tokens=1024, temperature=0.0,
                 effort=None):
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
    batch_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, c: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.batch_calls += int(c.batch)
            self.input_tokens += c.input_tokens
            self.output_tokens += c.output_tokens
            self.cache_read_tokens += c.cache_read_tokens
            self.cache_write_tokens += c.cache_write_tokens
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def as_dict(self) -> dict:
        return {"calls": self.calls, "batch_calls": self.batch_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "calls_by_model": dict(self.by_model)}


def build_client(llm_cfg: dict) -> LLMClient:
    provider = llm_cfg.get("provider", "callable")
    if provider == "callable":
        if not llm_cfg.get("chat"):
            raise ValueError("llm.chat must name your chat function, e.g. 'my_glue:chat'")
        return CallableClient(llm_cfg["chat"], llm_cfg.get("embed"),
                              int(llm_cfg.get("max_concurrency", 8)),
                              batch=llm_cfg.get("batch"),
                              batch_min_requests=int(llm_cfg.get("batch_min_requests", 5)))
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
    effort: str | None = None,
) -> Any:
    """Ask for JSON; on a parse or validation error, show the model the error and retry."""
    messages = [{"role": "user", "content": prompt}]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        c = client.complete(model=model, system=system, messages=messages,
                            max_tokens=max_tokens, **({"effort": effort} if effort else {}))
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


def complete_json_many(
    client: LLMClient,
    jobs: list[tuple[str, str, str]],
    *,
    model: str,
    max_tokens: int = 4000,
    effort: str | None = None,
    workers: int = 8,
    validate: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Run many independent one-shot JSON prompts. ``jobs`` is (key, system, prompt).

    If the client can batch (``llm.batch``, e.g. the Anthropic Batch API) and there are
    enough jobs, they go in one batch. Any that error, are declined, or don't parse or
    validate are retried as normal calls (where the provider's own retries and refusal
    fallback apply). Returns key -> parsed JSON; keys that still fail are left out, so
    callers keep their fallback.
    """
    out: dict[str, Any] = {}
    if (getattr(client, "batch_enabled", False) and hasattr(client, "complete_many")
            and len(jobs) >= getattr(client, "batch_min", 1)):
        reqs = [{"model": model, "system": system, "max_tokens": max_tokens, "effort": effort,
                 "messages": [{"role": "user", "content": prompt}]} for _, system, prompt in jobs]
        try:
            results = client.complete_many(reqs)
        except Exception as e:   # batch endpoint unavailable, timed out, ...: use normal calls
            print(f"      batch failed ({type(e).__name__}: {e}); sending "
                  f"{len(jobs)} requests as normal calls", flush=True)
            results = [e] * len(jobs)
        retry = []
        for job, res in zip(jobs, results):
            if isinstance(res, Completion) and res.stop_reason not in ("refusal", "max_tokens"):
                try:
                    data = extract_json(res.text)
                    if validate:
                        validate(data)
                    out[job[0]] = data
                    continue
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
            retry.append(job)
        jobs = retry

    def one(job):
        key, system, prompt = job
        try:
            return key, complete_json(client, model=model, system=system, prompt=prompt,
                                      max_tokens=max_tokens, validate=validate, effort=effort)
        except ValueError:
            return key, None

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for key, data in pool.map(one, jobs):
                if data is not None:
                    out[key] = data
    return out
