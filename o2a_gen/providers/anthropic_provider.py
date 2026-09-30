"""Anthropic Messages API chat, e.g. claude-opus-5-5.

Config:

    llm:
      provider: callable
      chat: o2a_gen.providers.anthropic_provider:chat
      batch: o2a_gen.providers.anthropic_provider:batch   # optional: Message Batches API
      batch_min_requests: 5                               # smaller groups use chat

Environment:
    ANTHROPIC_API_KEY      required. Never put the key in config files or the repo.
    ANTHROPIC_BASE_URL     optional, default https://api.anthropic.com
    O2A_ANTHROPIC_EFFORT   optional: low | medium | high (model default when unset). A call
                           that passes ``effort`` (e.g. compile calls, catalog.effort) wins.
    O2A_ANTHROPIC_BATCH_POLL_SECONDS       optional, default 30
    O2A_ANTHROPIC_BATCH_MAX_WAIT_MINUTES   optional, default 120; then the batch is cancelled
    O2A_ANTHROPIC_FALLBACK_MODEL
                           model that answers a request the main model declines (a safety
                           classifier can block harmless text, e.g. "ICMP" as a document
                           repository name). Default claude-sonnet-5; set to "" to disable.

Opus 5.5 notes: `temperature` is rejected, so it is never sent. The model always
thinks before answering and thinking counts against `max_tokens`, so a floor is
applied (and lowered to the model's maximum if the API says it is too high).
Responses are streamed, so a long answer (e.g. extracting every step of a large
procedure) never hits a read timeout: the timeout applies to each gap between
streamed events, not to the whole answer. The system prompt is marked for prompt
caching; in a conversation (retries, multi-turn) the latest user turn is marked too,
so the history is read from cache on the next turn. A one-shot prompt's own text is
not marked: it is never read back, so caching it would only add the write premium.

``batch`` sends many one-shot requests through the Message Batches API at half the
price (most batches finish within an hour). Results are matched by custom_id. The
compile steps (catalog cards, folder pages, procedure section summaries) use it via
``llm.complete_json_many``; anything a batch doesn't answer cleanly is retried with
``chat``, which also applies the refusal fallback. Standard library only.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request

_DEFAULT_BASE = "https://api.anthropic.com"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_MIN_MAX_TOKENS = 16000


class AnthropicError(RuntimeError):
    pass


class RefusalError(AnthropicError, ValueError):
    """The model declined. A ValueError, so callers that fall back on a bad answer
    (summary cards, folder and section summaries) fall back here too."""


class TruncatedError(AnthropicError, ValueError):
    """The answer stopped at max_tokens, so it is incomplete."""


class _Retry(Exception):
    pass


def _blocks(text: str, cache: bool) -> list[dict]:
    block: dict = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def chat(*, model: str, system: str | None, messages: list[dict], max_tokens: int,
         temperature: float = 0.0, effort: str | None = None, retries: int = 5,
         timeout: float = 300.0) -> dict:
    """One completion. ``temperature`` is accepted for interface compatibility and ignored.
    A declined request is retried once on the fallback model."""
    try:
        return _chat(model, system, messages, max_tokens, effort, retries, timeout)
    except RefusalError:
        fallback = os.environ.get("O2A_ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-5").strip()
        if not fallback or fallback == model:
            raise
        print(f"  ({model} declined a request; answered by {fallback})", flush=True)
        return _chat(fallback, system, messages, max_tokens, effort, retries, timeout)


def _key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise AnthropicError("ANTHROPIC_API_KEY is not set in the environment")
    return key


def _base() -> str:
    return os.environ.get("ANTHROPIC_BASE_URL", _DEFAULT_BASE).rstrip("/")


def _body(model, system, messages, max_tokens, effort) -> dict:
    """Request body shared by chat and batch: caching marks, thinking floor, effort."""
    multi = len(messages) > 1
    msgs = [{"role": m["role"],
             "content": _blocks(str(m["content"]), multi and i == len(messages) - 1)}
            for i, m in enumerate(messages)]
    body: dict = {"model": model, "max_tokens": max(int(max_tokens), _MIN_MAX_TOKENS),
                  "messages": msgs}
    if system:
        body["system"] = _blocks(system, True)
    effort = effort or os.environ.get("O2A_ANTHROPIC_EFFORT")
    if effort:
        body["output_config"] = {"effort": effort}
    return body


def _chat(model, system, messages, max_tokens, effort, retries, timeout) -> dict:
    key = _key()
    body = _body(model, system, messages, max_tokens, effort)
    body["stream"] = True
    url = _base() + "/v1/messages"

    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "content-type": "application/json", "accept": "text/event-stream"})
        wait = min(60.0, 2 ** attempt)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = _read_stream(resp)
        except urllib.error.HTTPError as e:
            detail = e.read()[:1000].decode("utf-8", "replace")
            limit = _max_tokens_limit(detail, body["max_tokens"]) if e.code == 400 else None
            if limit:
                body["max_tokens"] = limit       # the model allows less; ask again with that
                last = f"max_tokens lowered to {limit}"
                continue
            if e.code not in _RETRY_STATUS:
                raise AnthropicError(f"Anthropic HTTP {e.code} for {model!r}: {detail}") from None
            last = f"HTTP {e.code}: {detail}"
            wait = float(e.headers.get("retry-after") or 0) or wait
        except (_Retry, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                http.client.HTTPException, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r["stop_reason"] == "refusal":
                raise RefusalError(f"{model} declined this request")
            if r["stop_reason"] == "max_tokens":
                raise TruncatedError(f"{model} answer was cut off at max_tokens="
                                     f"{body['max_tokens']}")
            return {"text": r["text"], "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "cache_read_tokens": r["cache_read_tokens"],
                    "cache_write_tokens": r["cache_write_tokens"]}
        if attempt < retries:
            _pause(wait + random.random())
    raise AnthropicError(f"Anthropic call failed after {retries + 1} tries: {last}")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _read_stream(resp) -> dict:
    """Collect a streamed Messages response (server-sent events)."""
    text: list[str] = []
    kinds: dict[int, str] = {}
    out = {"stop_reason": None, "input_tokens": 0, "output_tokens": 0,
           "cache_read_tokens": 0, "cache_write_tokens": 0}
    done = False
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:].strip() or "{}")
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            out["input_tokens"] = sum(int(u.get(k) or 0) for k in (
                "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            out["output_tokens"] = int(u.get("output_tokens") or 0)
            out["cache_read_tokens"] = int(u.get("cache_read_input_tokens") or 0)
            out["cache_write_tokens"] = int(u.get("cache_creation_input_tokens") or 0)
        elif t == "content_block_start":
            kinds[ev.get("index", 0)] = (ev.get("content_block") or {}).get("type", "")
            if kinds[ev.get("index", 0)] == "text":
                text.append((ev.get("content_block") or {}).get("text", ""))
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta":
                text.append(d.get("text", ""))
        elif t == "message_delta":
            out["stop_reason"] = (ev.get("delta") or {}).get("stop_reason") or out["stop_reason"]
            u = ev.get("usage") or {}
            if u.get("output_tokens") is not None:
                out["output_tokens"] = int(u["output_tokens"])
        elif t == "message_stop":
            done = True
            break
        elif t == "error":
            err = ev.get("error") or {}
            raise _Retry(f"stream error {err.get('type')}: {err.get('message')}")
    if not done:
        raise _Retry("stream ended before message_stop")
    out["text"] = "".join(text)
    return out


def _max_tokens_limit(detail: str, asked: int) -> int | None:
    """If a 400 says max_tokens is too high, return the limit it names, or else half of
    what was asked (never below the thinking floor)."""
    if "max_tokens" not in detail:
        return None
    nums = [int(n) for n in re.findall(r"\d{3,}", detail) if 1000 <= int(n) < asked]
    if nums:
        return max(nums)
    return asked // 2 if asked // 2 >= _MIN_MAX_TOKENS else None


# ---------------------------------------------------------------------------
# Message Batches API
# ---------------------------------------------------------------------------

def batch(requests: list[dict]) -> list[dict | Exception]:
    """Run chat-style requests (model, system, messages, max_tokens, effort) as one
    batch. Returns, in input order, a chat-style result dict (with ``stop_reason``)
    or an exception for each request."""
    if not requests:
        return []
    poll = float(os.environ.get("O2A_ANTHROPIC_BATCH_POLL_SECONDS", "30"))
    max_wait = float(os.environ.get("O2A_ANTHROPIC_BATCH_MAX_WAIT_MINUTES", "120")) * 60
    items = [{"custom_id": f"r{i}",
              "params": _body(r["model"], r.get("system"), r["messages"],
                              r.get("max_tokens", 1024), r.get("effort"))}
             for i, r in enumerate(requests)]
    b = _api("POST", "/v1/messages/batches", {"requests": items})
    print(f"      batch {b['id']}: {len(items)} requests submitted", flush=True)
    started = time.monotonic()
    while b.get("processing_status") != "ended":
        if time.monotonic() - started > max_wait:
            _api("POST", f"/v1/messages/batches/{b['id']}/cancel", {})
            raise AnthropicError(f"batch {b['id']} not finished after {max_wait / 60:.0f} "
                                 "minutes; cancelled")
        _pause(poll)
        b = _api("GET", f"/v1/messages/batches/{b['id']}")

    results: dict[str, dict | Exception] = {}
    for line in _api("GET", b["results_url"], raw=True).splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        res = item.get("result") or {}
        if res.get("type") == "succeeded":
            msg = res.get("message") or {}
            u = msg.get("usage") or {}
            results[item["custom_id"]] = {
                "text": "".join(c.get("text", "") for c in msg.get("content") or []
                                if c.get("type") == "text"),
                "input_tokens": sum(int(u.get(k) or 0) for k in (
                    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                "output_tokens": int(u.get("output_tokens") or 0),
                "cache_read_tokens": int(u.get("cache_read_input_tokens") or 0),
                "cache_write_tokens": int(u.get("cache_creation_input_tokens") or 0),
                "stop_reason": msg.get("stop_reason") or ""}
        else:
            results[item["custom_id"]] = AnthropicError(
                f"batch item {res.get('type')}: {json.dumps(res.get('error'))[:300]}")
    return [results.get(f"r{i}", AnthropicError("no result in batch"))
            for i in range(len(requests))]


def _api(method: str, path: str, payload: dict | None = None, *, raw: bool = False,
         retries: int = 5, timeout: float = 120.0):
    """A plain (non-streamed) API call with retries. ``path`` may be a full URL."""
    url = path if path.startswith("http") else _base() + path
    data = json.dumps(payload).encode() if payload is not None else None
    last = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "x-api-key": _key(), "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        wait = min(60.0, 2 ** attempt)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
            return body if raw else json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read()[:1000].decode("utf-8", "replace")
            if e.code not in _RETRY_STATUS:
                raise AnthropicError(f"Anthropic HTTP {e.code} on {path}: {detail}") from None
            last = f"HTTP {e.code}: {detail}"
            wait = float(e.headers.get("retry-after") or 0) or wait
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                http.client.HTTPException, OSError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            _pause(wait + random.random())
    raise AnthropicError(f"Anthropic {method} {path} failed after {retries + 1} tries: {last}")
