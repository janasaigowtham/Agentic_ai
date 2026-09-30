"""Anthropic Messages API chat, e.g. claude-opus-5-5.

Config:

    llm:
      provider: callable
      chat: o2a_gen.providers.anthropic_provider:chat

Environment:
    ANTHROPIC_API_KEY      required. Never put the key in config files or the repo.
    ANTHROPIC_BASE_URL     optional, default https://api.anthropic.com
    O2A_ANTHROPIC_EFFORT   optional: low | medium | high (model default when unset)
    O2A_ANTHROPIC_FALLBACK_MODEL
                           model that answers a request the main model declines (a safety
                           classifier can block harmless text, e.g. "ICMP" as a document
                           repository name). Default claude-sonnet-5; set to "" to disable.

Opus 5.5 notes: `temperature` is rejected, so it is never sent. The model always
thinks before answering and thinking counts against `max_tokens`, so a floor is
applied (and lowered to the model's maximum if the API says it is too high).
Responses are streamed, so a long answer (e.g. extracting every step of a large
procedure) never hits a read timeout: the timeout applies to each gap between
streamed events, not to the whole answer. The system prompt and the latest user
turn are marked for prompt caching. Standard library only.
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
         temperature: float = 0.0, retries: int = 5, timeout: float = 300.0) -> dict:
    """One completion. ``temperature`` is accepted for interface compatibility and ignored.
    A declined request is retried once on the fallback model."""
    try:
        return _chat(model, system, messages, max_tokens, retries, timeout)
    except RefusalError:
        fallback = os.environ.get("O2A_ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-5").strip()
        if not fallback or fallback == model:
            raise
        print(f"  ({model} declined a request; answered by {fallback})", flush=True)
        return _chat(fallback, system, messages, max_tokens, retries, timeout)


def _chat(model, system, messages, max_tokens, retries, timeout) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise AnthropicError("ANTHROPIC_API_KEY is not set in the environment")
    msgs = [{"role": m["role"], "content": _blocks(str(m["content"]), i == len(messages) - 1)}
            for i, m in enumerate(messages)]
    body: dict = {"model": model, "max_tokens": max(int(max_tokens), _MIN_MAX_TOKENS),
                  "messages": msgs, "stream": True}
    if system:
        body["system"] = _blocks(system, True)
    effort = os.environ.get("O2A_ANTHROPIC_EFFORT")
    if effort:
        body["output_config"] = {"effort": effort}
    url = os.environ.get("ANTHROPIC_BASE_URL", _DEFAULT_BASE).rstrip("/") + "/v1/messages"

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
                    "output_tokens": r["output_tokens"]}
        if attempt < retries:
            _pause(wait + random.random())
    raise AnthropicError(f"Anthropic call failed after {retries + 1} tries: {last}")


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _read_stream(resp) -> dict:
    """Collect a streamed Messages response (server-sent events)."""
    text: list[str] = []
    kinds: dict[int, str] = {}
    out = {"stop_reason": None, "input_tokens": 0, "output_tokens": 0}
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
