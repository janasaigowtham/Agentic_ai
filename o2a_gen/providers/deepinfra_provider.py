"""DeepInfra embeddings (OpenAI-compatible API), e.g. nvidia/Nemotron-3-Embed-8B.

Config:

    llm:
      embed: o2a_gen.providers.deepinfra_provider:embed
    embedding:
      provider: llm
      model: nvidia/Nemotron-3-Embed-8B-BF16    # the model ID as DeepInfra lists it

Environment:
    DEEPINFRA_API_KEY    required. Never put the key in config files or the repo.
    DEEPINFRA_BASE_URL   optional, default https://api.deepinfra.com/v1/openai

Uses only the standard library, so no extra dependencies. HTTPS_PROXY is honoured.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_DEFAULT_BASE = "https://api.deepinfra.com/v1/openai"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    pass


def embed(*, texts: list[str], model: str, retries: int = 4, timeout: float = 120.0
          ) -> list[list[float]]:
    """One vector per input text, in input order."""
    if not texts:
        return []
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise EmbeddingError("DEEPINFRA_API_KEY is not set")
    url = os.environ.get("DEEPINFRA_BASE_URL", _DEFAULT_BASE).rstrip("/") + "/embeddings"
    body = json.dumps({"model": model, "input": list(texts), "encoding_format": "float"}).encode()

    last: str = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            rows = sorted(data["data"], key=lambda d: d["index"])
            if len(rows) != len(texts):
                raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(rows)}")
            return [r["embedding"] for r in rows]
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode("utf-8", "replace")
            if e.code not in _RETRY_STATUS:
                raise EmbeddingError(f"DeepInfra HTTP {e.code} for model {model!r}: {detail}") from None
            last = f"HTTP {e.code}: {detail}"
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if "403" in reason or "407" in reason:  # blocked by a network proxy: retrying won't help
                raise EmbeddingError(f"network policy blocked {url}: {reason}") from None
            last = reason
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20))
    raise EmbeddingError(f"DeepInfra embeddings failed after {retries + 1} tries: {last}")
