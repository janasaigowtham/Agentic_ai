"""Tachyon glue: the only file you need to edit to connect o2a_gen to Tachyon.

Point the config at these functions:

    llm:
      provider: callable
      chat: o2a_gen.providers.tachyon_provider:chat
      embed: o2a_gen.providers.tachyon_provider:embed   # only if embedding.provider = llm

Fill in the three marked blocks with your Tachyon SDK calls. Everything else
in o2a_gen (and the vendored Corpus2Skill) calls through these two functions,
so no other code needs to know about Tachyon.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _client():
    # ---- TODO(tachyon) 1/3: build and return an authenticated Tachyon client.
    # Read keys/endpoints from the environment, never hard-code them, e.g.
    #   import os, tachyon
    #   return tachyon.Client(api_key=os.environ["TACHYON_API_KEY"])
    raise NotImplementedError(
        "Edit o2a_gen/providers/tachyon_provider.py: create the Tachyon client in _client()"
    )


def chat(*, model: str, system: str | None, messages: list[dict],
         max_tokens: int, temperature: float):
    """One chat completion.

    messages: [{"role": "user"|"assistant", "content": str}, ...] (no system role;
              the system prompt is passed separately in ``system``).
    Return the reply text, or {"text": ..., "input_tokens": ..., "output_tokens": ...}.
    """
    client = _client()  # noqa: F841  (used once the TODO below is filled in)
    # ---- TODO(tachyon) 2/3: call Tachyon and return the text. If the SDK wants
    # the system prompt as a message, prepend {"role": "system", "content": system}.
    #   resp = client.chat(model=model, messages=[...], max_tokens=max_tokens,
    #                      temperature=temperature)
    #   return {"text": resp.text, "input_tokens": resp.usage.input,
    #           "output_tokens": resp.usage.output}
    raise NotImplementedError("Edit tachyon_provider.chat() to call Tachyon")


def embed(*, texts: list[str], model: str) -> list[list[float]]:
    """Embed a batch of texts. Only used when ``embedding.provider: llm``."""
    client = _client()  # noqa: F841  (used once the TODO below is filled in)
    # ---- TODO(tachyon) 3/3: return one vector per input text, same order.
    #   return client.embeddings(model=model, input=texts).vectors
    raise NotImplementedError("Edit tachyon_provider.embed() to call Tachyon embeddings")

