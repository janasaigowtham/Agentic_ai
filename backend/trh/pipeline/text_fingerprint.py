"""Plain-code text fingerprinting.

Used so a single-step specialist can flag that its prompt shares a long
sentence with another step's prompt, without itself doing any cross-step
reasoning -- that stays reserved for the Aggregator (spec section 5.3). This
module only ever hashes text and compares hashes; it forms no opinion about
whether an overlap is a problem.
"""
from __future__ import annotations

import hashlib
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_fingerprints(text: str, min_len: int = 40) -> list[str]:
    if not text:
        return []
    fingerprints: list[str] = []
    seen: set[str] = set()
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        cleaned = sentence.strip()
        if len(cleaned) < min_len:
            continue
        digest = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
        if digest not in seen:
            seen.add(digest)
            fingerprints.append(digest)
    return fingerprints
