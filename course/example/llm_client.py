"""Pluggable model client.

Defaults to a deterministic mock so the whole example runs offline,
reproducibly, with no API key - this is what makes the control-flow tests in
tests/test_pipeline.py possible without mocking a network call. Swap in a
real provider by setting AGENT_COURSE_LLM=anthropic (and ANTHROPIC_API_KEY);
nothing else in the pipeline changes, since agent_framework.py only ever
calls `llm.complete(prompt, session=...)`.
"""
from __future__ import annotations

import os
from typing import Any


class MockLLM:
    """Deterministic stand-in for a model call.

    Keyword-driven on purpose: real classification/drafting is replaced with
    simple substring checks so the example is reproducible without network
    access. Note the priority order below (refund checked before technical)
    is a deliberate ambiguity-resolution policy, not an oversight - see
    course/06-planning-and-reasoning.md.
    """

    def complete(self, prompt: str, session: dict[str, Any] | None = None) -> str:
        text = prompt.lower()

        if "classify" in text:
            if "refund" in text or "charged twice" in text:
                return "refund"
            if "broken" in text or "not working" in text or "error" in text:
                return "technical"
            return "general"

        if "draft a reply" in text:
            session = session or {}
            intent = session.get("intent", "general")
            order = session.get("order") or {}
            if intent == "refund":
                amount = session.get("refund_amount", 0.0)
                return (
                    f"We're sorry about order {order.get('order_id', 'N/A')}. "
                    f"A refund of ${amount:.2f} has been approved and will "
                    "post in 3-5 business days."
                )
            if intent == "technical":
                return (
                    "Thanks for the report - our engineering team is "
                    "investigating and will follow up within 24 hours."
                )
            return "Thanks for reaching out - a support specialist will follow up shortly."

        return "OK"


class AnthropicLLM:
    """Real provider, opt-in only.

    Not exercised by the test suite (needs network + ANTHROPIC_API_KEY).
    Included to show exactly where a real model call plugs in without
    touching agent_framework.py or pipeline.yaml.
    """

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        import anthropic  # imported lazily so the default mock path has no dependency

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str, session: dict[str, Any] | None = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


def build_llm():
    if os.environ.get("AGENT_COURSE_LLM") == "anthropic":
        return AnthropicLLM()
    return MockLLM()
