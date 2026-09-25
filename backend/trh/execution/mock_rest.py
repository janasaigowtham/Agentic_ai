"""Generic mock REST client for rest_api_agent nodes.

No rest_api_agent appears in the bundled fixture pipeline, but the executor
supports the class generically (it's one of the 9 specialist-taxonomy
agent classes) so a future fixture that declares one can still execute
without a real backend.
"""
from __future__ import annotations

from typing import Any


class MockRestClient:
    def call(self, node_name: str, input_keys: list[str], session: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "source": node_name,
            "echo": {k: session.get(k) for k in input_keys},
        }
