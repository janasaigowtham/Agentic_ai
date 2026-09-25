"""JudgeClient abstraction: MockJudgeClient (default) plus the two live clients.

Every stage and specialist calls a JudgeClient the same way regardless of mode --
mode only changes which concrete client `build_judge_clients` wires up. Live SDK
imports are deferred into each client's first real call so importing this module
(and running in mock mode) never requires google-genai or anthropic to be
installed.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from trh.config import HarnessConfig, JudgeMode, ModelLineage
from trh.judge.mock_fixtures import MOCK_ANSWER_KEY, MOCK_DEFAULT_BY_STAGE


class JudgeError(RuntimeError):
    """Raised when a live judge call fails or returns unparseable output."""


class JudgeClient(ABC):
    lineage: ModelLineage

    @abstractmethod
    def complete(
        self, *, stage: str, system_prompt: str, user_prompt: str, marker: str
    ) -> dict[str, Any]:
        """Runs one judge call and returns its parsed JSON response.

        `marker` identifies which call this is (see trh.judge.markers) and is
        the only thing MockJudgeClient looks at.
        """


class MockJudgeClient(JudgeClient):
    lineage = ModelLineage.MOCK

    def complete(
        self, *, stage: str, system_prompt: str, user_prompt: str, marker: str
    ) -> dict[str, Any]:
        answer = MOCK_ANSWER_KEY.get(marker) or MOCK_DEFAULT_BY_STAGE[stage]
        return json.loads(json.dumps(answer))  # defensive copy, callers may mutate


class GeminiJudgeClient(JudgeClient):
    lineage = ModelLineage.GEMINI

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self, *, stage: str, system_prompt: str, user_prompt: str, marker: str
    ) -> dict[str, Any]:
        client = self._ensure_client()
        response = client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
            },
        )
        text = response.text
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            raise JudgeError(
                f"Gemini returned non-JSON output for stage {stage!r} ({marker}): {text!r}"
            ) from exc


class ClaudeJudgeClient(JudgeClient):
    lineage = ModelLineage.CLAUDE

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self, *, stage: str, system_prompt: str, user_prompt: str, marker: str
    ) -> dict[str, Any]:
        client = self._ensure_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            raise JudgeError(
                f"Claude returned non-JSON output for stage {stage!r} ({marker}): {text!r}"
            ) from exc


def build_judge_clients(config: HarnessConfig) -> dict[ModelLineage, JudgeClient]:
    """Builds the lineage -> client map this run will use."""
    if config.mode is JudgeMode.MOCK:
        mock = MockJudgeClient()
        return {ModelLineage.MOCK: mock, ModelLineage.GEMINI: mock, ModelLineage.CLAUDE: mock}
    return {
        ModelLineage.GEMINI: GeminiJudgeClient(config.gemini_api_key, config.gemini_model),
        ModelLineage.CLAUDE: ClaudeJudgeClient(config.anthropic_api_key, config.claude_model),
    }


def client_for_stage(
    clients: dict[ModelLineage, JudgeClient], config: HarnessConfig, stage: str
) -> JudgeClient:
    return clients[config.lineage_for(stage)]
