"""
Core agent harness — the agentic loop, streaming, error handling,
and conversation management.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from agent_harness.tools import BUILTIN_TOOLS, ToolRegistry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
DEFAULT_MAX_TOKENS = 16_000
MAX_TOOL_TURNS = 25


@dataclass
class HarnessConfig:
    """All tunables for the agent harness."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_tool_turns: int = MAX_TOOL_TURNS
    system_prompt: str = (
        "You are a helpful AI assistant with access to tools. "
        "Use them when they would help answer the user's question. "
        "Think step by step and be precise."
    )
    stream: bool = True
    show_thinking: bool = True
    confirm_before_execute: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class UsageStats:
    """Accumulates token usage and cost across an entire run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    _model: str = ""

    def record(self, usage, model: str = ""):
        self.input_tokens += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.api_calls += 1
        if model:
            self._model = model

    @property
    def estimated_cost(self) -> float:
        rates = {
            "claude-sonnet-4-5-20250514": (3.0, 15.0),
            "claude-opus-5": (5.0, 25.0),
            "claude-sonnet-5": (2.0, 10.0),
            "claude-haiku-4-5": (1.0, 5.0),
        }
        in_rate, out_rate = rates.get(self._model, (3.0, 15.0))
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000

    def summary(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
            f"API calls: {self.api_calls} | Tool calls: {self.tool_calls} | "
            f"Est. cost: ${self.estimated_cost:.4f}"
        )


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


class AgentHarness:
    """Run a multi-turn agentic loop with tools, streaming, and retries.

    Typical usage:

        harness = AgentHarness(tools=[my_tool_func])
        answer  = harness.run("What files are in the current directory?")
        print(answer)
    """

    def __init__(
        self,
        *,
        tools: list[Callable] | None = None,
        config: HarnessConfig | None = None,
        api_key: str | None = None,
        include_builtins: bool = True,
    ):
        self.config = config or HarnessConfig()
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.registry = ToolRegistry()
        self.conversation: list[dict] = []
        self.stats = UsageStats()
        self._on_tool_call: Callable | None = None

        if include_builtins:
            for t in BUILTIN_TOOLS:
                self.registry.register(t, t._tool_schema)

        if tools:
            for t in tools:
                schema = getattr(t, "_tool_schema", None)
                self.registry.register(t, schema)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Send a message and drive the agentic loop to completion.

        Returns the final text response from the assistant.
        """
        self.conversation.append({"role": "user", "content": user_message})
        self.stats = UsageStats()

        for turn in range(self.config.max_tool_turns):
            response = self._call_api()

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            if response.stop_reason == "refusal":
                return "[Agent refused to respond — safety filter triggered]"

            if response.stop_reason == "max_tokens":
                self._log("Warning: hit max_tokens — response may be truncated")
                return self._extract_text(response)

            if response.stop_reason == "pause_turn":
                self.conversation.append(
                    {"role": "assistant", "content": response.content}
                )
                continue

            if response.stop_reason == "tool_use":
                self._handle_tool_calls(response)
                continue

            return self._extract_text(response)

        self._log(f"Stopping after {self.config.max_tool_turns} tool turns")
        return self._extract_text(response)

    def chat(self, user_message: str) -> str:
        """Convenience for multi-turn chat — preserves conversation history."""
        return self.run(user_message)

    def reset(self):
        """Clear conversation history and stats."""
        self.conversation.clear()
        self.stats = UsageStats()

    def on_tool_call(self, callback: Callable):
        """Register a callback invoked before each tool execution.

        The callback receives (tool_name, arguments) and should return
        True to allow execution or False to block it.
        """
        self._on_tool_call = callback

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _call_api(self) -> anthropic.types.Message:
        """Make a single API call with retry logic."""
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
            "messages": self.conversation,
        }

        tool_defs = self.registry.definitions
        if tool_defs:
            kwargs["tools"] = tool_defs

        if self.config.stream:
            return self._call_streaming(**kwargs)
        return self._call_with_retry(**kwargs)

    def _call_with_retry(self, max_retries: int = 3, **kwargs) -> anthropic.types.Message:
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(**kwargs)
                self.stats.record(response.usage, self.config.model)
                return response
            except anthropic.RateLimitError as exc:
                last_exc = exc
                wait = min(2 ** attempt + 1, 30)
                self._log(f"Rate limited — retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    wait = min(2 ** attempt + 1, 15)
                    self._log(f"Server error {exc.status_code} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                wait = min(2 ** attempt + 1, 10)
                self._log(f"Connection error — retrying in {wait}s")
                time.sleep(wait)
        raise last_exc

    def _call_streaming(self, **kwargs) -> anthropic.types.Message:
        """Stream the response, printing text deltas in real time."""
        last_exc = None
        for attempt in range(3):
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    current_block_type = None

                    for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            current_block_type = block.type
                            if block.type == "text":
                                pass
                            elif block.type == "thinking" and self.config.show_thinking:
                                self._print("\n[thinking] ", style="dim")
                            elif block.type == "tool_use":
                                self._print(f"\n[calling tool: {block.name}] ", style="tool")

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                self._print(delta.text)
                            elif delta.type == "thinking_delta" and self.config.show_thinking:
                                self._print(delta.thinking, style="dim")

                        elif event.type == "content_block_stop":
                            if current_block_type == "thinking" and self.config.show_thinking:
                                self._print("\n")
                            current_block_type = None

                    response = stream.get_final_message()

                self.stats.record(response.usage, self.config.model)
                return response

            except anthropic.RateLimitError as exc:
                last_exc = exc
                wait = min(2 ** attempt + 1, 30)
                self._log(f"Rate limited during stream — retrying in {wait}s")
                time.sleep(wait)
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                wait = min(2 ** attempt + 1, 10)
                self._log(f"Connection error during stream — retrying in {wait}s")
                time.sleep(wait)

        raise last_exc

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _handle_tool_calls(self, response: anthropic.types.Message):
        """Extract tool_use blocks, execute them, and append results."""
        self.conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            self.stats.tool_calls += 1

            if self._on_tool_call and not self._on_tool_call(block.name, block.input):
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Tool execution was denied by the user.",
                    "is_error": True,
                })
                continue

            if self.config.confirm_before_execute:
                if not self._confirm_tool(block.name, block.input):
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "User declined to execute this tool.",
                        "is_error": True,
                    })
                    continue

            self._log(f"Executing: {block.name}({json.dumps(block.input, default=str)[:200]})")
            result = self.registry.execute(block.name, block.input)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        self.conversation.append({"role": "user", "content": tool_results})

    @staticmethod
    def _confirm_tool(name: str, arguments: dict) -> bool:
        print(f"\n--- Tool call: {name} ---")
        print(json.dumps(arguments, indent=2, default=str)[:500])
        answer = input("Allow? [Y/n]: ").strip().lower()
        return answer in ("", "y", "yes")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _log(self, msg: str):
        if self.config.verbose:
            print(f"  [{msg}]", file=sys.stderr)

    @staticmethod
    def _print(text: str, style: str = ""):
        sys.stdout.write(text)
        sys.stdout.flush()
