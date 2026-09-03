"""
Interactive CLI for the agent harness.

Run:
    python -m agent_harness
    python -m agent_harness --model claude-opus-5 --no-stream
    python -m agent_harness --confirm   # ask before each tool execution
"""

import argparse
import sys

from agent_harness.harness import AgentHarness, HarnessConfig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive agent harness powered by the Claude API"
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-5-20250514",
        help="Claude model ID (default: claude-sonnet-4-5-20250514)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16_000,
        help="Max output tokens per API call (default: 16000)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=25,
        help="Max tool-use turns before stopping (default: 25)",
    )
    parser.add_argument(
        "--system", default=None,
        help="Custom system prompt (overrides the default)",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable streaming (wait for full response)",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Ask for confirmation before each tool execution",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show debug info (retries, tool execution, etc.)",
    )
    parser.add_argument(
        "--no-builtins", action="store_true",
        help="Don't register any built-in tools",
    )
    args = parser.parse_args()

    config = HarnessConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        max_tool_turns=args.max_turns,
        stream=not args.no_stream,
        confirm_before_execute=args.confirm,
        verbose=args.verbose,
    )
    if args.system:
        config.system_prompt = args.system

    harness = AgentHarness(config=config, include_builtins=not args.no_builtins)

    print("Agent Harness — Claude API")
    print(f"Model: {config.model}")
    print(f"Tools: {', '.join(t['name'] for t in harness.registry.definitions) or 'none'}")
    print(f"Streaming: {'on' if config.stream else 'off'}")
    print("Type 'quit' to exit, 'reset' to clear history, 'stats' for usage.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            harness.reset()
            print("[Conversation reset]\n")
            continue
        if user_input.lower() == "stats":
            print(f"[{harness.stats.summary()}]\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            response = harness.chat(user_input)
            if not config.stream:
                print(response)
            print(f"\n[{harness.stats.summary()}]\n")
        except Exception as exc:
            print(f"\nError: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
