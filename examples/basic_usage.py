#!/usr/bin/env python3
"""
Example: using the AgentHarness programmatically.

Set ANTHROPIC_API_KEY in your environment before running.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/basic_usage.py
"""

import json
from agent_harness import AgentHarness, tool
from agent_harness.harness import HarnessConfig


# ---------------------------------------------------------------
# 1. Define a custom tool with the @tool decorator
# ---------------------------------------------------------------

@tool
def search_contacts(query: str) -> str:
    """Search the company contact database.

    Args:
        query: Name or department to search for.
    """
    contacts = {
        "alice": {"name": "Alice Chen", "email": "alice@example.com", "dept": "Engineering"},
        "bob": {"name": "Bob Park", "email": "bob@example.com", "dept": "Sales"},
        "carol": {"name": "Carol Lima", "email": "carol@example.com", "dept": "Engineering"},
    }
    results = [v for k, v in contacts.items() if query.lower() in k or query.lower() in v["dept"].lower()]
    return json.dumps(results) if results else json.dumps({"message": "No contacts found"})


# ---------------------------------------------------------------
# 2. Basic single-shot usage
# ---------------------------------------------------------------

def example_single_shot():
    print("=" * 60)
    print("EXAMPLE 1: Single-shot query with built-in tools")
    print("=" * 60)

    harness = AgentHarness(
        config=HarnessConfig(
            system_prompt="You are a helpful file-system assistant.",
            verbose=True,
        )
    )

    response = harness.run("List the files in the current directory and tell me what this project is about.")
    print(f"\n\nFinal answer:\n{response}")
    print(f"\n{harness.stats.summary()}\n")


# ---------------------------------------------------------------
# 3. Custom tools + multi-turn conversation
# ---------------------------------------------------------------

def example_multi_turn():
    print("=" * 60)
    print("EXAMPLE 2: Multi-turn chat with a custom tool")
    print("=" * 60)

    harness = AgentHarness(
        tools=[search_contacts],
        config=HarnessConfig(
            system_prompt="You are a company directory assistant. Use the search_contacts tool to look up employees.",
            verbose=True,
        ),
        include_builtins=False,
    )

    r1 = harness.chat("Who works in engineering?")
    print(f"\n\nTurn 1: {r1}")

    r2 = harness.chat("What's Alice's email?")
    print(f"\n\nTurn 2: {r2}")

    print(f"\n{harness.stats.summary()}\n")


# ---------------------------------------------------------------
# 4. Human-in-the-loop approval
# ---------------------------------------------------------------

def example_approval_gate():
    print("=" * 60)
    print("EXAMPLE 3: Human-in-the-loop tool approval")
    print("=" * 60)

    harness = AgentHarness(
        config=HarnessConfig(
            system_prompt="You are a helpful assistant.",
            verbose=True,
        )
    )

    def approval_callback(tool_name: str, arguments: dict) -> bool:
        if tool_name in ("run_shell", "write_file"):
            print(f"\n  [GATE] Tool '{tool_name}' requires approval.")
            print(f"  [GATE] Args: {json.dumps(arguments, default=str)[:300]}")
            answer = input("  [GATE] Allow? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        return True

    harness.on_tool_call(approval_callback)

    response = harness.run("What is the current date and time?")
    print(f"\n\nFinal: {response}")
    print(f"\n{harness.stats.summary()}\n")


# ---------------------------------------------------------------
# 5. Non-streaming mode
# ---------------------------------------------------------------

def example_no_streaming():
    print("=" * 60)
    print("EXAMPLE 4: Non-streaming mode")
    print("=" * 60)

    harness = AgentHarness(
        config=HarnessConfig(
            stream=False,
            verbose=True,
        ),
        include_builtins=False,
    )

    response = harness.run("Explain what an agent harness is in 2 sentences.")
    print(f"Response: {response}")
    print(f"\n{harness.stats.summary()}\n")


# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Agent Harness — Example Usage\n")
    print("Set ANTHROPIC_API_KEY before running.\n")

    example_no_streaming()
    # Uncomment the examples you want to run:
    # example_single_shot()
    # example_multi_turn()
    # example_approval_gate()
