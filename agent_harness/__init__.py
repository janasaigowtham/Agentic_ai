"""
Agent Harness — A production-ready agentic loop built on the Claude API.

Usage:
    from agent_harness import AgentHarness, tool

    @tool
    def get_weather(location: str) -> str:
        \"\"\"Get current weather for a location.\"\"\"
        return f"72°F and sunny in {location}"

    harness = AgentHarness(tools=[get_weather])
    response = harness.run("What's the weather in Paris?")
    print(response)
"""

from agent_harness.harness import AgentHarness
from agent_harness.tools import tool, ToolRegistry

__all__ = ["AgentHarness", "tool", "ToolRegistry"]
