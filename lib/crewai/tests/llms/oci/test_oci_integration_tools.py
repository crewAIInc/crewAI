from __future__ import annotations

from pydantic import BaseModel

from crewai import Agent
from crewai.tools import tool


class CalculationResult(BaseModel):
    operation: str
    result: int
    explanation: str


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


def test_oci_agent_uses_tool(
    oci_tool_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    agent = Agent(
        role="Calculator",
        goal="Use tools to solve arithmetic problems",
        backstory="You are a precise calculator that must use the available tools.",
        llm=oci_live_llm_factory(
            oci_tool_model,
            max_tokens=oci_token_budget(oci_tool_model, "agent"),
            temperature=oci_temperature_for_model(oci_tool_model),
        ),
        tools=[add_numbers],
        verbose=True,
    )

    result = agent.kickoff(oci_prompts["tool"])

    assert "42" in result.raw
    assert add_numbers.current_usage_count >= 1


def test_oci_agent_kickoff_structured_output_with_tools(
    oci_tool_model: str,
    oci_live_llm_factory,
    oci_prompts: dict[str, str],
    oci_temperature_for_model,
    oci_token_budget,
):
    agent = Agent(
        role="Calculator",
        goal="Perform calculations using available tools",
        backstory="You are a calculator assistant that uses tools to compute results.",
        llm=oci_live_llm_factory(
            oci_tool_model,
            max_tokens=oci_token_budget(oci_tool_model, "agent"),
            temperature=oci_temperature_for_model(oci_tool_model),
        ),
        tools=[add_numbers],
        verbose=True,
    )

    result = agent.kickoff(
        messages=oci_prompts["tool_structured"],
        response_format=CalculationResult,
    )

    assert result.pydantic is not None
    assert isinstance(result.pydantic, CalculationResult)
    assert result.pydantic.result == 42
    assert result.pydantic.operation
    assert result.pydantic.explanation


def test_oci_agent_handles_multiple_tool_asks_in_sequence(
    oci_tool_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    agent = Agent(
        role="Calculator",
        goal="Use tools to solve arithmetic problems accurately across repeated asks",
        backstory="You are a calculator assistant that must use the available tool every time.",
        llm=oci_live_llm_factory(
            oci_tool_model,
            max_tokens=oci_token_budget(oci_tool_model, "agent"),
            temperature=oci_temperature_for_model(oci_tool_model),
        ),
        tools=[add_numbers],
        verbose=True,
    )

    prompts = [
        "Use add_numbers to calculate 2 + 5. Return only the final result.",
        "Use add_numbers to calculate 10 + 11. Return only the final result.",
        "Use add_numbers to calculate 20 + 22. Return only the final result.",
    ]

    results = [agent.kickoff(prompt) for prompt in prompts]

    assert "7" in results[0].raw
    assert "21" in results[1].raw
    assert "42" in results[2].raw
