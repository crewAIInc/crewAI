import asyncio
from typing import cast

import pytest
from pydantic import BaseModel, Field

from crewai import LLM
from crewai.lite_agent import LiteAgent
from crewai.tools import BaseTool
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.tool_usage_events import ToolUsageStartedEvent


# A simple test tool
class SecretLookupTool(BaseTool):
    name: str = "secret_lookup"
    description: str = "A tool to lookup secrets"

    def _run(self) -> str:
        return "SUPERSECRETPASSWORD123"


# Define Mock Search Tool
class WebSearchTool(BaseTool):
    """Tool for searching the web for information."""

    name: str = "search_web"
    description: str = "Search the web for information about a topic."

    def _run(self, query: str) -> str:
        """Search the web for information about a topic."""
        # This is a mock implementation
        if "tokyo" in query.lower():
            return "Tokyo's population in 2023 was approximately 21 million people in the city proper, and 37 million in the greater metropolitan area."
        elif "climate change" in query.lower() and "coral" in query.lower():
            return "Climate change severely impacts coral reefs through: 1) Ocean warming causing coral bleaching, 2) Ocean acidification reducing calcification, 3) Sea level rise affecting light availability, 4) Increased storm frequency damaging reef structures. Sources: NOAA Coral Reef Conservation Program, Global Coral Reef Alliance."
        else:
            return f"Found information about {query}: This is a simulated search result for demonstration purposes."


# Define Mock Calculator Tool
class CalculatorTool(BaseTool):
    """Tool for performing calculations."""

    name: str = "calculate"
    description: str = "Calculate the result of a mathematical expression."

    def _run(self, expression: str) -> str:
        """Calculate the result of a mathematical expression."""
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {str(e)}"


# Define a custom response format using Pydantic
class ResearchResult(BaseModel):
    """Structure for research results."""

    main_findings: str = Field(description="The main findings from the research")
    key_points: list[str] = Field(description="List of key points")
    sources: list[str] = Field(description="List of sources used")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_lite_agent_with_tools():
    """Test that LiteAgent can use tools."""
    # Create a LiteAgent with tools
    llm = LLM(model="gpt-4o-mini")
    agent = LiteAgent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant who can search for information about the population of Tokyo.",
        llm=llm,
        tools=[WebSearchTool()],
        verbose=True,
    )

    result = agent.kickoff(
        "What is the population of Tokyo and how many people would that be per square kilometer if Tokyo's area is 2,194 square kilometers?"
    )

    assert (
        "21 million" in result.raw or "37 million" in result.raw
    ), "Agent should find Tokyo's population"
    assert (
        "per square kilometer" in result.raw
    ), "Agent should calculate population density"

    received_events = []

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def event_handler(source, event):
        received_events.append(event)

    agent.kickoff("What are the effects of climate change on coral reefs?")

    # Verify tool usage events were emitted
    assert len(received_events) > 0, "Tool usage events should be emitted"
    event = received_events[0]
    assert isinstance(event, ToolUsageStartedEvent)
    assert event.agent_role == "Research Assistant"
    assert event.tool_name == "search_web"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_lite_agent_structured_output():
    """Test that LiteAgent can return a simple structured output."""

    class SimpleOutput(BaseModel):
        """Simple structure for agent outputs."""

        summary: str = Field(description="A brief summary of findings")
        confidence: int = Field(description="Confidence level from 1-100")

    web_search_tool = WebSearchTool()

    llm = LLM(model="gpt-4o-mini")
    agent = LiteAgent(
        role="Info Gatherer",
        goal="Provide brief information",
        backstory="You gather and summarize information quickly.",
        llm=llm,
        tools=[web_search_tool],
        verbose=True,
        response_format=SimpleOutput,
    )

    result = agent.kickoff(
        "What is the population of Tokyo? Return your strucutred output in JSON format with the following fields: summary, confidence"
    )

    print(f"\n=== Agent Result Type: {type(result)}")
    print(f"=== Agent Result: {result}")
    print(f"=== Pydantic: {result.pydantic}")

    assert result.pydantic is not None, "Should return a Pydantic model"

    output = cast(SimpleOutput, result.pydantic)

    assert isinstance(output.summary, str), "Summary should be a string"
    assert len(output.summary) > 0, "Summary should not be empty"
    assert isinstance(output.confidence, int), "Confidence should be an integer"
    assert 1 <= output.confidence <= 100, "Confidence should be between 1 and 100"

    assert "tokyo" in output.summary.lower() or "population" in output.summary.lower()

    assert result.usage_metrics is not None

    return result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_lite_agent_returns_usage_metrics():
    """Test that LiteAgent returns usage metrics."""
    llm = LLM(model="gpt-4o-mini")
    agent = LiteAgent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant who can search for information about the population of Tokyo.",
        llm=llm,
        tools=[WebSearchTool()],
        verbose=True,
    )

    result = agent.kickoff(
        "What is the population of Tokyo? Return your strucutred output in JSON format with the following fields: summary, confidence"
    )

    assert result.usage_metrics is not None
    assert result.usage_metrics["total_tokens"] > 0
