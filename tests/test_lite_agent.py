from collections import defaultdict
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from crewai import LLM, Agent
from crewai.flow import Flow, start
from crewai.lite_agent import LiteAgent, LiteAgentOutput
from crewai.tools import BaseTool
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.agent_events import LiteAgentExecutionStartedEvent
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
@pytest.mark.parametrize("verbose", [True, False])
def test_lite_agent_created_with_correct_parameters(monkeypatch, verbose):
    """Test that LiteAgent is created with the correct parameters when Agent.kickoff() is called."""
    # Create a test agent with specific parameters
    llm = LLM(model="gpt-4o-mini")
    custom_tools = [WebSearchTool(), CalculatorTool()]
    max_iter = 10
    max_execution_time = 300

    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory",
        llm=llm,
        tools=custom_tools,
        max_iter=max_iter,
        max_execution_time=max_execution_time,
        verbose=verbose,
    )

    # Create a mock to capture the created LiteAgent
    created_lite_agent = None
    original_lite_agent = LiteAgent

    # Define a mock LiteAgent class that captures its arguments
    class MockLiteAgent(original_lite_agent):
        def __init__(self, **kwargs):
            nonlocal created_lite_agent
            created_lite_agent = kwargs
            super().__init__(**kwargs)

    # Patch the LiteAgent class
    monkeypatch.setattr("crewai.agent.LiteAgent", MockLiteAgent)

    # Call kickoff to create the LiteAgent
    agent.kickoff("Test query")

    # Verify all parameters were passed correctly
    assert created_lite_agent is not None
    assert created_lite_agent["role"] == "Test Agent"
    assert created_lite_agent["goal"] == "Test Goal"
    assert created_lite_agent["backstory"] == "Test Backstory"
    assert created_lite_agent["llm"] == llm
    assert len(created_lite_agent["tools"]) == 2
    assert isinstance(created_lite_agent["tools"][0], WebSearchTool)
    assert isinstance(created_lite_agent["tools"][1], CalculatorTool)
    assert created_lite_agent["max_iterations"] == max_iter
    assert created_lite_agent["max_execution_time"] == max_execution_time
    assert created_lite_agent["verbose"] == verbose
    assert created_lite_agent["response_format"] is None

    # Test with a response_format
    monkeypatch.setattr("crewai.agent.LiteAgent", MockLiteAgent)

    class TestResponse(BaseModel):
        test_field: str

    agent.kickoff("Test query", response_format=TestResponse)
    assert created_lite_agent["response_format"] == TestResponse


@pytest.mark.vcr(filter_headers=["authorization"])
def test_lite_agent_with_tools():
    """Test that Agent can use tools."""
    # Create a LiteAgent with tools
    llm = LLM(model="gpt-4o-mini")
    agent = Agent(
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
    """Test that Agent can return a simple structured output."""

    class SimpleOutput(BaseModel):
        """Simple structure for agent outputs."""

        summary: str = Field(description="A brief summary of findings")
        confidence: int = Field(description="Confidence level from 1-100")

    web_search_tool = WebSearchTool()

    llm = LLM(model="gpt-4o-mini")
    agent = Agent(
        role="Info Gatherer",
        goal="Provide brief information",
        backstory="You gather and summarize information quickly.",
        llm=llm,
        tools=[web_search_tool],
        verbose=True,
    )

    result = agent.kickoff(
        "What is the population of Tokyo? Return your structured output in JSON format with the following fields: summary, confidence",
        response_format=SimpleOutput,
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
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant who can search for information about the population of Tokyo.",
        llm=llm,
        tools=[WebSearchTool()],
        verbose=True,
    )

    result = agent.kickoff(
        "What is the population of Tokyo? Return your structured output in JSON format with the following fields: summary, confidence"
    )

    assert result.usage_metrics is not None
    assert result.usage_metrics["total_tokens"] > 0


@pytest.mark.vcr(filter_headers=["authorization"])
@pytest.mark.asyncio
async def test_lite_agent_returns_usage_metrics_async():
    """Test that LiteAgent returns usage metrics when run asynchronously."""
    llm = LLM(model="gpt-4o-mini")
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the population of Tokyo",
        backstory="You are a helpful research assistant who can search for information about the population of Tokyo.",
        llm=llm,
        tools=[WebSearchTool()],
        verbose=True,
    )

    result = await agent.kickoff_async(
        "What is the population of Tokyo? Return your structured output in JSON format with the following fields: summary, confidence"
    )
    assert isinstance(result, LiteAgentOutput)
    assert "21 million" in result.raw or "37 million" in result.raw
    assert result.usage_metrics is not None
    assert result.usage_metrics["total_tokens"] > 0


class TestFlow(Flow):
    """A test flow that creates and runs an agent."""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        super().__init__()

    @start()
    def start(self):
        agent = Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory",
            llm=self.llm,
            tools=self.tools,
        )
        return agent.kickoff("Test query")


def verify_agent_parent_flow(result, agent, flow):
    """Verify that both the result and agent have the correct parent flow."""
    assert result.parent_flow is flow
    assert agent is not None
    assert agent.parent_flow is flow


def test_sets_parent_flow_when_inside_flow():
    captured_agent = None

    mock_llm = Mock(spec=LLM)
    mock_llm.call.return_value = "Test response"

    class MyFlow(Flow):
        @start()
        def start(self):
            agent = Agent(
                role="Test Agent",
                goal="Test Goal",
                backstory="Test Backstory",
                llm=mock_llm,
                tools=[WebSearchTool()],
            )
            return agent.kickoff("Test query")

    flow = MyFlow()
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
        def capture_agent(source, event):
            nonlocal captured_agent
            captured_agent = source

        flow.kickoff()
        assert captured_agent.parent_flow is flow

@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_is_called_using_string():
    guardrail_events = defaultdict(list)
    from crewai.utilities.events import LLMGuardrailCompletedEvent, LLMGuardrailStartedEvent
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def capture_guardrail_started(source, event):
            guardrail_events["started"].append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def capture_guardrail_completed(source, event):
            guardrail_events["completed"].append(event)

        agent = Agent(
            role="Sports Analyst",
            goal="Gather information about the best soccer players",
            backstory="""You are an expert at gathering and organizing information. You carefully collect details and present them in a structured way.""",
            guardrail="""Only include Brazilian players, both women and men""",
        )

        result = agent.kickoff(messages="Top 10 best players in the world?")

        assert len(guardrail_events['started']) == 2
        assert len(guardrail_events['completed']) == 2
        assert not guardrail_events['completed'][0].success
        assert guardrail_events['completed'][1].success
        assert "Here are the top 10 best soccer players in the world, focusing exclusively on Brazilian players" in result.raw

@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_is_called_using_callable():
    guardrail_events = defaultdict(list)
    from crewai.utilities.events import LLMGuardrailCompletedEvent, LLMGuardrailStartedEvent
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def capture_guardrail_started(source, event):
            guardrail_events["started"].append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def capture_guardrail_completed(source, event):
            guardrail_events["completed"].append(event)

        agent = Agent(
            role="Sports Analyst",
            goal="Gather information about the best soccer players",
            backstory="""You are an expert at gathering and organizing information. You carefully collect details and present them in a structured way.""",
            guardrail=lambda output: (True, "Pelé - Santos, 1958"),
        )

        result = agent.kickoff(messages="Top 1 best players in the world?")

        assert len(guardrail_events['started']) == 1
        assert len(guardrail_events['completed']) == 1
        assert guardrail_events['completed'][0].success
        assert "Pelé - Santos, 1958" in result.raw

@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_reached_attempt_limit():
    guardrail_events = defaultdict(list)
    from crewai.utilities.events import LLMGuardrailCompletedEvent, LLMGuardrailStartedEvent
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def capture_guardrail_started(source, event):
            guardrail_events["started"].append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def capture_guardrail_completed(source, event):
            guardrail_events["completed"].append(event)

        agent = Agent(
            role="Sports Analyst",
            goal="Gather information about the best soccer players",
            backstory="""You are an expert at gathering and organizing information. You carefully collect details and present them in a structured way.""",
            guardrail=lambda output: (False, "You are not allowed to include Brazilian players"),
            guardrail_max_retries=2,
        )

        with pytest.raises(Exception, match="Agent's guardrail failed validation after 2 retries"):
            agent.kickoff(messages="Top 10 best players in the world?")

        assert len(guardrail_events['started']) == 3 # 2 retries + 1 initial call
        assert len(guardrail_events['completed']) == 3 # 2 retries + 1 initial call
        assert not guardrail_events['completed'][0].success
        assert not guardrail_events['completed'][1].success
        assert not guardrail_events['completed'][2].success


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_output_when_guardrail_returns_base_model():
    class Player(BaseModel):
        name: str
        country: str

    agent = Agent(
        role="Sports Analyst",
        goal="Gather information about the best soccer players",
        backstory="""You are an expert at gathering and organizing information. You carefully collect details and present them in a structured way.""",
        guardrail=lambda output: (True, Player(name="Lionel Messi", country="Argentina")),
    )

    result = agent.kickoff(messages="Top 10 best players in the world?")

    assert result.pydantic == Player(name="Lionel Messi", country="Argentina")
