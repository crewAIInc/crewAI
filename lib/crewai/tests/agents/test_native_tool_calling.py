"""Integration tests for native tool calling functionality.

These tests verify that agents can use native function calling
when the LLM supports it, across multiple providers.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """A calculator tool that performs mathematical calculations."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations. Use this for any math operations."
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the calculation."""
        try:
            # Safe evaluation for basic math
            result = eval(expression)  # noqa: S307
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating {expression}: {e}"


class WeatherInput(BaseModel):
    """Input schema for weather tool."""

    location: str = Field(description="City name to get weather for")


class WeatherTool(BaseTool):
    """A mock weather tool for testing."""

    name: str = "get_weather"
    description: str = "Get the current weather for a location"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        """Get weather (mock implementation)."""
        return f"The weather in {location} is sunny with a temperature of 72°F"

class FailingTool(BaseTool):
    """A tool that always fails."""
    name: str = "failing_tool"
    description: str = "This tool always fails"
    def _run(self) -> str:
        raise Exception("This tool always fails")


class LocalSearchInput(BaseModel):
    query: str = Field(description="Search query")


class ParallelProbe:
    """Thread-safe in-memory recorder for tool execution windows."""

    _lock = threading.Lock()
    _windows: list[tuple[str, float, float]] = []

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._windows = []

    @classmethod
    def record(cls, tool_name: str, start: float, end: float) -> None:
        with cls._lock:
            cls._windows.append((tool_name, start, end))

    @classmethod
    def windows(cls) -> list[tuple[str, float, float]]:
        with cls._lock:
            return list(cls._windows)


def _parallel_prompt() -> str:
    return (
        "This is a tool-calling compliance test. "
        "In your next assistant turn, emit exactly 3 tool calls in the same response (parallel tool calls), in this order: "
        "1) parallel_local_search_one(query='latest OpenAI model release notes'), "
        "2) parallel_local_search_two(query='latest Anthropic model release notes'), "
        "3) parallel_local_search_three(query='latest Gemini model release notes'). "
        "Do not call any other tools and do not answer before those 3 tool calls are emitted. "
        "After the tool results return, provide a one paragraph summary."
    )


def _max_concurrency(windows: list[tuple[str, float, float]]) -> int:
    points: list[tuple[float, int]] = []
    for _, start, end in windows:
        points.append((start, 1))
        points.append((end, -1))
    points.sort(key=lambda p: (p[0], p[1]))

    current = 0
    maximum = 0
    for _, delta in points:
        current += delta
        if current > maximum:
            maximum = current
    return maximum


def _assert_tools_overlapped() -> None:
    windows = ParallelProbe.windows()
    local_windows = [
        w
        for w in windows
        if w[0].startswith("parallel_local_search_")
    ]

    assert len(local_windows) >= 3, f"Expected at least 3 local tool calls, got {len(local_windows)}"
    assert _max_concurrency(local_windows) >= 2, "Expected overlapping local tool executions"


@pytest.fixture
def calculator_tool() -> CalculatorTool:
    """Create a calculator tool for testing."""
    return CalculatorTool()


@pytest.fixture
def weather_tool() -> WeatherTool:
    """Create a weather tool for testing."""
    return WeatherTool()

@pytest.fixture
def failing_tool() -> BaseTool:
    """Create a weather tool for testing."""
    return FailingTool(

    )


@pytest.fixture
def parallel_tools() -> list[BaseTool]:
    """Create local tools used to verify native parallel execution deterministically."""

    class ParallelLocalSearchOne(BaseTool):
        name: str = "parallel_local_search_one"
        description: str = "Local search tool #1 for concurrency testing."
        args_schema: type[BaseModel] = LocalSearchInput

        def _run(self, query: str) -> str:
            start = time.perf_counter()
            time.sleep(1.0)
            end = time.perf_counter()
            ParallelProbe.record(self.name, start, end)
            return f"[one] {query}"

    class ParallelLocalSearchTwo(BaseTool):
        name: str = "parallel_local_search_two"
        description: str = "Local search tool #2 for concurrency testing."
        args_schema: type[BaseModel] = LocalSearchInput

        def _run(self, query: str) -> str:
            start = time.perf_counter()
            time.sleep(1.0)
            end = time.perf_counter()
            ParallelProbe.record(self.name, start, end)
            return f"[two] {query}"

    class ParallelLocalSearchThree(BaseTool):
        name: str = "parallel_local_search_three"
        description: str = "Local search tool #3 for concurrency testing."
        args_schema: type[BaseModel] = LocalSearchInput

        def _run(self, query: str) -> str:
            start = time.perf_counter()
            time.sleep(1.0)
            end = time.perf_counter()
            ParallelProbe.record(self.name, start, end)
            return f"[three] {query}"

    return [
        ParallelLocalSearchOne(),
        ParallelLocalSearchTwo(),
        ParallelLocalSearchThree(),
    ]


def _attach_parallel_probe_handler() -> None:
    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _capture_tool_window(_source, event: ToolUsageFinishedEvent):
        if not event.tool_name.startswith("parallel_local_search_"):
            return
        ParallelProbe.record(
            event.tool_name,
            event.started_at.timestamp(),
            event.finished_at.timestamp(),
        )

# =============================================================================
# OpenAI Provider Tests
# =============================================================================


class TestOpenAINativeToolCalling:
    """Tests for native tool calling with OpenAI models."""

    @pytest.mark.vcr()
    def test_openai_agent_with_native_tool_calling(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test OpenAI agent can use native tool calling."""
        agent = Agent(
            role="Math Assistant",
            goal="Help users with mathematical calculations",
            backstory="You are a helpful math assistant.",
            tools=[calculator_tool],
            llm=LLM(model="gpt-4o-mini"),
            verbose=False,
            max_iter=3,
        )

        task = Task(
            description="Calculate what is 15 * 8",
            expected_output="The result of the calculation",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None
        assert "120" in str(result.raw)

    def test_openai_agent_kickoff_with_tools_mocked(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test OpenAI agent kickoff with mocked LLM call."""
        llm = LLM(model="gpt-5-nano")

        with patch.object(llm, "call", return_value="The answer is 120.") as mock_call:
            agent = Agent(
                role="Math Assistant",
                goal="Calculate math",
                backstory="You calculate.",
                tools=[calculator_tool],
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Calculate 15 * 8",
                expected_output="Result",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()

            assert mock_call.called
            assert result is not None

    @pytest.mark.vcr()
    @pytest.mark.timeout(180)
    def test_openai_parallel_native_tool_calling_test_crew(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="gpt-5-nano", temperature=1),
            verbose=False,
            max_iter=3,
        )
        task = Task(
            description=_parallel_prompt(),
            expected_output="A one sentence summary of both tool outputs",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        assert result is not None
        _assert_tools_overlapped()

    @pytest.mark.vcr()
    @pytest.mark.timeout(180)
    def test_openai_parallel_native_tool_calling_test_agent_kickoff(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="gpt-4o-mini"),
            verbose=False,
            max_iter=3,
        )
        result = agent.kickoff(_parallel_prompt())
        assert result is not None
        _assert_tools_overlapped()


# =============================================================================
# Anthropic Provider Tests
# =============================================================================
class TestAnthropicNativeToolCalling:
    """Tests for native tool calling with Anthropic models."""

    @pytest.fixture(autouse=True)
    def mock_anthropic_api_key(self):
        """Mock ANTHROPIC_API_KEY for tests."""
        if "ANTHROPIC_API_KEY" not in os.environ:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                yield
        else:
            yield

    @pytest.mark.vcr()
    def test_anthropic_agent_with_native_tool_calling(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Anthropic agent can use native tool calling."""
        agent = Agent(
            role="Math Assistant",
            goal="Help users with mathematical calculations",
            backstory="You are a helpful math assistant.",
            tools=[calculator_tool],
            llm=LLM(model="anthropic/claude-3-5-haiku-20241022"),
            verbose=False,
            max_iter=3,
        )

        task = Task(
            description="Calculate what is 15 * 8",
            expected_output="The result of the calculation",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None

    def test_anthropic_agent_kickoff_with_tools_mocked(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Anthropic agent kickoff with mocked LLM call."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")

        with patch.object(llm, "call", return_value="The answer is 120.") as mock_call:
            agent = Agent(
                role="Math Assistant",
                goal="Calculate math",
                backstory="You calculate.",
                tools=[calculator_tool],
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Calculate 15 * 8",
                expected_output="Result",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()

            assert mock_call.called
            assert result is not None

    @pytest.mark.vcr()
    def test_anthropic_parallel_native_tool_calling_test_crew(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="anthropic/claude-sonnet-4-6"),
            verbose=False,
            max_iter=3,
        )
        task = Task(
            description=_parallel_prompt(),
            expected_output="A one sentence summary of both tool outputs",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        assert result is not None
        _assert_tools_overlapped()

    @pytest.mark.vcr()
    def test_anthropic_parallel_native_tool_calling_test_agent_kickoff(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="anthropic/claude-sonnet-4-6"),
            verbose=False,
            max_iter=3,
        )
        result = agent.kickoff(_parallel_prompt())
        assert result is not None
        _assert_tools_overlapped()


# =============================================================================
# Google/Gemini Provider Tests
# =============================================================================


class TestGeminiNativeToolCalling:
    """Tests for native tool calling with Gemini models."""

    @pytest.fixture(autouse=True)
    def mock_google_api_key(self):
        """Mock GOOGLE_API_KEY for tests."""
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                yield
        else:
            yield


    @pytest.mark.vcr()
    def test_gemini_agent_with_native_tool_calling(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Gemini agent can use native tool calling."""

        agent = Agent(
            role="Math Assistant",
            goal="Help users with mathematical calculations",
            backstory="You are a helpful math assistant.",
            tools=[calculator_tool],
            llm=LLM(model="gemini/gemini-2.5-flash"),
        )

        task = Task(
            description="Calculate what is 15 * 8",
            expected_output="The result of the calculation",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None

    def test_gemini_agent_kickoff_with_tools_mocked(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Gemini agent kickoff with mocked LLM call."""
        llm = LLM(model="gemini/gemini-2.5-flash")

        with patch.object(llm, "call", return_value="The answer is 120.") as mock_call:
            agent = Agent(
                role="Math Assistant",
                goal="Calculate math",
                backstory="You calculate.",
                tools=[calculator_tool],
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Calculate 15 * 8",
                expected_output="Result",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()

            assert mock_call.called
            assert result is not None

    @pytest.mark.vcr()
    def test_gemini_parallel_native_tool_calling_test_crew(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="gemini/gemini-2.5-flash"),
            verbose=False,
            max_iter=3,
        )
        task = Task(
            description=_parallel_prompt(),
            expected_output="A one sentence summary of both tool outputs",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        assert result is not None
        _assert_tools_overlapped()

    @pytest.mark.vcr()
    def test_gemini_parallel_native_tool_calling_test_agent_kickoff(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="gemini/gemini-2.5-flash"),
            verbose=False,
            max_iter=3,
        )
        result = agent.kickoff(_parallel_prompt())
        assert result is not None
        _assert_tools_overlapped()


# =============================================================================
# Azure Provider Tests
# =============================================================================


class TestAzureNativeToolCalling:
    """Tests for native tool calling with Azure OpenAI models."""

    @pytest.fixture(autouse=True)
    def mock_azure_env(self):
        """Mock Azure environment variables for tests."""
        env_vars = {
            "AZURE_API_KEY": "test-key",
            "AZURE_API_BASE": "https://test.openai.azure.com",
            "AZURE_API_VERSION": "2024-02-15-preview",
        }
        # Only patch if keys are not already in environment
        if "AZURE_API_KEY" not in os.environ:
            with patch.dict(os.environ, env_vars):
                yield
        else:
            yield

    @pytest.mark.vcr()
    def test_azure_agent_with_native_tool_calling(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Azure agent can use native tool calling."""
        agent = Agent(
            role="Math Assistant",
            goal="Help users with mathematical calculations",
            backstory="You are a helpful math assistant.",
            tools=[calculator_tool],
            llm=LLM(model="azure/gpt-5-nano"),
            verbose=False,
            max_iter=3,
        )

        task = Task(
            description="Calculate what is 15 * 8",
            expected_output="The result of the calculation",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None
        assert "120" in str(result.raw)

    def test_azure_agent_kickoff_with_tools_mocked(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Azure agent kickoff with mocked LLM call."""
        llm = LLM(
            model="azure/gpt-5-nano",
            api_key="test-key",
            base_url="https://test.openai.azure.com",
        )

        with patch.object(llm, "call", return_value="The answer is 120.") as mock_call:
            agent = Agent(
                role="Math Assistant",
                goal="Calculate math",
                backstory="You calculate.",
                tools=[calculator_tool],
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Calculate 15 * 8",
                expected_output="Result",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()

            assert mock_call.called
            assert result is not None

    @pytest.mark.vcr()
    def test_azure_parallel_native_tool_calling_test_crew(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="azure/gpt-5-nano"),
            verbose=False,
            max_iter=3,
        )
        task = Task(
            description=_parallel_prompt(),
            expected_output="A one sentence summary of both tool outputs",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        assert result is not None
        _assert_tools_overlapped()

    @pytest.mark.vcr()
    def test_azure_parallel_native_tool_calling_test_agent_kickoff(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="azure/gpt-5-nano"),
            verbose=False,
            max_iter=3,
        )
        result = agent.kickoff(_parallel_prompt())
        assert result is not None
        _assert_tools_overlapped()


# =============================================================================
# Bedrock Provider Tests
# =============================================================================


class TestBedrockNativeToolCalling:
    """Tests for native tool calling with AWS Bedrock models."""

    @pytest.fixture(autouse=True)
    def validate_bedrock_credentials_for_live_recording(self):
        """Run Bedrock tests only when explicitly enabled."""
        run_live_bedrock = os.getenv("RUN_BEDROCK_LIVE_TESTS", "false").lower() == "true"

        if not run_live_bedrock:
            pytest.skip(
                "Skipping Bedrock tests by default. "
                "Set RUN_BEDROCK_LIVE_TESTS=true with valid AWS credentials to enable."
            )

        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        if (
            not access_key
            or not secret_key
            or access_key.startswith(("fake-", "test-"))
            or secret_key.startswith(("fake-", "test-"))
        ):
            pytest.skip(
                "Skipping Bedrock tests: valid AWS credentials are required when "
                "RUN_BEDROCK_LIVE_TESTS=true."
            )

        yield

    @pytest.mark.vcr()
    def test_bedrock_agent_kickoff_with_tools_mocked(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test Bedrock agent kickoff with mocked LLM call."""
        llm = LLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0")

        agent = Agent(
            role="Math Assistant",
            goal="Calculate math",
            backstory="You calculate.",
            tools=[calculator_tool],
            llm=llm,
            verbose=False,
            max_iter=5,
        )

        task = Task(
            description="Calculate 15 * 8",
            expected_output="Result",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None
        assert "120" in str(result.raw)

    @pytest.mark.vcr()
    def test_bedrock_parallel_native_tool_calling_test_crew(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0"),
            verbose=False,
            max_iter=3,
        )
        task = Task(
            description=_parallel_prompt(),
            expected_output="A one sentence summary of both tool outputs",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        assert result is not None
        _assert_tools_overlapped()

    @pytest.mark.vcr()
    def test_bedrock_parallel_native_tool_calling_test_agent_kickoff(
        self, parallel_tools: list[BaseTool]
    ) -> None:
        ParallelProbe.reset()
        _attach_parallel_probe_handler()
        agent = Agent(
            role="Parallel Tool Agent",
            goal="Use both tools exactly as instructed",
            backstory="You follow tool instructions precisely.",
            tools=parallel_tools,
            llm=LLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0"),
            verbose=False,
            max_iter=3,
        )
        result = agent.kickoff(_parallel_prompt())
        assert result is not None
        _assert_tools_overlapped()


# =============================================================================
# Cross-Provider Native Tool Calling Behavior Tests
# =============================================================================


class TestNativeToolCallingBehavior:
    """Tests for native tool calling behavior across providers."""

    def test_supports_function_calling_check(self) -> None:
        """Test that supports_function_calling() is properly checked."""
        # OpenAI should support function calling
        openai_llm = LLM(model="gpt-5-nano")
        assert hasattr(openai_llm, "supports_function_calling")
        assert openai_llm.supports_function_calling() is True

    def test_anthropic_supports_function_calling(self) -> None:
        """Test that Anthropic models support function calling."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
            assert hasattr(llm, "supports_function_calling")
            assert llm.supports_function_calling() is True

    def test_gemini_supports_function_calling(self) -> None:
        """Test that Gemini models support function calling."""
        llm = LLM(model="gemini/gemini-2.5-flash")
        assert hasattr(llm, "supports_function_calling")
        assert llm.supports_function_calling() is True


# =============================================================================
# Token Usage Tests
# =============================================================================


class TestNativeToolCallingTokenUsage:
    """Tests for token usage with native tool calling."""

    @pytest.mark.vcr()
    def test_openai_native_tool_calling_token_usage(
        self, calculator_tool: CalculatorTool
    ) -> None:
        """Test token usage tracking with OpenAI native tool calling."""
        agent = Agent(
            role="Calculator",
            goal="Perform calculations efficiently",
            backstory="You calculate things.",
            tools=[calculator_tool],
            llm=LLM(model="gpt-5-nano"),
            verbose=False,
            max_iter=3,
        )

        task = Task(
            description="What is 100 / 4?",
            expected_output="The result",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.token_usage is not None
        assert result.token_usage.total_tokens > 0
        assert result.token_usage.successful_requests >= 1

        print(f"\n[OPENAI NATIVE TOOL CALLING TOKEN USAGE]")
        print(f"  Prompt tokens: {result.token_usage.prompt_tokens}")
        print(f"  Completion tokens: {result.token_usage.completion_tokens}")
        print(f"  Total tokens: {result.token_usage.total_tokens}")

@pytest.mark.vcr()
def test_native_tool_calling_error_handling(failing_tool: FailingTool):
    """Test that native tool calling handles errors properly and emits error events."""
    import threading
    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageErrorEvent

    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def handle_tool_error(source, event):
        received_events.append(event)
        event_received.set()

    agent = Agent(
        role="Calculator",
        goal="Perform calculations efficiently",
        backstory="You calculate things.",
        tools=[failing_tool],
        llm=LLM(model="gpt-5-nano"),
        verbose=False,
        max_iter=3,
    )

    result = agent.kickoff("Use the failing_tool to do something.")
    assert result is not None

    # Verify error event was emitted
    assert event_received.wait(timeout=10), "ToolUsageErrorEvent was not emitted"
    assert len(received_events) >= 1

    # Verify event attributes
    error_event = received_events[0]
    assert error_event.tool_name == "failing_tool"
    assert error_event.agent_role == agent.role
    assert "This tool always fails" in str(error_event.error)


# =============================================================================
# Max Usage Count Tests for Native Tool Calling
# =============================================================================


class CountingInput(BaseModel):
    """Input schema for counting tool."""

    value: str = Field(description="Value to count")


class CountingTool(BaseTool):
    """A tool that counts its usage."""

    name: str = "counting_tool"
    description: str = "A tool that counts how many times it's been called"
    args_schema: type[BaseModel] = CountingInput

    def _run(self, value: str) -> str:
        """Return the value with a count prefix."""
        return f"Counted: {value}"


class TestMaxUsageCountWithNativeToolCalling:
    """Tests for max_usage_count with native tool calling."""

    @pytest.mark.vcr()
    def test_max_usage_count_tracked_in_native_tool_calling(self) -> None:
        """Test that max_usage_count is properly tracked when using native tool calling."""
        tool = CountingTool(max_usage_count=3)

        # Verify initial state
        assert tool.max_usage_count == 3
        assert tool.current_usage_count == 0

        agent = Agent(
            role="Counting Agent",
            goal="Call the counting tool multiple times",
            backstory="You are an agent that counts things.",
            tools=[tool],
            llm=LLM(model="gpt-5-nano"),
            verbose=False,
            max_iter=5,
        )

        task = Task(
            description="Call the counting_tool 3 times with values 'first', 'second', and 'third'",
            expected_output="The results of the counting operations",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        # Verify usage count was tracked
        assert tool.max_usage_count == 3
        assert tool.current_usage_count <= tool.max_usage_count

    @pytest.mark.vcr()
    def test_max_usage_count_limit_enforced_in_native_tool_calling(self) -> None:
        """Test that when max_usage_count is reached, tool returns error message."""
        tool = CountingTool(max_usage_count=2)

        agent = Agent(
            role="Counting Agent",
            goal="Use the counting tool as many times as requested",
            backstory="You are an agent that counts things. You must try to use the tool for each value requested.",
            tools=[tool],
            llm=LLM(model="gpt-5-nano"),
            verbose=False,
            max_iter=5,
        )

        # Request more tool calls than the max_usage_count allows
        task = Task(
            description="Call the counting_tool 4 times with values 'one', 'two', 'three', and 'four'",
            expected_output="The results of the counting operations, noting any failures",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        # The tool should have been limited to max_usage_count (2) calls
        assert result is not None
        assert tool.current_usage_count == tool.max_usage_count
        # After hitting the limit, further calls should have been rejected

    @pytest.mark.vcr()
    def test_tool_usage_increments_after_successful_execution(self) -> None:
        """Test that usage count increments after each successful native tool call."""
        tool = CountingTool(max_usage_count=10)

        assert tool.current_usage_count == 0

        agent = Agent(
            role="Counting Agent",
            goal="Use the counting tool exactly as requested",
            backstory="You are an agent that counts things precisely.",
            tools=[tool],
            llm=LLM(model="gpt-5-nano"),
            verbose=False,
            max_iter=5,
        )

        task = Task(
            description="Call the counting_tool exactly 2 times: first with value 'alpha', then with value 'beta'",
            expected_output="The results showing both 'Counted: alpha' and 'Counted: beta'",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        # Verify usage count was incremented for each successful call
        assert tool.current_usage_count == 2
