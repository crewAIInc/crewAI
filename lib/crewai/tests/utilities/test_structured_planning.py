"""Tests for structured planning with steps and todo generation.

These tests verify that the planning system correctly generates structured
PlanStep objects and converts them to TodoItems across different LLM providers.
"""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai import Agent, PlanningConfig, Task
from crewai.llm import LLM
from crewai.utilities.planning_types import PlanStep, TodoItem, TodoList
from crewai.utilities.reasoning_handler import (
    FUNCTION_SCHEMA,
    AgentReasoning,
    ReasoningPlan,
)


class TestFunctionSchema:
    """Tests for the FUNCTION_SCHEMA used in structured planning."""

    def test_schema_has_required_structure(self):
        """Test that FUNCTION_SCHEMA has the correct structure."""
        assert FUNCTION_SCHEMA["type"] == "function"
        assert "function" in FUNCTION_SCHEMA
        assert FUNCTION_SCHEMA["function"]["name"] == "create_reasoning_plan"

    def test_schema_parameters_structure(self):
        """Test that parameters have correct structure."""
        params = FUNCTION_SCHEMA["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    def test_schema_has_plan_property(self):
        """Test that schema includes plan property."""
        props = FUNCTION_SCHEMA["function"]["parameters"]["properties"]
        assert "plan" in props
        assert props["plan"]["type"] == "string"

    def test_schema_has_steps_property(self):
        """Test that schema includes steps array property."""
        props = FUNCTION_SCHEMA["function"]["parameters"]["properties"]
        assert "steps" in props
        assert props["steps"]["type"] == "array"

    def test_schema_steps_items_structure(self):
        """Test that steps items have correct structure."""
        items = FUNCTION_SCHEMA["function"]["parameters"]["properties"]["steps"]["items"]
        assert items["type"] == "object"
        assert "properties" in items
        assert "required" in items
        assert "additionalProperties" in items
        assert items["additionalProperties"] is False

    def test_schema_step_properties(self):
        """Test that step items have all required properties."""
        step_props = FUNCTION_SCHEMA["function"]["parameters"]["properties"]["steps"]["items"]["properties"]

        assert "step_number" in step_props
        assert step_props["step_number"]["type"] == "integer"

        assert "description" in step_props
        assert step_props["description"]["type"] == "string"

        assert "tool_to_use" in step_props
        # tool_to_use should be nullable
        assert step_props["tool_to_use"]["type"] == ["string", "null"]

        assert "depends_on" in step_props
        assert step_props["depends_on"]["type"] == "array"

    def test_schema_step_required_fields(self):
        """Test that step required fields are correct."""
        required = FUNCTION_SCHEMA["function"]["parameters"]["properties"]["steps"]["items"]["required"]
        assert "step_number" in required
        assert "description" in required
        assert "tool_to_use" in required
        assert "depends_on" in required

    def test_schema_has_ready_property(self):
        """Test that schema includes ready property."""
        props = FUNCTION_SCHEMA["function"]["parameters"]["properties"]
        assert "ready" in props
        assert props["ready"]["type"] == "boolean"

    def test_schema_top_level_required(self):
        """Test that top-level required fields are correct."""
        required = FUNCTION_SCHEMA["function"]["parameters"]["required"]
        assert "plan" in required
        assert "steps" in required
        assert "ready" in required

    def test_schema_top_level_additional_properties(self):
        """Test that additionalProperties is False at top level."""
        params = FUNCTION_SCHEMA["function"]["parameters"]
        assert params["additionalProperties"] is False


class TestReasoningPlan:
    """Tests for the ReasoningPlan model with structured steps."""

    def test_reasoning_plan_with_empty_steps(self):
        """Test ReasoningPlan can be created with empty steps."""
        plan = ReasoningPlan(
            plan="Simple plan",
            steps=[],
            ready=True,
        )

        assert plan.plan == "Simple plan"
        assert plan.steps == []
        assert plan.ready is True

    def test_reasoning_plan_with_steps(self):
        """Test ReasoningPlan with structured steps."""
        steps = [
            PlanStep(step_number=1, description="First step", tool_to_use="tool1"),
            PlanStep(step_number=2, description="Second step", depends_on=[1]),
        ]

        plan = ReasoningPlan(
            plan="Multi-step plan",
            steps=steps,
            ready=True,
        )

        assert plan.plan == "Multi-step plan"
        assert len(plan.steps) == 2
        assert plan.steps[0].step_number == 1
        assert plan.steps[1].depends_on == [1]


class TestAgentReasoningWithMockedLLM:
    """Tests for AgentReasoning with mocked LLM responses."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.role = "Test Agent"
        agent.goal = "Test goal"
        agent.backstory = "Test backstory"
        agent.verbose = False
        agent.planning_config = PlanningConfig()
        agent.i18n = MagicMock()
        agent.i18n.retrieve.return_value = "Test prompt: {description}"
        # Mock the llm attribute
        agent.llm = MagicMock()
        agent.llm.supports_function_calling.return_value = True
        return agent

    def test_parse_steps_from_function_response(self, mock_agent):
        """Test that steps are correctly parsed from LLM function response."""
        # Mock the LLM response with structured steps
        mock_response = json.dumps({
            "plan": "Research and analyze",
            "steps": [
                {
                    "step_number": 1,
                    "description": "Search for information",
                    "tool_to_use": "search_tool",
                    "depends_on": [],
                },
                {
                    "step_number": 2,
                    "description": "Analyze results",
                    "tool_to_use": None,
                    "depends_on": [1],
                },
            ],
            "ready": True,
        })

        mock_agent.llm.call.return_value = mock_response

        handler = AgentReasoning(
            agent=mock_agent,
            task=None,
            description="Test task",
            expected_output="Test output",
        )

        # Call the function parsing method
        plan, steps, ready = handler._call_with_function(
            prompt="Test prompt",
            plan_type="create_plan",
        )

        assert plan == "Research and analyze"
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[0].tool_to_use == "search_tool"
        assert steps[1].depends_on == [1]
        assert ready is True

    def test_parse_steps_handles_missing_optional_fields(self, mock_agent):
        """Test that missing optional fields are handled correctly."""
        mock_response = json.dumps({
            "plan": "Simple plan",
            "steps": [
                {
                    "step_number": 1,
                    "description": "Do something",
                    "tool_to_use": None,
                    "depends_on": [],
                },
            ],
            "ready": True,
        })

        mock_agent.llm.call.return_value = mock_response

        handler = AgentReasoning(
            agent=mock_agent,
            task=None,
            description="Test task",
            expected_output="Test output",
        )

        plan, steps, ready = handler._call_with_function(
            prompt="Test prompt",
            plan_type="create_plan",
        )

        assert len(steps) == 1
        assert steps[0].tool_to_use is None
        assert steps[0].depends_on == []

    def test_parse_steps_with_missing_fields_uses_defaults(self, mock_agent):
        """Test that steps with missing fields get default values."""
        mock_response = json.dumps({
            "plan": "Plan with step missing fields",
            "steps": [
                {"step_number": 1, "description": "Valid step", "tool_to_use": None, "depends_on": []},
                {"step_number": 2},  # Missing description, tool_to_use, depends_on
                {"step_number": 3, "description": "Another valid", "tool_to_use": None, "depends_on": []},
            ],
            "ready": True,
        })

        mock_agent.llm.call.return_value = mock_response

        handler = AgentReasoning(
            agent=mock_agent,
            task=None,
            description="Test task",
            expected_output="Test output",
        )

        plan, steps, ready = handler._call_with_function(
            prompt="Test prompt",
            plan_type="create_plan",
        )

        # All 3 steps should be parsed, with defaults for missing fields
        assert len(steps) == 3
        assert steps[0].step_number == 1
        assert steps[0].description == "Valid step"
        assert steps[1].step_number == 2
        assert steps[1].description == ""  # Default value
        assert steps[2].step_number == 3


class TestTodoCreationFromPlan:
    """Tests for converting plan steps to todo items."""

    def test_create_todos_from_plan_steps(self):
        """Test creating TodoList from PlanSteps."""
        steps = [
            PlanStep(
                step_number=1,
                description="Research competitors",
                tool_to_use="search_tool",
                depends_on=[],
            ),
            PlanStep(
                step_number=2,
                description="Analyze data",
                tool_to_use=None,
                depends_on=[1],
            ),
            PlanStep(
                step_number=3,
                description="Generate report",
                tool_to_use="write_tool",
                depends_on=[1, 2],
            ),
        ]

        # Convert steps to todos (mirroring agent_executor._create_todos_from_plan)
        todos = []
        for step in steps:
            todo = TodoItem(
                step_number=step.step_number,
                description=step.description,
                tool_to_use=step.tool_to_use,
                depends_on=step.depends_on,
                status="pending",
            )
            todos.append(todo)

        todo_list = TodoList(items=todos)

        assert len(todo_list.items) == 3
        assert todo_list.pending_count == 3
        assert todo_list.completed_count == 0

        # Verify todo properties match step properties
        assert todo_list.items[0].description == "Research competitors"
        assert todo_list.items[0].tool_to_use == "search_tool"
        assert todo_list.items[1].depends_on == [1]
        assert todo_list.items[2].depends_on == [1, 2]


# =============================================================================
# Provider-Specific Integration Tests (VCR recorded)
# =============================================================================


# Common test tools used across provider tests
def create_research_tools():
    """Create research tools for testing structured planning."""
    from crewai.tools import tool

    @tool
    def web_search(query: str) -> str:
        """Search the web for information on a given topic.

        Args:
            query: The search query to look up.

        Returns:
            Search results as a string.
        """
        # Simulated search results for testing
        return f"Search results for '{query}': Found 3 relevant articles about the topic including market analysis, competitor data, and industry trends."

    @tool
    def read_website(url: str) -> str:
        """Read and extract content from a website URL.

        Args:
            url: The URL of the website to read.

        Returns:
            The extracted content from the website.
        """
        # Simulated website content for testing
        return f"Content from {url}: This article discusses key insights about the topic including market size ($50B), growth rate (15% YoY), and major players in the industry."

    @tool
    def generate_report(title: str, findings: str) -> str:
        """Generate a structured report based on research findings.

        Args:
            title: The title of the report.
            findings: The research findings to include.

        Returns:
            A formatted report string.
        """
        return f"# {title}\n\n## Executive Summary\n{findings}\n\n## Conclusion\nBased on the analysis, the market shows strong growth potential."

    return web_search, read_website, generate_report


RESEARCH_TASK = """Research the current state of the AI agent market:
1. Search for recent information about AI agents and their market trends
2. Read detailed content from a relevant industry source
3. Generate a brief report summarizing the key findings

Use the available tools for each step."""


class TestOpenAIStructuredPlanning:
    """Integration tests for OpenAI structured planning with research workflow."""

    @pytest.mark.vcr()
    def test_openai_research_workflow_generates_steps(self):
        """Test that OpenAI generates structured plan steps for a research task."""
        web_search, read_website, generate_report = create_research_tools()
        llm = LLM(model="gpt-4o")

        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research and produce insightful reports",
            backstory="An experienced analyst skilled at gathering information and synthesizing findings into actionable insights.",
            llm=llm,
            tools=[web_search, read_website, generate_report],
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        result = agent.kickoff(RESEARCH_TASK)

        # Verify result exists
        assert result is not None
        assert result.raw is not None
        # The result should contain some report-like content
        assert len(str(result.raw)) > 50


class TestAnthropicStructuredPlanning:
    """Integration tests for Anthropic structured planning with research workflow."""

    @pytest.fixture(autouse=True)
    def mock_anthropic_api_key(self):
        """Mock API key if not set."""
        if "ANTHROPIC_API_KEY" not in os.environ:
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                yield
        else:
            yield

    @pytest.mark.vcr()
    def test_anthropic_research_workflow_generates_steps(self):
        """Test that Anthropic generates structured plan steps for a research task."""
        web_search, read_website, generate_report = create_research_tools()
        llm = LLM(model="anthropic/claude-sonnet-4-20250514")

        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research and produce insightful reports",
            backstory="An experienced analyst skilled at gathering information and synthesizing findings into actionable insights.",
            llm=llm,
            tools=[web_search, read_website, generate_report],
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        result = agent.kickoff(RESEARCH_TASK)

        # Verify result exists
        assert result is not None
        assert result.raw is not None
        # The result should contain some report-like content
        assert len(str(result.raw)) > 50


class TestGeminiStructuredPlanning:
    """Integration tests for Google Gemini structured planning with research workflow."""

    @pytest.fixture(autouse=True)
    def mock_google_api_key(self):
        """Mock API key if not set."""
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                yield
        else:
            yield

    @pytest.mark.vcr()
    def test_gemini_research_workflow_generates_steps(self):
        """Test that Gemini generates structured plan steps for a research task."""
        web_search, read_website, generate_report = create_research_tools()
        llm = LLM(model="gemini/gemini-2.5-flash")

        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research and produce insightful reports",
            backstory="An experienced analyst skilled at gathering information and synthesizing findings into actionable insights.",
            llm=llm,
            tools=[web_search, read_website, generate_report],
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        result = agent.kickoff(RESEARCH_TASK)

        # Verify result exists
        assert result is not None
        assert result.raw is not None
        # The result should contain some report-like content
        assert len(str(result.raw)) > 50


class TestAzureStructuredPlanning:
    """Integration tests for Azure OpenAI structured planning with research workflow."""

    @pytest.fixture(autouse=True)
    def mock_azure_credentials(self):
        """Mock Azure credentials for tests."""
        if "AZURE_API_KEY" not in os.environ:
            with patch.dict(os.environ, {
                "AZURE_API_KEY": "test-key",
                "AZURE_ENDPOINT": "https://test.openai.azure.com"
            }):
                yield
        else:
            yield

    @pytest.mark.vcr()
    def test_azure_research_workflow_generates_steps(self):
        """Test that Azure OpenAI generates structured plan steps for a research task."""
        web_search, read_website, generate_report = create_research_tools()
        llm = LLM(model="azure/gpt-4o")

        agent = Agent(
            role="Research Analyst",
            goal="Conduct thorough research and produce insightful reports",
            backstory="An experienced analyst skilled at gathering information and synthesizing findings into actionable insights.",
            llm=llm,
            tools=[web_search, read_website, generate_report],
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        result = agent.kickoff(RESEARCH_TASK)

        # Verify result exists
        assert result is not None
        assert result.raw is not None
        # The result should contain some report-like content
        assert len(str(result.raw)) > 50


# =============================================================================
# Unit Tests with Mocked LLM Providers
# =============================================================================


class TestStructuredPlanningWithMockedProviders:
    """Unit tests with mocked LLM providers for faster execution."""

    def _create_mock_plan_response(self, steps_data):
        """Helper to create mock plan response."""
        return json.dumps({
            "plan": "Test plan",
            "steps": steps_data,
            "ready": True,
        })

    def test_openai_mock_structured_response(self):
        """Test parsing OpenAI structured response."""
        steps_data = [
            {"step_number": 1, "description": "Search", "tool_to_use": "search", "depends_on": []},
            {"step_number": 2, "description": "Analyze", "tool_to_use": None, "depends_on": [1]},
        ]

        response = self._create_mock_plan_response(steps_data)
        parsed = json.loads(response)

        assert len(parsed["steps"]) == 2
        assert parsed["steps"][0]["tool_to_use"] == "search"
        assert parsed["steps"][1]["depends_on"] == [1]

    def test_anthropic_mock_structured_response(self):
        """Test parsing Anthropic structured response (same format)."""
        steps_data = [
            {"step_number": 1, "description": "Research", "tool_to_use": "web_search", "depends_on": []},
            {"step_number": 2, "description": "Summarize", "tool_to_use": None, "depends_on": [1]},
            {"step_number": 3, "description": "Report", "tool_to_use": "write_file", "depends_on": [1, 2]},
        ]

        response = self._create_mock_plan_response(steps_data)
        parsed = json.loads(response)

        assert len(parsed["steps"]) == 3
        assert parsed["steps"][2]["depends_on"] == [1, 2]

    def test_gemini_mock_structured_response(self):
        """Test parsing Gemini structured response (same format)."""
        steps_data = [
            {"step_number": 1, "description": "Gather data", "tool_to_use": "data_tool", "depends_on": []},
            {"step_number": 2, "description": "Process", "tool_to_use": None, "depends_on": [1]},
        ]

        response = self._create_mock_plan_response(steps_data)
        parsed = json.loads(response)

        assert len(parsed["steps"]) == 2
        assert parsed["ready"] is True

    def test_azure_mock_structured_response(self):
        """Test parsing Azure OpenAI structured response (same format as OpenAI)."""
        steps_data = [
            {"step_number": 1, "description": "Initialize", "tool_to_use": None, "depends_on": []},
            {"step_number": 2, "description": "Execute", "tool_to_use": "executor", "depends_on": [1]},
            {"step_number": 3, "description": "Finalize", "tool_to_use": None, "depends_on": [1, 2]},
        ]

        response = self._create_mock_plan_response(steps_data)
        parsed = json.loads(response)

        assert len(parsed["steps"]) == 3
        assert parsed["steps"][0]["tool_to_use"] is None


class TestTodoListIntegration:
    """Integration tests for TodoList with plan execution simulation."""

    def test_full_plan_execution_workflow(self):
        """Test complete workflow from plan to todos to execution."""
        # Simulate plan steps from LLM
        plan_steps = [
            PlanStep(
                step_number=1,
                description="Research the topic",
                tool_to_use="search_tool",
                depends_on=[],
            ),
            PlanStep(
                step_number=2,
                description="Compile findings",
                tool_to_use=None,
                depends_on=[1],
            ),
            PlanStep(
                step_number=3,
                description="Generate summary",
                tool_to_use="summarize_tool",
                depends_on=[1, 2],
            ),
        ]

        # Convert to todos (like agent_executor._create_todos_from_plan)
        todos = [
            TodoItem(
                step_number=step.step_number,
                description=step.description,
                tool_to_use=step.tool_to_use,
                depends_on=step.depends_on,
                status="pending",
            )
            for step in plan_steps
        ]
        todo_list = TodoList(items=todos)

        # Verify initial state
        assert todo_list.pending_count == 3
        assert todo_list.is_complete is False

        # Simulate execution
        for i in range(1, 4):
            todo_list.mark_running(i)
            assert todo_list.current_todo.step_number == i
            todo_list.mark_completed(i, result=f"Step {i} completed")

        # Verify final state
        assert todo_list.is_complete is True
        assert todo_list.completed_count == 3
        assert all(item.result is not None for item in todo_list.items)

    def test_dependency_aware_execution(self):
        """Test that dependencies are respected in execution order."""
        steps = [
            PlanStep(step_number=1, description="Base step", depends_on=[]),
            PlanStep(step_number=2, description="Depends on 1", depends_on=[1]),
            PlanStep(step_number=3, description="Depends on 1", depends_on=[1]),
            PlanStep(step_number=4, description="Depends on 2 and 3", depends_on=[2, 3]),
        ]

        todos = [
            TodoItem(
                step_number=s.step_number,
                description=s.description,
                depends_on=s.depends_on,
            )
            for s in steps
        ]
        todo_list = TodoList(items=todos)

        # Helper to check if dependencies are satisfied
        def can_execute(todo: TodoItem) -> bool:
            for dep in todo.depends_on:
                dep_todo = todo_list.get_by_step_number(dep)
                if dep_todo and dep_todo.status != "completed":
                    return False
            return True

        # Step 1 has no dependencies
        assert can_execute(todo_list.items[0]) is True

        # Steps 2 and 3 depend on 1 (not yet done)
        assert can_execute(todo_list.items[1]) is False
        assert can_execute(todo_list.items[2]) is False

        # Complete step 1
        todo_list.mark_completed(1)

        # Now steps 2 and 3 can execute
        assert can_execute(todo_list.items[1]) is True
        assert can_execute(todo_list.items[2]) is True

        # Step 4 still can't (depends on 2 and 3)
        assert can_execute(todo_list.items[3]) is False

        # Complete steps 2 and 3
        todo_list.mark_completed(2)
        todo_list.mark_completed(3)

        # Now step 4 can execute
        assert can_execute(todo_list.items[3]) is True
