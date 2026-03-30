"""Tests for trace serialization optimization to prevent trace table bloat.

These tests verify that trace events don't contain redundant full crew/task/agent
objects, reducing event sizes from 50-100KB to a few KB per event.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from crewai.events.listeners.tracing.trace_listener import (
    TRACE_EXCLUDE_FIELDS,
    TraceCollectionListener,
    _serialize_for_trace,
)


class TestTraceExcludeFields:
    """Test that TRACE_EXCLUDE_FIELDS contains all the heavy/redundant fields."""

    def test_contains_back_references(self):
        """Verify back-reference fields are excluded."""
        back_refs = {"crew", "agent", "agents", "tasks", "context"}
        assert back_refs.issubset(TRACE_EXCLUDE_FIELDS)

    def test_contains_heavy_fields(self):
        """Verify heavy objects are excluded.

        Note: 'tools' is NOT in TRACE_EXCLUDE_FIELDS because LLMCallStartedEvent.tools
        is a lightweight list of tool schemas. Agent.tools exclusion is handled
        explicitly in _build_crew_started_data.
        """
        heavy_fields = {
            "llm",
            "function_calling_llm",
            "step_callback",
            "task_callback",
            "crew_callback",
            "callbacks",
            "_memory",
            "_cache",
            "knowledge_sources",
        }
        assert heavy_fields.issubset(TRACE_EXCLUDE_FIELDS)
        # tools is NOT excluded globally - LLM events need it
        assert "tools" not in TRACE_EXCLUDE_FIELDS


class TestSerializeForTrace:
    """Test the _serialize_for_trace helper function."""

    def test_excludes_crew_field(self):
        """Verify crew field is excluded from serialization."""
        event = MagicMock()
        event.crew = MagicMock(name="TestCrew")
        event.crew_name = "TestCrew"
        event.timestamp = None

        result = _serialize_for_trace(event)

        # crew_name should be present (scalar field)
        # crew should be excluded (back-reference)
        assert "crew" not in result or result.get("crew") is None

    def test_excludes_agent_field(self):
        """Verify agent field is excluded from serialization."""
        event = MagicMock()
        event.agent = MagicMock(role="TestAgent")
        event.agent_role = "TestAgent"

        result = _serialize_for_trace(event)

        assert "agent" not in result or result.get("agent") is None

    def test_preserves_tools_field(self):
        """Verify tools field is preserved for LLM events (lightweight schemas)."""

        class EventWithTools:
            def __init__(self):
                self.tools = [{"name": "search", "description": "Search tool"}]
                self.tool_name = "test_tool"

        event = EventWithTools()
        result = _serialize_for_trace(event)

        # tools should be preserved (lightweight for LLM events)
        assert "tools" in result
        assert result["tools"] == [{"name": "search", "description": "Search tool"}]

    def test_preserves_scalar_fields(self):
        """Verify scalar fields needed by AMP frontend are preserved."""

        class SimpleEvent:
            def __init__(self):
                self.agent_role = "Researcher"
                self.task_name = "Research Task"
                self.task_id = str(uuid.uuid4())
                self.duration_ms = 1500
                self.tokens_used = 500

        event = SimpleEvent()
        result = _serialize_for_trace(event)

        # Scalar fields should be preserved
        assert result.get("agent_role") == "Researcher"
        assert result.get("task_name") == "Research Task"
        assert result.get("duration_ms") == 1500
        assert result.get("tokens_used") == 500

    def test_extra_exclude_parameter(self):
        """Verify extra_exclude adds to the default exclusions."""

        class EventWithCustomField:
            def __init__(self):
                self.custom_heavy_field = {"large": "data" * 1000}
                self.keep_this = "small"

        event = EventWithCustomField()
        result = _serialize_for_trace(event, extra_exclude={"custom_heavy_field"})

        assert "custom_heavy_field" not in result
        assert result.get("keep_this") == "small"


class TestBuildEventData:
    """Test _build_event_data method for different event types."""

    @pytest.fixture
    def listener(self):
        """Create a trace listener for testing."""
        # Reset singleton
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_task_started_no_full_task_object(self, listener):
        """Verify task_started event doesn't include full task object."""
        mock_task = MagicMock()
        mock_task.name = "Test Task"
        mock_task.description = "A test task description"
        mock_task.expected_output = "Expected result"
        mock_task.id = uuid.uuid4()
        # Add heavy fields that should NOT appear in output
        mock_task.crew = MagicMock(name="HeavyCrew")
        mock_task.agent = MagicMock(role="HeavyAgent")
        mock_task.context = [MagicMock(), MagicMock()]
        mock_task.tools = [MagicMock(), MagicMock()]

        mock_event = MagicMock()
        mock_event.task = mock_task
        mock_event.context = "test context"

        mock_source = MagicMock()
        mock_source.agent = MagicMock()
        mock_source.agent.role = "Worker"

        result = listener._build_event_data("task_started", mock_event, mock_source)

        # Should have scalar fields
        assert result["task_name"] == "Test Task"
        assert result["task_description"] == "A test task description"
        assert result["agent_role"] == "Worker"
        assert result["task_id"] == str(mock_task.id)

        # Should NOT have full objects
        assert "crew" not in result
        assert "tools" not in result
        # task and agent should not be full objects
        assert result.get("task") is None or not hasattr(result.get("task"), "crew")

    def test_task_completed_no_full_task_object(self, listener):
        """Verify task_completed event doesn't include full task object."""
        mock_task = MagicMock()
        mock_task.name = "Completed Task"
        mock_task.description = "Task description"
        mock_task.id = uuid.uuid4()

        mock_output = MagicMock()
        mock_output.raw = "Task result"
        mock_output.output_format = "text"
        mock_output.agent = "Worker"

        mock_event = MagicMock()
        mock_event.task = mock_task
        mock_event.output = mock_output

        result = listener._build_event_data("task_completed", mock_event, None)

        # Should have scalar fields
        assert result["task_name"] == "Completed Task"
        assert result["output_raw"] == "Task result"
        assert result["agent_role"] == "Worker"

        # Should NOT have full task object
        assert "crew" not in result
        assert "tools" not in result

    def test_agent_execution_started_no_full_agent(self, listener):
        """Verify agent_execution_started extracts only scalar fields."""
        mock_agent = MagicMock()
        mock_agent.role = "Analyst"
        mock_agent.goal = "Analyze data"
        mock_agent.backstory = "Expert analyst"
        # Heavy fields
        mock_agent.tools = [MagicMock(), MagicMock()]
        mock_agent.llm = MagicMock()
        mock_agent.crew = MagicMock()

        mock_event = MagicMock()
        mock_event.agent = mock_agent

        result = listener._build_event_data(
            "agent_execution_started", mock_event, None
        )

        # Should have scalar fields
        assert result["agent_role"] == "Analyst"
        assert result["agent_goal"] == "Analyze data"
        assert result["agent_backstory"] == "Expert analyst"

        # Should NOT have heavy objects
        assert "tools" not in result
        assert "llm" not in result
        assert "crew" not in result

    def test_llm_call_started_excludes_heavy_fields(self, listener):
        """Verify llm_call_started uses lightweight serialization.

        LLMCallStartedEvent.tools is a lightweight list of tool schemas (dicts),
        not heavy Agent.tools objects, so it should be preserved.
        """

        class MockLLMEvent:
            def __init__(self):
                self.task_name = "LLM Task"
                self.model = "gpt-4"
                self.tokens = 100
                # Heavy fields that should be excluded
                self.crew = MagicMock()
                self.agent = MagicMock()
                # LLM event tools are lightweight schemas (dicts), should be kept
                self.tools = [{"name": "search", "description": "Search tool"}]

        mock_event = MockLLMEvent()

        result = listener._build_event_data("llm_call_started", mock_event, None)

        # task_name should be present
        assert result["task_name"] == "LLM Task"

        # Heavy fields should be excluded
        assert "crew" not in result or result.get("crew") is None
        assert "agent" not in result or result.get("agent") is None
        # LLM tools (lightweight schemas) should be preserved
        assert result.get("tools") == [{"name": "search", "description": "Search tool"}]

    def test_llm_call_completed_excludes_heavy_fields(self, listener):
        """Verify llm_call_completed uses lightweight serialization."""

        class MockLLMCompletedEvent:
            def __init__(self):
                self.response = "LLM response"
                self.tokens_used = 150
                self.duration_ms = 500
                # Heavy fields
                self.crew = MagicMock()
                self.agent = MagicMock()

        mock_event = MockLLMCompletedEvent()

        result = listener._build_event_data("llm_call_completed", mock_event, None)

        # Scalar fields preserved
        assert result.get("response") == "LLM response"
        assert result.get("tokens_used") == 150

        # Heavy fields excluded
        assert "crew" not in result or result.get("crew") is None
        assert "agent" not in result or result.get("agent") is None


class TestCrewKickoffStartedEvent:
    """Test that crew_kickoff_started event has full structure."""

    @pytest.fixture
    def listener(self):
        """Create a trace listener for testing."""
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_crew_started_has_crew_structure(self, listener):
        """Verify crew_kickoff_started includes the crew_structure field."""
        # Create mock crew with agents and tasks
        mock_agent1 = MagicMock()
        mock_agent1.id = uuid.uuid4()
        mock_agent1.role = "Researcher"
        mock_agent1.goal = "Research things"
        mock_agent1.backstory = "Expert researcher"
        mock_agent1.verbose = True
        mock_agent1.allow_delegation = False
        mock_agent1.max_iter = 10
        mock_agent1.max_rpm = None
        mock_agent1.tools = [MagicMock(name="search_tool"), MagicMock(name="read_tool")]

        mock_agent2 = MagicMock()
        mock_agent2.id = uuid.uuid4()
        mock_agent2.role = "Writer"
        mock_agent2.goal = "Write content"
        mock_agent2.backstory = "Expert writer"
        mock_agent2.verbose = False
        mock_agent2.allow_delegation = True
        mock_agent2.max_iter = 5
        mock_agent2.max_rpm = 10
        mock_agent2.tools = []

        mock_task1 = MagicMock()
        mock_task1.id = uuid.uuid4()
        mock_task1.name = "Research Task"
        mock_task1.description = "Do research"
        mock_task1.expected_output = "Research results"
        mock_task1.async_execution = False
        mock_task1.human_input = False
        mock_task1.agent = mock_agent1
        mock_task1.context = None

        mock_task2 = MagicMock()
        mock_task2.id = uuid.uuid4()
        mock_task2.name = "Writing Task"
        mock_task2.description = "Write report"
        mock_task2.expected_output = "Written report"
        mock_task2.async_execution = True
        mock_task2.human_input = True
        mock_task2.agent = mock_agent2
        mock_task2.context = [mock_task1]

        mock_crew = MagicMock()
        mock_crew.agents = [mock_agent1, mock_agent2]
        mock_crew.tasks = [mock_task1, mock_task2]
        mock_crew.process = "sequential"
        mock_crew.verbose = True
        mock_crew.memory = False

        mock_event = MagicMock()
        mock_event.crew = mock_crew
        mock_event.crew_name = "TestCrew"
        mock_event.inputs = {"key": "value"}

        result = listener._build_event_data("crew_kickoff_started", mock_event, None)

        # Should have crew_structure
        assert "crew_structure" in result
        crew_structure = result["crew_structure"]

        # Verify agents are serialized with tool names
        assert len(crew_structure["agents"]) == 2
        agent1_data = crew_structure["agents"][0]
        assert agent1_data["role"] == "Researcher"
        assert agent1_data["goal"] == "Research things"
        assert "tool_names" in agent1_data
        assert len(agent1_data["tool_names"]) == 2

        # Verify tasks have lightweight agent references
        assert len(crew_structure["tasks"]) == 2
        task2_data = crew_structure["tasks"][1]
        assert task2_data["name"] == "Writing Task"
        assert "agent_ref" in task2_data
        assert task2_data["agent_ref"]["role"] == "Writer"

        # Verify context uses task IDs
        assert "context_task_ids" in task2_data
        assert str(mock_task1.id) in task2_data["context_task_ids"]

    def test_crew_started_agents_no_full_tools(self, listener):
        """Verify agents in crew_structure have tool_names, not full tool objects."""
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.description = "Search the web"
        mock_tool.func = lambda x: x  # Heavy callable
        mock_tool.args_schema = {"type": "object"}  # Schema

        mock_agent = MagicMock()
        mock_agent.id = uuid.uuid4()
        mock_agent.role = "Searcher"
        mock_agent.goal = "Search"
        mock_agent.backstory = "Expert"
        mock_agent.verbose = False
        mock_agent.allow_delegation = False
        mock_agent.max_iter = 5
        mock_agent.max_rpm = None
        mock_agent.tools = [mock_tool]

        mock_crew = MagicMock()
        mock_crew.agents = [mock_agent]
        mock_crew.tasks = []
        mock_crew.process = "sequential"
        mock_crew.verbose = False
        mock_crew.memory = False

        mock_event = MagicMock()
        mock_event.crew = mock_crew

        result = listener._build_event_data("crew_kickoff_started", mock_event, None)

        agent_data = result["crew_structure"]["agents"][0]

        # Should have tool_names (list of strings)
        assert "tool_names" in agent_data
        assert agent_data["tool_names"] == ["web_search"]

        # Should NOT have full tools array
        assert "tools" not in agent_data

    def test_crew_started_tasks_no_full_agent(self, listener):
        """Verify tasks have agent_ref, not full agent object."""
        mock_agent = MagicMock()
        mock_agent.id = uuid.uuid4()
        mock_agent.role = "Worker"
        mock_agent.goal = "Work hard"
        mock_agent.backstory = "Dedicated worker"
        mock_agent.tools = [MagicMock(), MagicMock()]
        mock_agent.llm = MagicMock()

        mock_task = MagicMock()
        mock_task.id = uuid.uuid4()
        mock_task.name = "Work Task"
        mock_task.description = "Do work"
        mock_task.expected_output = "Work done"
        mock_task.async_execution = False
        mock_task.human_input = False
        mock_task.agent = mock_agent
        mock_task.context = None

        mock_crew = MagicMock()
        mock_crew.agents = [mock_agent]
        mock_crew.tasks = [mock_task]
        mock_crew.process = "sequential"
        mock_crew.verbose = False
        mock_crew.memory = False

        mock_event = MagicMock()
        mock_event.crew = mock_crew

        result = listener._build_event_data("crew_kickoff_started", mock_event, None)

        task_data = result["crew_structure"]["tasks"][0]

        # Should have lightweight agent_ref
        assert "agent_ref" in task_data
        assert task_data["agent_ref"]["id"] == str(mock_agent.id)
        assert task_data["agent_ref"]["role"] == "Worker"

        # agent_ref should ONLY have id and role (not tools, llm, etc.)
        assert len(task_data["agent_ref"]) == 2

        # Should NOT have full agent
        assert "agent" not in task_data


class TestNonCrewStartedEvents:
    """Test that non-crew_started events don't have redundant data."""

    @pytest.fixture
    def listener(self):
        """Create a trace listener for testing."""
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_generic_event_no_crew(self, listener):
        """Verify generic events exclude crew object.

        Note: 'tools' is now preserved since LLMCallStartedEvent.tools is lightweight.
        """

        class GenericEvent:
            def __init__(self):
                self.event_type = "some_event"
                self.data = "some_data"
                # These should be excluded
                self.crew = MagicMock()
                self.agents = [MagicMock()]
                self.tasks = [MagicMock()]
                # tools is now preserved (for LLM events it's lightweight)
                self.tools = [{"name": "search"}]

        mock_event = GenericEvent()

        result = listener._build_event_data("some_event", mock_event, None)

        # Scalar fields preserved
        assert result.get("event_type") == "some_event"
        assert result.get("data") == "some_data"

        # Heavy fields excluded
        assert "crew" not in result or result.get("crew") is None
        assert "agents" not in result or result.get("agents") is None
        assert "tasks" not in result or result.get("tasks") is None
        # tools is now preserved (lightweight for LLM events)
        assert result.get("tools") == [{"name": "search"}]

    def test_crew_kickoff_completed_no_full_crew(self, listener):
        """Verify crew_kickoff_completed doesn't repeat full crew structure."""

        class CrewCompletedEvent:
            def __init__(self):
                self.crew_name = "TestCrew"
                self.total_tokens = 5000
                self.output = "Final output"
                # Should be excluded
                self.crew = MagicMock()
                self.crew.agents = [MagicMock(), MagicMock()]
                self.crew.tasks = [MagicMock()]

        mock_event = CrewCompletedEvent()

        result = listener._build_event_data("crew_kickoff_completed", mock_event, None)

        # Scalar fields preserved
        assert result.get("crew_name") == "TestCrew"
        assert result.get("total_tokens") == 5000

        # Should NOT have full crew object
        assert "crew" not in result or result.get("crew") is None
        # Should NOT have crew_structure (that's only for crew_started)
        assert "crew_structure" not in result


class TestSizeReduction:
    """Test that the optimization actually reduces serialized size."""

    @pytest.fixture
    def listener(self):
        """Create a trace listener for testing."""
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_task_event_size_reduction(self, listener):
        """Verify task events are much smaller than naive serialization."""
        import json

        # Create a realistic task with many fields
        mock_agent = MagicMock()
        mock_agent.id = uuid.uuid4()
        mock_agent.role = "Researcher"
        mock_agent.goal = "Research" * 50  # Longer goal
        mock_agent.backstory = "Expert" * 100  # Longer backstory
        mock_agent.tools = [MagicMock() for _ in range(5)]
        mock_agent.llm = MagicMock()
        mock_agent.crew = MagicMock()

        mock_task = MagicMock()
        mock_task.name = "Research Task"
        mock_task.description = "Detailed description" * 20
        mock_task.expected_output = "Expected" * 10
        mock_task.id = uuid.uuid4()
        mock_task.agent = mock_agent
        mock_task.context = [MagicMock() for _ in range(3)]
        mock_task.crew = MagicMock()
        mock_task.tools = [MagicMock() for _ in range(3)]

        mock_event = MagicMock()
        mock_event.task = mock_task
        mock_event.context = "test context"

        mock_source = MagicMock()
        mock_source.agent = mock_agent

        result = listener._build_event_data("task_started", mock_event, mock_source)

        # The result should be relatively small
        serialized = json.dumps(result, default=str)

        # Should be under 2KB for task_started (was potentially 50-100KB before)
        assert len(serialized) < 2000, f"task_started too large: {len(serialized)} bytes"

        # Should have the essential fields
        assert "task_name" in result
        assert "task_id" in result
        assert "agent_role" in result
