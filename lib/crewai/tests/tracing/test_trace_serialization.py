"""Tests for trace serialization optimization using Pydantic v2 context-based serialization.

These tests verify that trace events use @field_serializer with SerializationInfo.context
to produce lightweight representations, reducing event sizes from 50-100KB to a few KB.
"""

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ConfigDict

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.base_events import _trace_agent_ref, _trace_task_ref, _trace_tool_names
from crewai.events.listeners.tracing.utils import safe_serialize_to_dict
from crewai.security import SecurityConfig
from crewai.utilities.serialization import to_serializable


# ---------------------------------------------------------------------------
# Lightweight BaseAgent subclass for tests (avoids heavy dependencies)
# ---------------------------------------------------------------------------

class _StubAgent(BaseAgent):
    """Minimal BaseAgent subclass that satisfies validation without heavy deps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute_task(self, *a: Any, **kw: Any) -> str:
        return ""

    def create_agent_executor(self, *a: Any, **kw: Any) -> None:
        pass

    def _parse_tools(self, *a: Any, **kw: Any) -> list:
        return []

    def get_delegation_tools(self, *a: Any, **kw: Any) -> list:
        return []

    def get_output_converter(self, *a: Any, **kw: Any) -> Any:
        return None

    def get_multimodal_tools(self, *a: Any, **kw: Any) -> list:
        return []

    async def aexecute_task(self, *a: Any, **kw: Any) -> str:
        return ""

    def get_mcp_tools(self, *a: Any, **kw: Any) -> list:
        return []

    def get_platform_tools(self, *a: Any, **kw: Any) -> list:
        return []


def _make_stub_agent(**overrides) -> _StubAgent:
    """Create a minimal BaseAgent instance for testing."""
    defaults = {
        "role": "Researcher",
        "goal": "Research things",
        "backstory": "Expert researcher",
        "tools": [],
    }
    defaults.update(overrides)
    return _StubAgent(**defaults)


# ---------------------------------------------------------------------------
# Helpers to build realistic mock objects for event fields
# ---------------------------------------------------------------------------

def _make_mock_task(**overrides):
    task = MagicMock()
    task.id = overrides.get("id", uuid.uuid4())
    task.name = overrides.get("name", "Research Task")
    task.description = overrides.get("description", "Do research")
    task.expected_output = overrides.get("expected_output", "Research results")
    task.async_execution = overrides.get("async_execution", False)
    task.human_input = overrides.get("human_input", False)
    task.agent = overrides.get("agent", _make_stub_agent())
    task.context = overrides.get("context", None)
    task.crew = MagicMock()
    task.tools = overrides.get("tools", [MagicMock(), MagicMock()])

    fp = MagicMock()
    fp.uuid_str = str(uuid.uuid4())
    fp.metadata = {"name": task.name}
    task.fingerprint = fp

    return task


def _make_stub_tool(tool_name="web_search") -> Any:
    """Create a minimal BaseTool instance for testing."""
    from crewai.tools.base_tool import BaseTool

    class _StubTool(BaseTool):
        name: str = "stub"
        description: str = "stub tool"

        def _run(self, *a: Any, **kw: Any) -> str:
            return ""

    return _StubTool(name=tool_name, description=f"{tool_name} tool")


# ---------------------------------------------------------------------------
# Unit tests: trace ref helpers
# ---------------------------------------------------------------------------

class TestTraceRefHelpers:
    def test_trace_agent_ref(self):
        agent = _make_stub_agent(role="Analyst")
        ref = _trace_agent_ref(agent)
        assert ref["role"] == "Analyst"
        assert "id" in ref
        assert len(ref) == 2  # only id and role

    def test_trace_agent_ref_none(self):
        assert _trace_agent_ref(None) is None

    def test_trace_task_ref(self):
        task = _make_mock_task(name="Write Report")
        ref = _trace_task_ref(task)
        assert ref["name"] == "Write Report"
        assert "id" in ref
        assert len(ref) == 2

    def test_trace_task_ref_falls_back_to_description(self):
        task = _make_mock_task(name=None, description="Describe the report")
        ref = _trace_task_ref(task)
        assert ref["name"] == "Describe the report"

    def test_trace_task_ref_none(self):
        assert _trace_task_ref(None) is None

    def test_trace_tool_names(self):
        tools = [_make_stub_tool("search"), _make_stub_tool("read")]
        names = _trace_tool_names(tools)
        assert names == ["search", "read"]

    def test_trace_tool_names_empty(self):
        assert _trace_tool_names([]) is None
        assert _trace_tool_names(None) is None


# ---------------------------------------------------------------------------
# Integration tests: field serializers on real event classes
# ---------------------------------------------------------------------------

class TestAgentEventFieldSerializers:
    """Test that agent event field serializers respond to trace context."""

    def test_agent_execution_started_trace_context(self):
        from crewai.events.types.agent_events import AgentExecutionStartedEvent

        agent = _make_stub_agent(role="Researcher")
        task = _make_mock_task(name="Research Task")
        tools = [_make_stub_tool("search"), _make_stub_tool("read")]

        event = AgentExecutionStartedEvent(
            agent=agent, task=task, tools=tools, task_prompt="Do research"
        )

        # With trace context: lightweight refs
        trace_dump = event.model_dump(context={"trace": True})
        assert trace_dump["agent"] == {"id": str(agent.id), "role": "Researcher"}
        assert trace_dump["task"] == {"id": str(task.id), "name": "Research Task"}
        assert trace_dump["tools"] == ["search", "read"]

    def test_agent_execution_started_no_context(self):
        from crewai.events.types.agent_events import AgentExecutionStartedEvent

        agent = _make_stub_agent(role="SpecificRole")
        task = _make_mock_task()

        event = AgentExecutionStartedEvent(
            agent=agent, task=task, tools=None, task_prompt="Do research"
        )

        # Without context: full agent dict (Pydantic model_dump expands it)
        normal_dump = event.model_dump()
        assert isinstance(normal_dump["agent"], dict)
        assert normal_dump["agent"]["role"] == "SpecificRole"
        # Should have ALL agent fields, not just the lightweight ref
        assert "goal" in normal_dump["agent"]
        assert "backstory" in normal_dump["agent"]
        assert "max_iter" in normal_dump["agent"]

    def test_agent_execution_error_preserves_identification(self):
        from crewai.events.types.agent_events import AgentExecutionErrorEvent

        agent = _make_stub_agent(role="Analyst")
        task = _make_mock_task(name="Analysis Task")

        event = AgentExecutionErrorEvent(
            agent=agent, task=task, error="Something went wrong"
        )

        trace_dump = event.model_dump(context={"trace": True})
        # Error events should still have agent/task identification as refs
        assert trace_dump["agent"]["role"] == "Analyst"
        assert trace_dump["task"]["name"] == "Analysis Task"
        assert trace_dump["error"] == "Something went wrong"

    def test_agent_execution_completed_trace_context(self):
        from crewai.events.types.agent_events import AgentExecutionCompletedEvent

        agent = _make_stub_agent(role="Writer")
        task = _make_mock_task(name="Writing Task")

        event = AgentExecutionCompletedEvent(
            agent=agent, task=task, output="Final output"
        )

        trace_dump = event.model_dump(context={"trace": True})
        assert trace_dump["agent"]["role"] == "Writer"
        assert trace_dump["task"]["name"] == "Writing Task"
        assert trace_dump["output"] == "Final output"


class TestTaskEventFieldSerializers:
    """Test that task event field serializers respond to trace context."""

    def test_task_started_trace_context(self):
        from crewai.events.types.task_events import TaskStartedEvent

        task = _make_mock_task(name="Test Task")
        event = TaskStartedEvent(task=task, context="some context")

        trace_dump = event.model_dump(context={"trace": True})
        assert trace_dump["task"] == {"id": str(task.id), "name": "Test Task"}
        assert trace_dump["context"] == "some context"

    def test_task_failed_trace_context(self):
        from crewai.events.types.task_events import TaskFailedEvent

        task = _make_mock_task(name="Failing Task")
        event = TaskFailedEvent(task=task, error="Task failed")

        trace_dump = event.model_dump(context={"trace": True})
        assert trace_dump["task"]["name"] == "Failing Task"
        assert trace_dump["error"] == "Task failed"


class TestCrewEventFieldSerializers:
    """Test that crew event field serializers respond to trace context."""

    def test_crew_kickoff_started_excludes_crew_in_trace(self):
        from crewai.events.types.crew_events import CrewKickoffStartedEvent

        crew = MagicMock()
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        event = CrewKickoffStartedEvent(
            crew=crew, crew_name="TestCrew", inputs={"key": "value"}
        )

        trace_dump = event.model_dump(context={"trace": True})
        # crew field should be None in trace context
        assert trace_dump["crew"] is None
        # scalar fields preserved
        assert trace_dump["crew_name"] == "TestCrew"
        assert trace_dump["inputs"] == {"key": "value"}

    def test_crew_event_no_context_preserves_crew(self):
        from crewai.events.types.crew_events import CrewKickoffStartedEvent

        crew = MagicMock()
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        event = CrewKickoffStartedEvent(
            crew=crew, crew_name="TestCrew", inputs=None
        )

        normal_dump = event.model_dump()
        # Without trace context, crew should NOT be None (field serializer didn't fire)
        assert normal_dump["crew"] is not None


class TestLLMEventFieldSerializers:
    """Test that LLM event field serializers respond to trace context."""

    def test_llm_call_started_excludes_callbacks_in_trace(self):
        from crewai.events.types.llm_events import LLMCallStartedEvent

        event = LLMCallStartedEvent(
            call_id="test-call",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "search", "description": "Search tool"}],
            callbacks=[MagicMock(), MagicMock()],
            available_functions={"search": MagicMock()},
        )

        trace_dump = event.model_dump(context={"trace": True})
        # callbacks and available_functions excluded
        assert trace_dump["callbacks"] is None
        assert trace_dump["available_functions"] is None
        # tools preserved (lightweight list of dicts)
        assert trace_dump["tools"] == [{"name": "search", "description": "Search tool"}]
        # messages preserved
        assert trace_dump["messages"] == [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Integration tests: safe_serialize_to_dict with context
# ---------------------------------------------------------------------------

class TestSafeSerializeWithContext:
    """Test that safe_serialize_to_dict properly passes context through."""

    def test_context_flows_through_to_field_serializers(self):
        from crewai.events.types.agent_events import AgentExecutionErrorEvent

        agent = _make_stub_agent(role="Worker")
        task = _make_mock_task(name="Work Task")

        event = AgentExecutionErrorEvent(
            agent=agent, task=task, error="error msg"
        )

        result = safe_serialize_to_dict(event, context={"trace": True})
        # Field serializers should have fired
        assert result["agent"] == {"id": str(agent.id), "role": "Worker"}
        assert result["task"] == {"id": str(task.id), "name": "Work Task"}
        assert result["error"] == "error msg"

    def test_no_context_preserves_full_serialization(self):
        from crewai.events.types.task_events import TaskFailedEvent

        task = _make_mock_task(name="Test")
        event = TaskFailedEvent(task=task, error="fail")

        result = safe_serialize_to_dict(event)
        # Without context, task should not be a lightweight ref
        assert result.get("task") is not None
        # It should be the raw object (model_dump returns it as-is for Any fields)
        # to_serializable will then repr() or process it further


# ---------------------------------------------------------------------------
# Integration tests: TraceCollectionListener._build_event_data
# ---------------------------------------------------------------------------

class TestBuildEventData:
    @pytest.fixture
    def listener(self):
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_crew_kickoff_started_has_crew_structure(self, listener):
        agent = _make_stub_agent(role="Researcher")
        agent.tools = [_make_stub_tool("search"), _make_stub_tool("read")]

        task = _make_mock_task(name="Research Task", agent=agent)
        task.context = None

        crew = MagicMock()
        crew.agents = [agent]
        crew.tasks = [task]
        crew.process = "sequential"
        crew.verbose = True
        crew.memory = False
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        from crewai.events.types.crew_events import CrewKickoffStartedEvent
        event = CrewKickoffStartedEvent(
            crew=crew, crew_name="TestCrew", inputs={"key": "value"}
        )

        result = listener._build_event_data("crew_kickoff_started", event, None)

        assert "crew_structure" in result
        cs = result["crew_structure"]
        assert len(cs["agents"]) == 1
        assert cs["agents"][0]["role"] == "Researcher"
        assert cs["agents"][0]["tool_names"] == ["search", "read"]
        assert len(cs["tasks"]) == 1
        assert cs["tasks"][0]["name"] == "Research Task"
        assert "agent_ref" in cs["tasks"][0]
        assert cs["tasks"][0]["agent_ref"]["role"] == "Researcher"

    def test_crew_kickoff_started_context_task_ids(self, listener):
        agent = _make_stub_agent()
        task1 = _make_mock_task(name="Task 1", agent=agent)
        task1.context = None
        task2 = _make_mock_task(name="Task 2", agent=agent)
        task2.context = [task1]

        crew = MagicMock()
        crew.agents = [agent]
        crew.tasks = [task1, task2]
        crew.process = "sequential"
        crew.verbose = False
        crew.memory = False
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        from crewai.events.types.crew_events import CrewKickoffStartedEvent
        event = CrewKickoffStartedEvent(
            crew=crew, crew_name="TestCrew", inputs=None
        )

        result = listener._build_event_data("crew_kickoff_started", event, None)
        task2_data = result["crew_structure"]["tasks"][1]
        assert "context_task_ids" in task2_data
        assert str(task1.id) in task2_data["context_task_ids"]

    def test_generic_event_uses_trace_context(self, listener):
        """Non-complex events should use context-based serialization."""
        from crewai.events.types.crew_events import CrewKickoffCompletedEvent

        crew = MagicMock()
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        event = CrewKickoffCompletedEvent(
            crew=crew, crew_name="TestCrew", output="Final result", total_tokens=5000
        )

        result = listener._build_event_data("crew_kickoff_completed", event, None)

        # Scalar fields preserved
        assert result.get("crew_name") == "TestCrew"
        assert result.get("total_tokens") == 5000
        # crew excluded by field serializer
        assert result.get("crew") is None
        # No crew_structure (that's only for kickoff_started)
        assert "crew_structure" not in result

    def test_task_started_custom_projection(self, listener):
        task = _make_mock_task(name="Test Task")
        from crewai.events.types.task_events import TaskStartedEvent
        event = TaskStartedEvent(task=task, context="test context")
        source = MagicMock()
        source.agent = _make_stub_agent(role="Worker")

        result = listener._build_event_data("task_started", event, source)

        assert result["task_name"] == "Test Task"
        assert result["agent_role"] == "Worker"
        assert result["task_id"] == str(task.id)
        assert result["context"] == "test context"

    def test_llm_call_started_uses_trace_context(self, listener):
        from crewai.events.types.llm_events import LLMCallStartedEvent

        event = LLMCallStartedEvent(
            call_id="test",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "search"}],
            callbacks=[MagicMock()],
            available_functions={"fn": MagicMock()},
        )

        result = listener._build_event_data("llm_call_started", event, None)

        # callbacks and available_functions excluded via field serializer
        assert result.get("callbacks") is None
        assert result.get("available_functions") is None
        # tools preserved (lightweight schemas)
        assert result.get("tools") == [{"name": "search"}]

    def test_agent_execution_error_preserves_identification(self, listener):
        """Error events should preserve agent/task identification via field serializers."""
        from crewai.events.types.agent_events import AgentExecutionErrorEvent

        agent = _make_stub_agent(role="Analyst")
        task = _make_mock_task(name="Analysis")

        event = AgentExecutionErrorEvent(
            agent=agent, task=task, error="Something broke"
        )

        result = listener._build_event_data("agent_execution_error", event, None)

        # Field serializers return lightweight refs, not None
        assert result["agent"] == {"id": str(agent.id), "role": "Analyst"}
        assert result["task"] == {"id": str(task.id), "name": "Analysis"}
        assert result["error"] == "Something broke"

    def test_task_failed_preserves_identification(self, listener):
        from crewai.events.types.task_events import TaskFailedEvent

        task = _make_mock_task(name="Failed Task")
        event = TaskFailedEvent(task=task, error="Task failed")

        result = listener._build_event_data("task_failed", event, None)

        assert result["task"] == {"id": str(task.id), "name": "Failed Task"}
        assert result["error"] == "Task failed"


# ---------------------------------------------------------------------------
# Size reduction verification
# ---------------------------------------------------------------------------

class TestSizeReduction:
    @pytest.fixture
    def listener(self):
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False
        return TraceCollectionListener()

    def test_task_started_event_size(self, listener):
        """task_started event data should be well under 2KB."""
        agent = _make_stub_agent(
            role="Researcher",
            goal="Research" * 50,
            backstory="Expert" * 100,
        )
        agent.tools = [_make_stub_tool(f"tool_{i}") for i in range(5)]

        task = _make_mock_task(
            name="Research Task",
            description="Detailed description" * 20,
            expected_output="Expected" * 10,
            agent=agent,
        )
        task.context = [_make_mock_task() for _ in range(3)]
        task.tools = [_make_stub_tool(f"t_{i}") for i in range(3)]

        from crewai.events.types.task_events import TaskStartedEvent
        event = TaskStartedEvent(task=task, context="test context")
        source = MagicMock()
        source.agent = agent

        result = listener._build_event_data("task_started", event, source)
        serialized = json.dumps(result, default=str)

        assert len(serialized) < 2000, f"task_started too large: {len(serialized)} bytes"
        assert "task_name" in result
        assert "agent_role" in result

    def test_error_event_size(self, listener):
        """Error events should be small despite having agent/task refs."""
        from crewai.events.types.agent_events import AgentExecutionErrorEvent

        agent = _make_stub_agent(
            goal="Very long goal " * 100,
            backstory="Very long backstory " * 100,
        )
        task = _make_mock_task(description="Very long description " * 100)

        event = AgentExecutionErrorEvent(
            agent=agent, task=task, error="error"
        )

        result = listener._build_event_data("agent_execution_error", event, None)
        serialized = json.dumps(result, default=str)

        # Should be small - agent/task are just {id, role/name} refs
        assert len(serialized) < 5000, f"error event too large: {len(serialized)} bytes"


# ---------------------------------------------------------------------------
# to_serializable context threading
# ---------------------------------------------------------------------------

class TestToSerializableContext:
    """Test that context parameter flows through to_serializable correctly."""

    def test_context_passed_to_model_dump(self):
        from crewai.events.types.agent_events import AgentExecutionErrorEvent

        agent = _make_stub_agent(role="Tester")
        task = _make_mock_task(name="Test Task")

        event = AgentExecutionErrorEvent(
            agent=agent, task=task, error="test error"
        )

        # Directly use to_serializable with context
        result = to_serializable(event, context={"trace": True})
        assert isinstance(result, dict)
        assert result["agent"] == {"id": str(agent.id), "role": "Tester"}
        assert result["task"] == {"id": str(task.id), "name": "Test Task"}

    def test_no_context_does_not_trigger_serializers(self):
        from crewai.events.types.crew_events import CrewKickoffStartedEvent

        crew = MagicMock()
        crew.fingerprint = MagicMock()
        crew.fingerprint.uuid_str = str(uuid.uuid4())
        crew.fingerprint.metadata = {}

        event = CrewKickoffStartedEvent(
            crew=crew, crew_name="Test", inputs=None
        )

        # Without context, crew should NOT be None
        result = event.model_dump()
        assert result["crew"] is not None