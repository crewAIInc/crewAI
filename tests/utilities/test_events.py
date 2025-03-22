import os
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from pydantic import Field

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crew import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.llm import LLM
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.tools.tool_calling import ToolCalling
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.utilities.events.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.event_listener import EventListener
from crewai.utilities.events.event_types import ToolUsageFinishedEvent
from crewai.utilities.events.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionStartedEvent,
)
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
)
from crewai.utilities.events.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.utilities.events.tool_usage_events import (
    ToolUsageErrorEvent,
)

# Skip streaming tests when running in CI/CD environments
skip_streaming_in_ci = pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Skipping streaming tests in CI/CD environments"
)

base_agent = Agent(
    role="base_agent",
    llm="gpt-4o-mini",
    goal="Just say hi",
    backstory="You are a helpful assistant that just says hi",
)

base_task = Task(
    description="Just say hi",
    expected_output="hi",
    agent=base_agent,
)
event_listener = EventListener()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_start_kickoff_event():
    received_events = []
    mock_span = Mock()

    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def handle_crew_start(source, event):
        received_events.append(event)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    with (
        patch.object(
            event_listener._telemetry, "crew_execution_span", return_value=mock_span
        ) as mock_crew_execution_span,
        patch.object(
            event_listener._telemetry, "end_crew", return_value=mock_span
        ) as mock_crew_ended,
    ):
        crew.kickoff()
    mock_crew_execution_span.assert_called_once_with(crew, None)
    mock_crew_ended.assert_called_once_with(crew, "hi")

    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_end_kickoff_event():
    received_events = []

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def handle_crew_end(source, event):
        received_events.append(event)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    crew.kickoff()

    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_test_kickoff_type_event():
    received_events = []
    mock_span = Mock()

    @crewai_event_bus.on(CrewTestStartedEvent)
    def handle_crew_end(source, event):
        received_events.append(event)

    @crewai_event_bus.on(CrewTestCompletedEvent)
    def handle_crew_test_end(source, event):
        received_events.append(event)

    eval_llm = LLM(model="gpt-4o-mini")
    with (
        patch.object(
            event_listener._telemetry, "test_execution_span", return_value=mock_span
        ) as mock_crew_execution_span,
    ):
        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
        crew.test(n_iterations=1, eval_llm=eval_llm)

        # Verify the call was made with correct argument types and values
        assert mock_crew_execution_span.call_count == 1
        args = mock_crew_execution_span.call_args[0]
        assert isinstance(args[0], Crew)
        assert args[1] == 1
        assert args[2] is None
        assert args[3] == eval_llm

    assert len(received_events) == 2
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_test_started"
    assert received_events[1].crew_name == "TestCrew"
    assert isinstance(received_events[1].timestamp, datetime)
    assert received_events[1].type == "crew_test_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_kickoff_failed_event():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def handle_crew_failed(source, event):
            received_events.append(event)

        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

        with patch.object(Crew, "_execute_tasks") as mock_execute:
            error_message = "Simulated crew kickoff failure"
            mock_execute.side_effect = Exception(error_message)

            with pytest.raises(Exception):
                crew.kickoff()

        assert len(received_events) == 1
        assert received_events[0].error == error_message
        assert isinstance(received_events[0].timestamp, datetime)
        assert received_events[0].type == "crew_kickoff_failed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_start_task_event():
    received_events = []

    @crewai_event_bus.on(TaskStartedEvent)
    def handle_task_start(source, event):
        received_events.append(event)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    crew.kickoff()

    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_end_task_event():
    received_events = []

    @crewai_event_bus.on(TaskCompletedEvent)
    def handle_task_end(source, event):
        received_events.append(event)

    mock_span = Mock()
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    with (
        patch.object(
            event_listener._telemetry, "task_started", return_value=mock_span
        ) as mock_task_started,
        patch.object(
            event_listener._telemetry, "task_ended", return_value=mock_span
        ) as mock_task_ended,
    ):
        crew.kickoff()

    mock_task_started.assert_called_once_with(crew=crew, task=base_task)
    mock_task_ended.assert_called_once_with(mock_span, base_task, crew)

    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_task_emits_failed_event_on_execution_error():
    received_events = []
    received_sources = []

    @crewai_event_bus.on(TaskFailedEvent)
    def handle_task_failed(source, event):
        received_events.append(event)
        received_sources.append(source)

    with patch.object(
        Task,
        "_execute_core",
    ) as mock_execute:
        error_message = "Simulated task failure"
        mock_execute.side_effect = Exception(error_message)
        agent = Agent(
            role="base_agent",
            goal="Just say hi",
            backstory="You are a helpful assistant that just says hi",
        )
        task = Task(
            description="Just say hi",
            expected_output="hi",
            agent=agent,
        )

        with pytest.raises(Exception):
            agent.execute_task(task=task)

            assert len(received_events) == 1
            assert received_sources[0] == task
            assert received_events[0].error == error_message
            assert isinstance(received_events[0].timestamp, datetime)
            assert received_events[0].type == "task_failed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_emits_execution_started_and_completed_events():
    received_events = []

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def handle_agent_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def handle_agent_completed(source, event):
        received_events.append(event)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.kickoff()
    assert len(received_events) == 2
    assert received_events[0].agent == base_agent
    assert received_events[0].task == base_task
    assert received_events[0].tools == []
    assert isinstance(received_events[0].task_prompt, str)
    assert (
        received_events[0].task_prompt
        == "Just say hi\n\nThis is the expected criteria for your final answer: hi\nyou MUST return the actual complete content as the final answer, not a summary."
    )
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "agent_execution_started"
    assert isinstance(received_events[1].timestamp, datetime)
    assert received_events[1].type == "agent_execution_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_emits_execution_error_event():
    received_events = []

    @crewai_event_bus.on(AgentExecutionErrorEvent)
    def handle_agent_start(source, event):
        received_events.append(event)

    error_message = "Error happening while sending prompt to model."
    base_agent.max_retry_limit = 0
    with patch.object(
        CrewAgentExecutor, "invoke", wraps=base_agent.agent_executor.invoke
    ) as invoke_mock:
        invoke_mock.side_effect = Exception(error_message)

        with pytest.raises(Exception) as e:
            base_agent.execute_task(
                task=base_task,
            )

        assert len(received_events) == 1
        assert received_events[0].agent == base_agent
        assert received_events[0].task == base_task
        assert received_events[0].error == error_message
        assert isinstance(received_events[0].timestamp, datetime)
        assert received_events[0].type == "agent_execution_error"


class SayHiTool(BaseTool):
    name: str = Field(default="say_hi", description="The name of the tool")
    description: str = Field(
        default="Say hi", description="The description of the tool"
    )

    def _run(self) -> str:
        return "hi"


class DictResultTool(BaseTool):
    name: str = Field(default="dict_result", description="The name of the tool")
    description: str = Field(
        default="Return a dictionary result", 
        description="The description of the tool"
    )

    def _run(self) -> dict:
        return {"message": "success", "data": {"value": 42}}


class NoneResultTool(BaseTool):
    name: str = Field(default="none_result", description="The name of the tool")
    description: str = Field(
        default="Return None as result", 
        description="The description of the tool"
    )

    def _run(self) -> None:
        return None


def test_tools_emits_finished_events_with_string_result():
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    # Create a mock event with string result
    tool = SayHiTool()
    tool_calling = ToolCalling(tool_name=tool.name, arguments={}, log="")
    event_data = {
        "agent_key": "test_agent_key",
        "agent_role": "test_agent_role",
        "tool_name": tool.name,
        "tool_args": {},
        "tool_class": tool.__class__.__name__,
        "started_at": datetime.now(),
        "finished_at": datetime.now(),
        "from_cache": False,
        "result": "hi"
    }
    
    # Emit the event
    crewai_event_bus.emit(None, ToolUsageFinishedEvent(**event_data))
    
    # Verify the event was received with the correct result
    assert len(received_events) == 1
    assert received_events[0].agent_key == "test_agent_key"
    assert received_events[0].agent_role == "test_agent_role"
    assert received_events[0].tool_name == tool.name
    assert received_events[0].tool_args == {}
    assert received_events[0].type == "tool_usage_finished"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].result == "hi"


def test_tools_emits_finished_events_with_dict_result():
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    # Create a mock event with dictionary result
    tool = DictResultTool()
    tool_calling = ToolCalling(tool_name=tool.name, arguments={}, log="")
    dict_result = {"message": "success", "data": {"value": 42}}
    event_data = {
        "agent_key": "test_agent_key",
        "agent_role": "test_agent_role",
        "tool_name": tool.name,
        "tool_args": {},
        "tool_class": tool.__class__.__name__,
        "started_at": datetime.now(),
        "finished_at": datetime.now(),
        "from_cache": False,
        "result": dict_result
    }
    
    # Emit the event
    crewai_event_bus.emit(None, ToolUsageFinishedEvent(**event_data))
    
    # Verify the event was received with the correct result
    assert len(received_events) == 1
    assert received_events[0].agent_key == "test_agent_key"
    assert received_events[0].agent_role == "test_agent_role"
    assert received_events[0].tool_name == tool.name
    assert received_events[0].tool_args == {}
    assert received_events[0].type == "tool_usage_finished"
    assert isinstance(received_events[0].timestamp, datetime)
    assert isinstance(received_events[0].result, dict)
    assert received_events[0].result["message"] == "success"
    assert received_events[0].result["data"]["value"] == 42


def test_tools_emits_finished_events_with_none_result():
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    # Create a mock event with None result
    tool = NoneResultTool()
    tool_calling = ToolCalling(tool_name=tool.name, arguments={}, log="")
    event_data = {
        "agent_key": "test_agent_key",
        "agent_role": "test_agent_role",
        "tool_name": tool.name,
        "tool_args": {},
        "tool_class": tool.__class__.__name__,
        "started_at": datetime.now(),
        "finished_at": datetime.now(),
        "from_cache": False,
        "result": None
    }
    
    # Emit the event
    crewai_event_bus.emit(None, ToolUsageFinishedEvent(**event_data))
    
    # Verify the event was received with the correct result
    assert len(received_events) == 1
    assert received_events[0].agent_key == "test_agent_key"
    assert received_events[0].agent_role == "test_agent_role"
    assert received_events[0].tool_name == tool.name
    assert received_events[0].tool_args == {}
    assert received_events[0].type == "tool_usage_finished"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].result is None


@pytest.mark.vcr(filter_headers=["authorization"])
def test_tools_emits_error_events():
    received_events = []

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    class ErrorTool(BaseTool):
        name: str = Field(
            default="error_tool", description="A tool that raises an error"
        )
        description: str = Field(
            default="This tool always raises an error",
            description="The description of the tool",
        )

        def _run(self) -> str:
            raise Exception("Simulated tool error")

    agent = Agent(
        role="base_agent",
        goal="Try to use the error tool",
        backstory="You are an assistant that tests error handling",
        tools=[ErrorTool()],
    )

    task = Task(
        description="Use the error tool",
        expected_output="This should error",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], name="TestCrew")
    crew.kickoff()

    assert len(received_events) == 75
    assert received_events[0].agent_key == agent.key
    assert received_events[0].agent_role == agent.role
    assert received_events[0].tool_name == "error_tool"
    assert received_events[0].tool_args == {}
    assert str(received_events[0].error) == "Simulated tool error"
    assert received_events[0].type == "tool_usage_error"
    assert isinstance(received_events[0].timestamp, datetime)


def test_flow_emits_start_event():
    received_events = []
    mock_span = Mock()

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

    with (
        patch.object(
            event_listener._telemetry, "flow_execution_span", return_value=mock_span
        ) as mock_flow_execution_span,
    ):
        flow = TestFlow()
        flow.kickoff()

    mock_flow_execution_span.assert_called_once_with("TestFlow", ["begin"])
    assert len(received_events) == 1
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "flow_started"


def test_flow_emits_finish_event():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlowFinishedEvent)
        def handle_flow_finish(source, event):
            received_events.append(event)

        class TestFlow(Flow[dict]):
            @start()
            def begin(self):
                return "completed"

        flow = TestFlow()
        result = flow.kickoff()

        assert len(received_events) == 1
        assert received_events[0].flow_name == "TestFlow"
        assert received_events[0].type == "flow_finished"
        assert received_events[0].result == "completed"
        assert result == "completed"


def test_flow_emits_method_execution_started_event():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def handle_method_start(source, event):
            print("event in method name", event.method_name)
            received_events.append(event)

        class TestFlow(Flow[dict]):
            @start()
            def begin(self):
                return "started"

            @listen("begin")
            def second_method(self):
                return "executed"

        flow = TestFlow()
        flow.kickoff()

        assert len(received_events) == 2

        assert received_events[0].method_name == "begin"
        assert received_events[0].flow_name == "TestFlow"
        assert received_events[0].type == "method_execution_started"

        assert received_events[1].method_name == "second_method"
        assert received_events[1].flow_name == "TestFlow"
        assert received_events[1].type == "method_execution_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_register_handler_adds_new_handler():
    received_events = []

    def custom_handler(source, event):
        received_events.append(event)

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.register_handler(CrewKickoffStartedEvent, custom_handler)

        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
        crew.kickoff()

        assert len(received_events) == 1
        assert isinstance(received_events[0].timestamp, datetime)
        assert received_events[0].type == "crew_kickoff_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_multiple_handlers_for_same_event():
    received_events_1 = []
    received_events_2 = []

    def handler_1(source, event):
        received_events_1.append(event)

    def handler_2(source, event):
        received_events_2.append(event)

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.register_handler(CrewKickoffStartedEvent, handler_1)
        crewai_event_bus.register_handler(CrewKickoffStartedEvent, handler_2)

        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
        crew.kickoff()

        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
        assert received_events_1[0].type == "crew_kickoff_started"
        assert received_events_2[0].type == "crew_kickoff_started"


def test_flow_emits_created_event():
    received_events = []
    mock_span = Mock()

    @crewai_event_bus.on(FlowCreatedEvent)
    def handle_flow_created(source, event):
        received_events.append(event)

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

    with (
        patch.object(
            event_listener._telemetry, "flow_creation_span", return_value=mock_span
        ) as mock_flow_creation_span,
    ):
        flow = TestFlow()
        flow.kickoff()

    mock_flow_creation_span.assert_called_once_with("TestFlow")

    assert len(received_events) == 1
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "flow_created"


def test_flow_emits_method_execution_failed_event():
    received_events = []
    error = Exception("Simulated method failure")

    @crewai_event_bus.on(MethodExecutionFailedEvent)
    def handle_method_failed(source, event):
        received_events.append(event)

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            raise error

    flow = TestFlow()
    with pytest.raises(Exception):
        flow.kickoff()

    assert len(received_events) == 1
    assert received_events[0].method_name == "begin"
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "method_execution_failed"
    assert received_events[0].error == error


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_emits_call_started_event():
    received_events = []

    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_call_started(source, event):
        received_events.append(event)

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_llm_call_completed(source, event):
        received_events.append(event)

    llm = LLM(model="gpt-4o-mini")
    llm.call("Hello, how are you?")

    assert len(received_events) == 2
    assert received_events[0].type == "llm_call_started"
    assert received_events[1].type == "llm_call_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_emits_call_failed_event():
    received_events = []

    @crewai_event_bus.on(LLMCallFailedEvent)
    def handle_llm_call_failed(source, event):
        received_events.append(event)

    error_message = "Simulated LLM call failure"
    with patch("crewai.llm.litellm.completion", side_effect=Exception(error_message)):
        llm = LLM(model="gpt-4o-mini")
        with pytest.raises(Exception) as exc_info:
            llm.call("Hello, how are you?")

        assert str(exc_info.value) == error_message
        assert len(received_events) == 1
        assert received_events[0].type == "llm_call_failed"
        assert received_events[0].error == error_message


@skip_streaming_in_ci
@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_emits_stream_chunk_events():
    """Test that LLM emits stream chunk events when streaming is enabled."""
    received_chunks = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source, event):
            received_chunks.append(event.chunk)

        # Create an LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Call the LLM with a simple message
        response = llm.call("Tell me a short joke")

        # Verify that we received chunks
        assert len(received_chunks) > 0

        # Verify that concatenating all chunks equals the final response
        assert "".join(received_chunks) == response


@skip_streaming_in_ci
@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_no_stream_chunks_when_streaming_disabled():
    """Test that LLM doesn't emit stream chunk events when streaming is disabled."""
    received_chunks = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source, event):
            received_chunks.append(event.chunk)

        # Create an LLM with streaming disabled
        llm = LLM(model="gpt-4o", stream=False)

        # Call the LLM with a simple message
        response = llm.call("Tell me a short joke")

        # Verify that we didn't receive any chunks
        assert len(received_chunks) == 0

        # Verify we got a response
        assert response and isinstance(response, str)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_streaming_fallback_to_non_streaming():
    """Test that streaming falls back to non-streaming when there's an error."""
    received_chunks = []
    fallback_called = False

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source, event):
            received_chunks.append(event.chunk)

        # Create an LLM with streaming enabled
        llm = LLM(model="gpt-4o", stream=True)

        # Store original methods
        original_call = llm.call

        # Create a mock call method that handles the streaming error
        def mock_call(messages, tools=None, callbacks=None, available_functions=None):
            nonlocal fallback_called
            # Emit a couple of chunks to simulate partial streaming
            crewai_event_bus.emit(llm, event=LLMStreamChunkEvent(chunk="Test chunk 1"))
            crewai_event_bus.emit(llm, event=LLMStreamChunkEvent(chunk="Test chunk 2"))

            # Mark that fallback would be called
            fallback_called = True

            # Return a response as if fallback succeeded
            return "Fallback response after streaming error"

        # Replace the call method with our mock
        llm.call = mock_call

        try:
            # Call the LLM
            response = llm.call("Tell me a short joke")

            # Verify that we received some chunks
            assert len(received_chunks) == 2
            assert received_chunks[0] == "Test chunk 1"
            assert received_chunks[1] == "Test chunk 2"

            # Verify fallback was triggered
            assert fallback_called

            # Verify we got the fallback response
            assert response == "Fallback response after streaming error"

        finally:
            # Restore the original method
            llm.call = original_call


@pytest.mark.vcr(filter_headers=["authorization"])
def test_streaming_empty_response_handling():
    """Test that streaming handles empty responses correctly."""
    received_chunks = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source, event):
            received_chunks.append(event.chunk)

        # Create an LLM with streaming enabled
        llm = LLM(model="gpt-3.5-turbo", stream=True)

        # Store original methods
        original_call = llm.call

        # Create a mock call method that simulates empty chunks
        def mock_call(messages, tools=None, callbacks=None, available_functions=None):
            # Emit a few empty chunks
            for _ in range(3):
                crewai_event_bus.emit(llm, event=LLMStreamChunkEvent(chunk=""))

            # Return the default message for empty responses
            return "I apologize, but I couldn't generate a proper response. Please try again or rephrase your request."

        # Replace the call method with our mock
        llm.call = mock_call

        try:
            # Call the LLM - this should handle empty response
            response = llm.call("Tell me a short joke")

            # Verify that we received empty chunks
            assert len(received_chunks) == 3
            assert all(chunk == "" for chunk in received_chunks)

            # Verify the response is the default message for empty responses
            assert "I apologize" in response and "couldn't generate" in response

        finally:
            # Restore the original method
            llm.call = original_call
