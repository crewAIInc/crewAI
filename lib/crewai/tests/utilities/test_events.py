import threading
from datetime import datetime
import os
from unittest.mock import Mock, patch

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestResultEvent,
    CrewTestStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
)
from crewai.flow.flow import Flow, listen, start
from crewai.llm import LLM
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, Field
import pytest

from ..utils import wait_for_event_handlers


@pytest.fixture(scope="module")
def base_agent():
    return Agent(
        role="base_agent",
        llm="gpt-4o-mini",
        goal="Just say hi",
        backstory="You are a helpful assistant that just says hi",
    )


@pytest.fixture(scope="module")
def base_task(base_agent):
    return Task(
        description="Just say hi",
        expected_output="hi",
        agent=base_agent,
    )


@pytest.fixture
def reset_event_listener_singleton():
    """Reset EventListener singleton for clean test state."""
    original_instance = EventListener._instance
    original_initialized = (
        getattr(EventListener._instance, "_initialized", False)
        if EventListener._instance
        else False
    )

    EventListener._instance = None

    yield

    EventListener._instance = original_instance
    if original_instance and original_initialized:
        EventListener._instance._initialized = original_initialized


@pytest.mark.vcr()
def test_crew_emits_start_kickoff_event(
    base_agent, base_task, reset_event_listener_singleton
):
    received_events = []
    mock_span = Mock()

    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def handle_crew_start(source, event):
        received_events.append(event)

    mock_telemetry = Mock()
    mock_telemetry.crew_execution_span = Mock(return_value=mock_span)
    mock_telemetry.end_crew = Mock(return_value=mock_span)
    mock_telemetry.set_tracer = Mock()
    mock_telemetry.task_started = Mock(return_value=mock_span)
    mock_telemetry.task_ended = Mock(return_value=mock_span)

    # Patch the Telemetry class to return our mock
    with patch("crewai.events.event_listener.Telemetry", return_value=mock_telemetry):
        # Now when Crew creates EventListener, it will use our mocked telemetry
        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
        crew.kickoff()
    wait_for_event_handlers()

    mock_telemetry.crew_execution_span.assert_called_once_with(crew, None)
    mock_telemetry.end_crew.assert_called_once_with(crew, "hi")

    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_started"


@pytest.mark.vcr()
def test_crew_emits_end_kickoff_event(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def handle_crew_end(source, event):
        received_events.append(event)
        event_received.set()

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    crew.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for crew kickoff completed event"
    )
    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_completed"


@pytest.mark.vcr()
def test_crew_emits_test_kickoff_type_event(base_agent, base_task):
    received_events = []

    @crewai_event_bus.on(CrewTestStartedEvent)
    def handle_crew_end(source, event):
        received_events.append(event)

    @crewai_event_bus.on(CrewTestCompletedEvent)
    def handle_crew_test_end(source, event):
        received_events.append(event)

    @crewai_event_bus.on(CrewTestResultEvent)
    def handle_crew_test_result(source, event):
        received_events.append(event)

    eval_llm = LLM(model="gpt-4o-mini")
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.test(n_iterations=1, eval_llm=eval_llm)
    wait_for_event_handlers()

    assert len(received_events) == 3
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_test_started"
    assert received_events[1].crew_name == "TestCrew"
    assert isinstance(received_events[1].timestamp, datetime)
    assert received_events[1].type == "crew_test_result"
    assert received_events[2].crew_name == "TestCrew"
    assert isinstance(received_events[2].timestamp, datetime)
    assert received_events[2].type == "crew_test_completed"


@pytest.mark.vcr()
def test_crew_emits_kickoff_failed_event(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(CrewKickoffFailedEvent)
    def handle_crew_failed(source, event):
        received_events.append(event)
        event_received.set()

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    with patch.object(Crew, "_execute_tasks") as mock_execute:
        error_message = "Simulated crew kickoff failure"
        mock_execute.side_effect = Exception(error_message)

        with pytest.raises(Exception):  # noqa: B017
            crew.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for failed event"
    assert len(received_events) == 1
    assert received_events[0].error == error_message
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_failed"


@pytest.mark.vcr()
def test_crew_emits_start_task_event(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(TaskStartedEvent)
    def handle_task_start(source, event):
        received_events.append(event)
        event_received.set()

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    crew.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for task started event"
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_started"


@pytest.mark.vcr()
def test_crew_emits_end_task_event(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(TaskCompletedEvent)
    def handle_task_end(source, event):
        received_events.append(event)
        event_received.set()

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for task completed event"
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_completed"


@pytest.mark.vcr()
def test_task_emits_failed_event_on_execution_error(base_agent, base_task):
    received_events = []
    received_sources = []
    event_received = threading.Event()

    @crewai_event_bus.on(TaskFailedEvent)
    def handle_task_failed(source, event):
        received_events.append(event)
        received_sources.append(source)
        event_received.set()

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

        with pytest.raises(Exception):  # noqa: B017
            agent.execute_task(task=task)

            assert event_received.wait(timeout=5), (
                "Timeout waiting for task failed event"
            )
            assert len(received_events) == 1
            assert received_sources[0] == task
            assert received_events[0].error == error_message
            assert isinstance(received_events[0].timestamp, datetime)
            assert received_events[0].type == "task_failed"


@pytest.mark.vcr()
def test_agent_emits_execution_started_and_completed_events(base_agent, base_task):
    started_events: list[AgentExecutionStartedEvent] = []
    completed_events: list[AgentExecutionCompletedEvent] = []
    condition = threading.Condition()

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def handle_agent_start(source, event):
        with condition:
            started_events.append(event)
            condition.notify()

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def handle_agent_completed(source, event):
        with condition:
            completed_events.append(event)
            condition.notify()

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.kickoff()

    with condition:
        success = condition.wait_for(
            lambda: len(started_events) >= 1 and len(completed_events) >= 1,
            timeout=10,
        )
    assert success, "Timeout waiting for agent execution events"

    assert len(started_events) == 1
    assert len(completed_events) == 1
    assert started_events[0].agent == base_agent
    assert started_events[0].task == base_task
    assert started_events[0].tools == []
    assert isinstance(started_events[0].task_prompt, str)
    assert (
        started_events[0].task_prompt
        == "Just say hi\n\nThis is the expected criteria for your final answer: hi\nyou MUST return the actual complete content as the final answer, not a summary."
    )
    assert isinstance(started_events[0].timestamp, datetime)
    assert started_events[0].type == "agent_execution_started"
    assert isinstance(completed_events[0].timestamp, datetime)
    assert completed_events[0].type == "agent_execution_completed"


@pytest.mark.vcr()
def test_agent_emits_execution_error_event(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(AgentExecutionErrorEvent)
    def handle_agent_start(source, event):
        received_events.append(event)
        event_received.set()

    error_message = "Error happening while sending prompt to model."
    base_agent.max_retry_limit = 0
    with patch.object(
        CrewAgentExecutor, "invoke", wraps=base_agent.agent_executor.invoke
    ) as invoke_mock:
        invoke_mock.side_effect = Exception(error_message)

        with pytest.raises(Exception):  # noqa: B017
            base_agent.execute_task(
                task=base_task,
            )

        assert event_received.wait(timeout=5), (
            "Timeout waiting for agent execution error event"
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


@pytest.mark.vcr()
def test_tools_emits_finished_events():
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)
        event_received.set()

    agent = Agent(
        role="base_agent",
        goal="Just say hi",
        backstory="You are a helpful assistant that just says hi",
        tools=[SayHiTool()],
    )

    task = Task(
        description="Just say hi",
        expected_output="hi",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], name="TestCrew")
    crew.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for tool usage finished event"
    )
    assert len(received_events) == 1
    assert received_events[0].agent_key == agent.key
    assert received_events[0].agent_role == agent.role
    assert received_events[0].tool_name == SayHiTool().name
    assert received_events[0].tool_args == "{}" or received_events[0].tool_args == {}
    assert received_events[0].type == "tool_usage_finished"
    assert isinstance(received_events[0].timestamp, datetime)


@pytest.mark.vcr()
def test_tools_emits_error_events():
    received_events = []
    lock = threading.Lock()
    all_events_received = threading.Event()

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def handle_tool_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) >= 48:
                all_events_received.set()

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
        llm=LLM(model="gpt-4o-mini"),
    )

    task = Task(
        description="Use the error tool",
        expected_output="This should error",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], name="TestCrew")
    crew.kickoff()

    assert all_events_received.wait(timeout=5), (
        "Timeout waiting for tool usage error events"
    )
    assert len(received_events) == 48
    assert received_events[0].agent_key == agent.key
    assert received_events[0].agent_role == agent.role
    assert received_events[0].tool_name == "error_tool"
    assert received_events[0].tool_args == "{}" or received_events[0].tool_args == {}
    assert str(received_events[0].error) == "Simulated tool error"
    assert received_events[0].type == "tool_usage_error"
    assert isinstance(received_events[0].timestamp, datetime)


def test_flow_emits_start_event(reset_event_listener_singleton):
    received_events = []
    event_received = threading.Event()
    mock_span = Mock()

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)
        event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

    mock_telemetry = Mock()
    mock_telemetry.flow_execution_span = Mock(return_value=mock_span)
    mock_telemetry.flow_creation_span = Mock()
    mock_telemetry.set_tracer = Mock()

    with patch("crewai.events.event_listener.Telemetry", return_value=mock_telemetry):
        # Force creation of EventListener singleton with mocked telemetry
        _ = EventListener()

        flow = TestFlow()
        flow.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for flow started event"
    mock_telemetry.flow_execution_span.assert_called_once_with("TestFlow", ["begin"])
    assert len(received_events) == 1
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "flow_started"


def test_flow_name_emitted_to_event_bus():
    received_events = []
    event_received = threading.Event()

    class MyFlowClass(Flow):
        name = "PRODUCTION_FLOW"

        @start()
        def start(self):
            return "Hello, world!"

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)
        event_received.set()

    flow = MyFlowClass()
    flow.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for flow started event"
    assert len(received_events) == 1
    assert received_events[0].flow_name == "PRODUCTION_FLOW"


def test_flow_emits_finish_event():
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_finish(source, event):
        received_events.append(event)
        event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "completed"

    flow = TestFlow()
    result = flow.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for finish event"
    assert len(received_events) == 1
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "flow_finished"
    assert received_events[0].result == "completed"
    assert result == "completed"


def test_flow_emits_method_execution_started_event():
    received_events = []
    lock = threading.Lock()
    second_event_received = threading.Event()

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    async def handle_method_start(source, event):
        with lock:
            received_events.append(event)
            if event.method_name == "second_method":
                second_event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

        @listen("begin")
        def second_method(self):
            return "executed"

    flow = TestFlow()
    flow.kickoff()

    assert second_event_received.wait(timeout=5), (
        "Timeout waiting for second_method event"
    )
    assert len(received_events) == 2

    # Events may arrive in any order due to async handlers, so check both are present
    method_names = {event.method_name for event in received_events}
    assert method_names == {"begin", "second_method"}

    for event in received_events:
        assert event.flow_name == "TestFlow"
        assert event.type == "method_execution_started"


@pytest.mark.vcr()
def test_register_handler_adds_new_handler(base_agent, base_task):
    received_events = []
    event_received = threading.Event()

    def custom_handler(source, event):
        received_events.append(event)
        event_received.set()

    crewai_event_bus.register_handler(CrewKickoffStartedEvent, custom_handler)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for handler event"
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_started"


@pytest.mark.vcr()
def test_multiple_handlers_for_same_event(base_agent, base_task):
    received_events_1 = []
    received_events_2 = []
    event_received = threading.Event()

    def handler_1(source, event):
        received_events_1.append(event)

    def handler_2(source, event):
        received_events_2.append(event)
        event_received.set()

    crewai_event_bus.register_handler(CrewKickoffStartedEvent, handler_1)
    crewai_event_bus.register_handler(CrewKickoffStartedEvent, handler_2)

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")
    crew.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for handler events"
    assert len(received_events_1) == 1
    assert len(received_events_2) == 1
    assert received_events_1[0].type == "crew_kickoff_started"
    assert received_events_2[0].type == "crew_kickoff_started"


def test_flow_emits_created_event():
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(FlowCreatedEvent)
    def handle_flow_created(source, event):
        received_events.append(event)
        event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

    flow = TestFlow()
    flow.kickoff()

    assert event_received.wait(timeout=5), "Timeout waiting for flow created event"
    assert len(received_events) == 1
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "flow_created"


def test_flow_emits_method_execution_failed_event():
    received_events = []
    event_received = threading.Event()
    error = Exception("Simulated method failure")

    @crewai_event_bus.on(MethodExecutionFailedEvent)
    def handle_method_failed(source, event):
        received_events.append(event)
        event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            raise error

    flow = TestFlow()
    with pytest.raises(Exception):  # noqa: B017
        flow.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for method execution failed event"
    )
    assert len(received_events) == 1
    assert received_events[0].method_name == "begin"
    assert received_events[0].flow_name == "TestFlow"
    assert received_events[0].type == "method_execution_failed"
    assert received_events[0].error == error


def test_flow_method_execution_started_includes_unstructured_state():
    """Test that MethodExecutionStartedEvent includes unstructured (dict) state."""
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_started(source, event):
        received_events.append(event)
        if event.method_name == "process":
            event_received.set()

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            self.state["counter"] = 1
            self.state["message"] = "test"
            return "started"

        @listen("begin")
        def process(self):
            self.state["counter"] = 2
            return "processed"

    flow = TestFlow()
    flow.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for method execution started event"
    )

    # Find the events for each method
    begin_event = next(e for e in received_events if e.method_name == "begin")
    process_event = next(e for e in received_events if e.method_name == "process")

    # Verify state is included and is a dict
    assert begin_event.state is not None
    assert isinstance(begin_event.state, dict)
    assert "id" in begin_event.state  # Auto-generated ID

    # Verify state from begin method is captured in process event
    assert process_event.state is not None
    assert isinstance(process_event.state, dict)
    assert process_event.state["counter"] == 1
    assert process_event.state["message"] == "test"


def test_flow_method_execution_started_includes_structured_state():
    """Test that MethodExecutionStartedEvent includes structured (BaseModel) state and serializes it properly."""
    received_events = []
    event_received = threading.Event()

    class FlowState(BaseModel):
        counter: int = 0
        message: str = ""
        items: list[str] = []

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_started(source, event):
        received_events.append(event)
        if event.method_name == "process":
            event_received.set()

    class TestFlow(Flow[FlowState]):
        @start()
        def begin(self):
            self.state.counter = 1
            self.state.message = "initial"
            self.state.items = ["a", "b"]
            return "started"

        @listen("begin")
        def process(self):
            self.state.counter += 1
            return "processed"

    flow = TestFlow()
    flow.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for method execution started event"
    )

    begin_event = next(e for e in received_events if e.method_name == "begin")
    process_event = next(e for e in received_events if e.method_name == "process")

    assert begin_event.state is not None
    assert isinstance(begin_event.state, dict)
    assert begin_event.state["counter"] == 0  # Initial state
    assert begin_event.state["message"] == ""
    assert begin_event.state["items"] == []

    assert process_event.state is not None
    assert isinstance(process_event.state, dict)
    assert process_event.state["counter"] == 1
    assert process_event.state["message"] == "initial"
    assert process_event.state["items"] == ["a", "b"]


def test_flow_method_execution_finished_includes_serialized_state():
    """Test that MethodExecutionFinishedEvent includes properly serialized state."""
    received_events = []
    event_received = threading.Event()

    class FlowState(BaseModel):
        result: str = ""
        completed: bool = False

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def handle_method_finished(source, event):
        received_events.append(event)
        if event.method_name == "process":
            event_received.set()

    class TestFlow(Flow[FlowState]):
        @start()
        def begin(self):
            self.state.result = "begin done"
            return "started"

        @listen("begin")
        def process(self):
            self.state.result = "process done"
            self.state.completed = True
            return "final_result"

    flow = TestFlow()
    final_output = flow.kickoff()

    assert event_received.wait(timeout=5), (
        "Timeout waiting for method execution finished event"
    )

    begin_finished = next(e for e in received_events if e.method_name == "begin")
    process_finished = next(e for e in received_events if e.method_name == "process")

    assert begin_finished.state is not None
    assert isinstance(begin_finished.state, dict)
    assert begin_finished.state["result"] == "begin done"
    assert begin_finished.state["completed"] is False
    assert begin_finished.result == "started"

    # Verify process finished event has final state and result
    assert process_finished.state is not None
    assert isinstance(process_finished.state, dict)
    assert process_finished.state["result"] == "process done"
    assert process_finished.state["completed"] is True
    assert process_finished.result == "final_result"
    assert final_output == "final_result"


@pytest.mark.vcr()
def test_llm_emits_call_started_event():
    started_events: list[LLMCallStartedEvent] = []
    completed_events: list[LLMCallCompletedEvent] = []
    condition = threading.Condition()

    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_call_started(source, event):
        with condition:
            started_events.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_llm_call_completed(source, event):
        with condition:
            completed_events.append(event)
            condition.notify()

    llm = LLM(model="gpt-4o-mini")
    llm.call("Hello, how are you?")

    with condition:
        success = condition.wait_for(
            lambda: len(started_events) >= 1 and len(completed_events) >= 1,
            timeout=10,
        )
    assert success, "Timeout waiting for LLM events"

    assert started_events[0].type == "llm_call_started"
    assert completed_events[0].type == "llm_call_completed"

    assert started_events[0].task_name is None
    assert started_events[0].agent_role is None
    assert started_events[0].agent_id is None
    assert started_events[0].task_id is None


@pytest.mark.vcr()
def test_llm_emits_call_failed_event():
    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(LLMCallFailedEvent)
    def handle_llm_call_failed(source, event):
        received_events.append(event)
        event_received.set()

    error_message = "OpenAI API call failed: Simulated API failure"

    with patch(
        "crewai.llms.providers.openai.completion.OpenAICompletion._handle_completion"
    ) as mock_handle_completion:
        mock_handle_completion.side_effect = Exception("Simulated API failure")

        llm = LLM(model="gpt-4o-mini")
        with pytest.raises(Exception) as exc_info:
            llm.call("Hello, how are you?")

        assert str(exc_info.value) == "Simulated API failure"
        assert event_received.wait(timeout=5), "Timeout waiting for failed event"
        assert len(received_events) == 1
        assert received_events[0].type == "llm_call_failed"
        assert received_events[0].error == error_message
        assert received_events[0].task_name is None
        assert received_events[0].agent_role is None
        assert received_events[0].agent_id is None
        assert received_events[0].task_id is None


@pytest.mark.vcr()
def test_llm_emits_stream_chunk_events():
    """Test that LLM emits stream chunk events when streaming is enabled."""
    received_chunks = []
    event_received = threading.Event()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_stream_chunk(source, event):
        received_chunks.append(event.chunk)
        if len(received_chunks) >= 1:
            event_received.set()

    # Create an LLM with streaming enabled
    llm = LLM(model="gpt-4o", stream=True)

    # Call the LLM with a simple message
    response = llm.call("Tell me a short joke")

    # Wait for at least one chunk
    assert event_received.wait(timeout=5), "Timeout waiting for stream chunks"

    # Verify that we received chunks
    assert len(received_chunks) > 0

    # Verify that concatenating all chunks equals the final response
    assert "".join(received_chunks) == response


@pytest.mark.vcr()
def test_llm_no_stream_chunks_when_streaming_disabled():
    """Test that LLM doesn't emit stream chunk events when streaming is disabled."""
    received_chunks = []

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


@pytest.mark.vcr()
def test_streaming_fallback_to_non_streaming():
    """Test that streaming falls back to non-streaming when there's an error."""
    received_chunks = []
    fallback_called = False
    event_received = threading.Event()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_stream_chunk(source, event):
        received_chunks.append(event.chunk)
        if len(received_chunks) >= 2:
            event_received.set()

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
        wait_for_event_handlers()

        assert event_received.wait(timeout=5), "Timeout waiting for stream chunks"

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


@pytest.mark.vcr()
def test_streaming_empty_response_handling():
    """Test that streaming handles empty responses correctly."""
    received_chunks = []
    event_received = threading.Event()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_stream_chunk(source, event):
        received_chunks.append(event.chunk)
        if len(received_chunks) >= 3:
            event_received.set()

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

        assert event_received.wait(timeout=5), "Timeout waiting for empty chunks"

        # Verify that we received empty chunks
        assert len(received_chunks) == 3
        assert all(chunk == "" for chunk in received_chunks)

        # Verify the response is the default message for empty responses
        assert "I apologize" in response and "couldn't generate" in response

    finally:
        # Restore the original method
        llm.call = original_call


@pytest.mark.vcr()
def test_stream_llm_emits_event_with_task_and_agent_info():
    completed_event = []
    failed_event = []
    started_event = []
    stream_event = []
    condition = threading.Condition()

    @crewai_event_bus.on(LLMCallFailedEvent)
    def handle_llm_failed(source, event):
        with condition:
            failed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_started(source, event):
        with condition:
            started_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_llm_completed(source, event):
        with condition:
            completed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_llm_stream_chunk(source, event):
        with condition:
            stream_event.append(event)
            condition.notify()

    agent = Agent(
        role="TestAgent",
        llm=LLM(model="gpt-4o-mini", stream=True),
        goal="Just say hi",
        backstory="You are a helpful assistant that just says hi",
    )
    task = Task(
        description="Just say hi",
        expected_output="hi",
        llm=LLM(model="gpt-4o-mini", stream=True),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    crew.kickoff()

    with condition:
        success = condition.wait_for(
            lambda: len(completed_event) >= 1
            and len(started_event) >= 1
            and len(stream_event) >= 12,
            timeout=10,
        )
    assert success, "Timeout waiting for LLM events"
    assert len(completed_event) == 1
    assert len(failed_event) == 0
    assert len(started_event) == 1
    assert len(stream_event) == 12

    all_events = completed_event + failed_event + started_event + stream_event
    all_agent_roles = [event.agent_role for event in all_events]
    all_agent_id = [event.agent_id for event in all_events]
    all_task_id = [event.task_id for event in all_events]
    all_task_name = [event.task_name for event in all_events]

    # ensure all events have the agent + task props set
    assert len(all_agent_roles) == 14
    assert len(all_agent_id) == 14
    assert len(all_task_id) == 14
    assert len(all_task_name) == 14

    assert set(all_agent_roles) == {agent.role}
    assert set(all_agent_id) == {str(agent.id)}
    assert set(all_task_id) == {str(task.id)}
    assert set(all_task_name) == {task.name or task.description}


@pytest.mark.vcr()
def test_llm_emits_event_with_task_and_agent_info(base_agent, base_task):
    completed_event: list[LLMCallCompletedEvent] = []
    failed_event: list[LLMCallFailedEvent] = []
    started_event: list[LLMCallStartedEvent] = []
    stream_event: list[LLMStreamChunkEvent] = []
    condition = threading.Condition()

    @crewai_event_bus.on(LLMCallFailedEvent)
    def handle_llm_failed(source, event):
        with condition:
            failed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_started(source, event):
        with condition:
            started_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_llm_completed(source, event):
        with condition:
            completed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_llm_stream_chunk(source, event):
        with condition:
            stream_event.append(event)
            condition.notify()

    crew = Crew(agents=[base_agent], tasks=[base_task])
    crew.kickoff()

    with condition:
        success = condition.wait_for(
            lambda: len(completed_event) >= 1 and len(started_event) >= 1,
            timeout=10,
        )
    assert success, "Timeout waiting for LLM events"
    assert len(completed_event) == 1
    assert len(failed_event) == 0
    assert len(started_event) == 1
    assert len(stream_event) == 0

    all_events = completed_event + failed_event + started_event + stream_event
    all_agent_roles = [event.agent_role for event in all_events]
    all_agent_id = [event.agent_id for event in all_events]
    all_task_id = [event.task_id for event in all_events]
    all_task_name = [event.task_name for event in all_events]

    # ensure all events have the agent + task props set
    assert len(all_agent_roles) == 2
    assert len(all_agent_id) == 2
    assert len(all_task_id) == 2
    assert len(all_task_name) == 2

    assert set(all_agent_roles) == {base_agent.role}
    assert set(all_agent_id) == {str(base_agent.id)}
    assert set(all_task_id) == {str(base_task.id)}
    assert set(all_task_name) == {base_task.name or base_task.description}


@pytest.mark.vcr()
def test_llm_emits_event_with_lite_agent():
    completed_event = []
    failed_event = []
    started_event = []
    stream_event = []
    condition = threading.Condition()

    @crewai_event_bus.on(LLMCallFailedEvent)
    def handle_llm_failed(source, event):
        with condition:
            failed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallStartedEvent)
    def handle_llm_started(source, event):
        with condition:
            started_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def handle_llm_completed(source, event):
        with condition:
            completed_event.append(event)
            condition.notify()

    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handle_llm_stream_chunk(source, event):
        with condition:
            stream_event.append(event)
            condition.notify()

    agent = Agent(
        role="Speaker",
        llm=LLM(model="gpt-4o-mini", stream=True),
        goal="Just say hi",
        backstory="You are a helpful assistant that just says hi",
    )
    agent.kickoff(messages=[{"role": "user", "content": "say hi!"}])

    with condition:
        success = condition.wait_for(
            lambda: len(completed_event) >= 1
            and len(started_event) >= 1
            and len(stream_event) >= 15,
            timeout=10,
        )
    assert success, "Timeout waiting for all events"

    assert len(completed_event) == 1
    assert len(failed_event) == 0
    assert len(started_event) == 1
    assert len(stream_event) == 15

    all_events = completed_event + failed_event + started_event + stream_event
    all_agent_roles = [event.agent_role for event in all_events]
    all_agent_id = [event.agent_id for event in all_events]
    all_task_id = [event.task_id for event in all_events if event.task_id]
    all_task_name = [event.task_name for event in all_events if event.task_name]

    # ensure all events have the agent + task props set
    assert len(all_agent_roles) == 17
    assert len(all_agent_id) == 17
    assert len(all_task_id) == 0
    assert len(all_task_name) == 0

    assert set(all_agent_roles) == {agent.role}
    assert set(all_agent_id) == {str(agent.id)}
