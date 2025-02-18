from datetime import datetime
from unittest.mock import patch

import pytest
from pydantic import Field

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.crew import Crew
from crewai.flow.flow import Flow, listen, start
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.utilities.events.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.event_types import ToolUsageFinishedEvent
from crewai.utilities.events.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionStartedEvent,
)
from crewai.utilities.events.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.utilities.events.tool_usage_events import ToolUsageErrorEvent

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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_start_kickoff_event():
    received_events = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def handle_crew_start(source, event):
            received_events.append(event)

        crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

        crew.kickoff()

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

    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    crew.kickoff()

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
    assert received_events[0].inputs == {
        "ask_for_human_input": False,
        "input": "Just say hi\n"
        "\n"
        "This is the expected criteria for your final answer: hi\n"
        "you MUST return the actual complete content as the final answer, not a "
        "summary.",
        "tool_names": "",
        "tools": "",
    }
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_tools_emits_finished_events():
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

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
    assert len(received_events) == 1
    assert received_events[0].agent_key == agent.key
    assert received_events[0].agent_role == agent.role
    assert received_events[0].tool_name == SayHiTool().name
    assert received_events[0].tool_args == {}
    assert received_events[0].type == "tool_usage_finished"
    assert isinstance(received_events[0].timestamp, datetime)


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

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlowStartedEvent)
        def handle_flow_start(source, event):
            received_events.append(event)

        class TestFlow(Flow[dict]):
            @start()
            def begin(self):
                return "started"

        flow = TestFlow()
        flow.kickoff()

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

    @crewai_event_bus.on(FlowCreatedEvent)
    def handle_flow_created(source, event):
        received_events.append(event)

    class TestFlow(Flow[dict]):
        @start()
        def begin(self):
            return "started"

    flow = TestFlow()
    flow.kickoff()

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
