"""Tests for event ordering and parent-child relationships."""

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskStartedEvent,
)
from crewai.flow.flow import Flow, listen, start
from crewai.task import Task


class EventCollector:
    """Collects events and provides helpers to find related events."""

    def __init__(self) -> None:
        self.events: list[BaseEvent] = []

    def add(self, event: BaseEvent) -> None:
        self.events.append(event)

    def first(self, event_type: type[BaseEvent]) -> BaseEvent | None:
        for e in self.events:
            if isinstance(e, event_type):
                return e
        return None

    def all_of(self, event_type: type[BaseEvent]) -> list[BaseEvent]:
        return [e for e in self.events if isinstance(e, event_type)]

    def with_parent(self, parent_id: str) -> list[BaseEvent]:
        return [e for e in self.events if e.parent_event_id == parent_id]


@pytest.fixture
def collector() -> EventCollector:
    """Fixture that collects events during test execution."""
    c = EventCollector()

    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def h1(source, event):
        c.add(event)

    @crewai_event_bus.on(CrewKickoffCompletedEvent)
    def h2(source, event):
        c.add(event)

    @crewai_event_bus.on(TaskStartedEvent)
    def h3(source, event):
        c.add(event)

    @crewai_event_bus.on(TaskCompletedEvent)
    def h4(source, event):
        c.add(event)

    @crewai_event_bus.on(AgentExecutionStartedEvent)
    def h5(source, event):
        c.add(event)

    @crewai_event_bus.on(AgentExecutionCompletedEvent)
    def h6(source, event):
        c.add(event)

    @crewai_event_bus.on(LLMCallStartedEvent)
    def h7(source, event):
        c.add(event)

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def h8(source, event):
        c.add(event)

    @crewai_event_bus.on(FlowStartedEvent)
    def h9(source, event):
        c.add(event)

    @crewai_event_bus.on(FlowFinishedEvent)
    def h10(source, event):
        c.add(event)

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def h11(source, event):
        c.add(event)

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def h12(source, event):
        c.add(event)

    return c


class TestCrewEventOrdering:
    """Tests for event ordering in crew execution."""

    @pytest.mark.vcr()
    def test_crew_events_have_event_ids(self, collector: EventCollector) -> None:
        """Every crew event should have a unique event_id."""
        agent = Agent(
            role="Responder",
            goal="Respond briefly",
            backstory="You give short answers.",
            verbose=False,
        )
        task = Task(
            description="Say 'hello' and nothing else.",
            expected_output="The word hello.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()
        crewai_event_bus.flush()

        started = collector.first(CrewKickoffStartedEvent)
        completed = collector.first(CrewKickoffCompletedEvent)

        assert started is not None
        assert started.event_id is not None
        assert len(started.event_id) > 0

        assert completed is not None
        assert completed.event_id is not None
        assert completed.event_id != started.event_id

    @pytest.mark.vcr()
    def test_crew_completed_after_started(self, collector: EventCollector) -> None:
        """Crew completed event should have higher sequence than started."""
        agent = Agent(
            role="Responder",
            goal="Respond briefly",
            backstory="You give short answers.",
            verbose=False,
        )
        task = Task(
            description="Say 'yes' and nothing else.",
            expected_output="The word yes.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()
        crewai_event_bus.flush()

        started = collector.first(CrewKickoffStartedEvent)
        completed = collector.first(CrewKickoffCompletedEvent)

        assert started is not None
        assert completed is not None
        assert started.emission_sequence is not None
        assert completed.emission_sequence is not None
        assert completed.emission_sequence > started.emission_sequence

    @pytest.mark.vcr()
    def test_task_parent_is_crew(self, collector: EventCollector) -> None:
        """Task events should have crew event as parent."""
        agent = Agent(
            role="Responder",
            goal="Respond briefly",
            backstory="You give short answers.",
            verbose=False,
        )
        task = Task(
            description="Say 'ok' and nothing else.",
            expected_output="The word ok.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        crew.kickoff()
        crewai_event_bus.flush()

        crew_started = collector.first(CrewKickoffStartedEvent)
        task_started = collector.first(TaskStartedEvent)

        assert crew_started is not None
        assert task_started is not None
        assert task_started.parent_event_id == crew_started.event_id


class TestAgentEventOrdering:
    """Tests for event ordering in agent execution."""

    @pytest.mark.vcr()
    def test_agent_events_have_event_ids(self, collector: EventCollector) -> None:
        """Agent execution events should have event_ids."""
        agent = Agent(
            role="Helper",
            goal="Help with tasks",
            backstory="You help.",
            verbose=False,
        )
        task = Task(
            description="Say 'done' and nothing else.",
            expected_output="The word done.",
            agent=agent,
        )
        agent.execute_task(task)
        crewai_event_bus.flush()

        started = collector.first(AgentExecutionStartedEvent)
        completed = collector.first(AgentExecutionCompletedEvent)

        if started:
            assert started.event_id is not None

        if completed:
            assert completed.event_id is not None

    @pytest.mark.vcr()
    def test_llm_events_have_parent(self, collector: EventCollector) -> None:
        """LLM call events should have a parent event."""
        agent = Agent(
            role="Helper",
            goal="Help with tasks",
            backstory="You help.",
            verbose=False,
        )
        task = Task(
            description="Say 'hi' and nothing else.",
            expected_output="The word hi.",
            agent=agent,
        )
        agent.execute_task(task)
        crewai_event_bus.flush()

        llm_started = collector.first(LLMCallStartedEvent)

        if llm_started:
            assert llm_started.event_id is not None
            # LLM events should have some parent in the hierarchy
            assert llm_started.parent_event_id is not None


class TestFlowWithCrewEventOrdering:
    """Tests for event ordering in flows containing crews."""

    @pytest.mark.vcr()
    def test_flow_events_have_ids(self, collector: EventCollector) -> None:
        """Flow events should have event_ids."""
        agent = Agent(
            role="Worker",
            goal="Do work",
            backstory="You work.",
            verbose=False,
        )
        task = Task(
            description="Say 'complete' and nothing else.",
            expected_output="The word complete.",
            agent=agent,
        )

        class SimpleFlow(Flow):
            @start()
            def run_crew(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return c.kickoff()

        flow = SimpleFlow()
        flow.kickoff()
        crewai_event_bus.flush()

        flow_started = collector.first(FlowStartedEvent)
        flow_finished = collector.first(FlowFinishedEvent)

        assert flow_started is not None
        assert flow_started.event_id is not None

        assert flow_finished is not None
        assert flow_finished.event_id is not None

    @pytest.mark.vcr()
    def test_method_parent_is_flow(self, collector: EventCollector) -> None:
        """Method execution events should have flow as parent."""
        agent = Agent(
            role="Worker",
            goal="Do work",
            backstory="You work.",
            verbose=False,
        )
        task = Task(
            description="Say 'ready' and nothing else.",
            expected_output="The word ready.",
            agent=agent,
        )

        class FlowWithMethod(Flow):
            @start()
            def my_method(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return c.kickoff()

        flow = FlowWithMethod()
        flow.kickoff()
        crewai_event_bus.flush()

        flow_started = collector.first(FlowStartedEvent)
        method_started = collector.first(MethodExecutionStartedEvent)

        assert flow_started is not None
        assert method_started is not None
        assert method_started.parent_event_id == flow_started.event_id

    @pytest.mark.vcr()
    def test_crew_parent_is_method(self, collector: EventCollector) -> None:
        """Crew inside flow method should have method as parent."""
        agent = Agent(
            role="Worker",
            goal="Do work",
            backstory="You work.",
            verbose=False,
        )
        task = Task(
            description="Say 'go' and nothing else.",
            expected_output="The word go.",
            agent=agent,
        )

        class FlowWithCrew(Flow):
            @start()
            def run_it(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return c.kickoff()

        flow = FlowWithCrew()
        flow.kickoff()
        crewai_event_bus.flush()

        method_started = collector.first(MethodExecutionStartedEvent)
        crew_started = collector.first(CrewKickoffStartedEvent)

        assert method_started is not None
        assert crew_started is not None
        assert crew_started.parent_event_id == method_started.event_id


class TestFlowWithMultipleCrewsEventOrdering:
    """Tests for event ordering in flows with multiple crews."""

    @pytest.mark.vcr()
    def test_two_crews_have_different_ids(self, collector: EventCollector) -> None:
        """Two crews in a flow should have different event_ids."""
        agent1 = Agent(
            role="First",
            goal="Be first",
            backstory="You go first.",
            verbose=False,
        )
        agent2 = Agent(
            role="Second",
            goal="Be second",
            backstory="You go second.",
            verbose=False,
        )
        task1 = Task(
            description="Say '1' and nothing else.",
            expected_output="The number 1.",
            agent=agent1,
        )
        task2 = Task(
            description="Say '2' and nothing else.",
            expected_output="The number 2.",
            agent=agent2,
        )

        class TwoCrewFlow(Flow):
            @start()
            def first(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return c.kickoff()

            @listen(first)
            def second(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return c.kickoff()

        flow = TwoCrewFlow()
        flow.kickoff()
        crewai_event_bus.flush()

        crew_started_events = collector.all_of(CrewKickoffStartedEvent)

        assert len(crew_started_events) >= 2
        assert crew_started_events[0].event_id != crew_started_events[1].event_id

    @pytest.mark.vcr()
    def test_second_crew_after_first(self, collector: EventCollector) -> None:
        """Second crew should have higher sequence than first."""
        agent1 = Agent(
            role="First",
            goal="Be first",
            backstory="You go first.",
            verbose=False,
        )
        agent2 = Agent(
            role="Second",
            goal="Be second",
            backstory="You go second.",
            verbose=False,
        )
        task1 = Task(
            description="Say 'a' and nothing else.",
            expected_output="The letter a.",
            agent=agent1,
        )
        task2 = Task(
            description="Say 'b' and nothing else.",
            expected_output="The letter b.",
            agent=agent2,
        )

        class SequentialCrewFlow(Flow):
            @start()
            def crew_a(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return c.kickoff()

            @listen(crew_a)
            def crew_b(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return c.kickoff()

        flow = SequentialCrewFlow()
        flow.kickoff()
        crewai_event_bus.flush()

        crew_started_events = collector.all_of(CrewKickoffStartedEvent)

        assert len(crew_started_events) >= 2
        first = crew_started_events[0]
        second = crew_started_events[1]

        assert first.emission_sequence is not None
        assert second.emission_sequence is not None
        assert second.emission_sequence > first.emission_sequence

    @pytest.mark.vcr()
    def test_tasks_have_correct_crew_parents(self, collector: EventCollector) -> None:
        """Tasks in different crews should have their own crew as parent."""
        agent1 = Agent(
            role="Alpha",
            goal="Do alpha work",
            backstory="You are alpha.",
            verbose=False,
        )
        agent2 = Agent(
            role="Beta",
            goal="Do beta work",
            backstory="You are beta.",
            verbose=False,
        )
        task1 = Task(
            description="Say 'alpha' and nothing else.",
            expected_output="The word alpha.",
            agent=agent1,
        )
        task2 = Task(
            description="Say 'beta' and nothing else.",
            expected_output="The word beta.",
            agent=agent2,
        )

        class ParentTestFlow(Flow):
            @start()
            def alpha_crew(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return c.kickoff()

            @listen(alpha_crew)
            def beta_crew(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return c.kickoff()

        flow = ParentTestFlow()
        flow.kickoff()
        crewai_event_bus.flush()

        crew_started_events = collector.all_of(CrewKickoffStartedEvent)
        task_started_events = collector.all_of(TaskStartedEvent)

        assert len(crew_started_events) >= 2
        assert len(task_started_events) >= 2

        crew1_id = crew_started_events[0].event_id
        crew2_id = crew_started_events[1].event_id

        task1_parent = task_started_events[0].parent_event_id
        task2_parent = task_started_events[1].parent_event_id

        assert task1_parent == crew1_id
        assert task2_parent == crew2_id
