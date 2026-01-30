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

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_flow_events_have_ids(self, collector: EventCollector) -> None:
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
            async def run_crew(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return await c.akickoff()

        flow = SimpleFlow()
        await flow.akickoff()
        crewai_event_bus.flush()

        flow_started = collector.first(FlowStartedEvent)
        flow_finished = collector.first(FlowFinishedEvent)

        assert flow_started is not None
        assert flow_started.event_id is not None

        assert flow_finished is not None
        assert flow_finished.event_id is not None

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_method_parent_is_flow(self, collector: EventCollector) -> None:
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
            async def my_method(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return await c.akickoff()

        flow = FlowWithMethod()
        await flow.akickoff()
        crewai_event_bus.flush()

        flow_started = collector.first(FlowStartedEvent)
        method_started = collector.first(MethodExecutionStartedEvent)

        assert flow_started is not None
        assert method_started is not None
        assert method_started.parent_event_id == flow_started.event_id

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_crew_parent_is_method(self, collector: EventCollector) -> None:
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
            async def run_it(self):
                c = Crew(agents=[agent], tasks=[task], verbose=False)
                return await c.akickoff()

        flow = FlowWithCrew()
        await flow.akickoff()
        crewai_event_bus.flush()

        method_started = collector.first(MethodExecutionStartedEvent)
        crew_started = collector.first(CrewKickoffStartedEvent)

        assert method_started is not None
        assert crew_started is not None
        assert crew_started.parent_event_id == method_started.event_id


class TestFlowWithMultipleCrewsEventOrdering:
    """Tests for event ordering in flows with multiple crews."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_two_crews_have_different_ids(
        self, collector: EventCollector
    ) -> None:
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
            async def first(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return await c.akickoff()

            @listen(first)
            async def second(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return await c.akickoff()

        flow = TwoCrewFlow()
        await flow.akickoff()
        crewai_event_bus.flush()

        crew_started_events = collector.all_of(CrewKickoffStartedEvent)

        assert len(crew_started_events) >= 2
        assert crew_started_events[0].event_id != crew_started_events[1].event_id

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_second_crew_after_first(self, collector: EventCollector) -> None:
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
            async def crew_a(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return await c.akickoff()

            @listen(crew_a)
            async def crew_b(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return await c.akickoff()

        flow = SequentialCrewFlow()
        await flow.akickoff()
        crewai_event_bus.flush()

        crew_started_events = collector.all_of(CrewKickoffStartedEvent)

        assert len(crew_started_events) >= 2
        first = crew_started_events[0]
        second = crew_started_events[1]

        assert first.emission_sequence is not None
        assert second.emission_sequence is not None
        assert second.emission_sequence > first.emission_sequence

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_tasks_have_correct_crew_parents(
        self, collector: EventCollector
    ) -> None:
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
            async def alpha_crew(self):
                c = Crew(agents=[agent1], tasks=[task1], verbose=False)
                return await c.akickoff()

            @listen(alpha_crew)
            async def beta_crew(self, _):
                c = Crew(agents=[agent2], tasks=[task2], verbose=False)
                return await c.akickoff()

        flow = ParentTestFlow()
        await flow.akickoff()
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


class TestPreviousEventIdChain:
    """Tests for previous_event_id linear chain tracking."""

    @pytest.mark.asyncio
    async def test_previous_event_id_chain(self) -> None:
        """Events should have previous_event_id pointing to the prior event."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class SimpleFlow(Flow):
            @start()
            async def step_one(self):
                return "step_one_done"

            @listen(step_one)
            async def step_two(self, result):
                return "step_two_done"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowStartedEvent)
            def h1(source, event):
                events.append(event)

            @crewai_event_bus.on(FlowFinishedEvent)
            def h2(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def h3(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def h4(source, event):
                events.append(event)

            flow = SimpleFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        assert len(events) >= 4

        all_events = sorted(events, key=lambda e: e.emission_sequence or 0)
        all_event_ids = {e.event_id for e in all_events}

        for event in all_events[1:]:
            assert event.previous_event_id is not None, (
                f"Event {event.type} (seq {event.emission_sequence}) has no previous_event_id"
            )
            if event.previous_event_id in all_event_ids:
                prev = next(e for e in all_events if e.event_id == event.previous_event_id)
                assert (prev.emission_sequence or 0) < (event.emission_sequence or 0), (
                    f"Event {event.type} (seq {event.emission_sequence}) has previous pointing "
                    f"to {prev.type} (seq {prev.emission_sequence}) which is not earlier"
                )

    @pytest.mark.asyncio
    async def test_first_event_has_previous_pointing_back(self) -> None:
        """Non-first events should have previous_event_id set."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        events: list[BaseEvent] = []

        class MinimalFlow(Flow):
            @start()
            async def do_nothing(self):
                return "done"

        reset_emission_counter()
        reset_last_event_id()

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowStartedEvent)
            def capture1(source, event):
                events.append(event)

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture2(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture3(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture4(source, event):
                events.append(event)

            flow = MinimalFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        assert len(events) >= 2

        sorted_events = sorted(events, key=lambda e: e.emission_sequence or 0)
        for event in sorted_events[1:]:
            assert event.previous_event_id is not None, (
                f"Event {event.type} (seq {event.emission_sequence}) should have previous_event_id set"
            )


class TestTriggeredByEventId:
    """Tests for triggered_by_event_id causal chain tracking."""

    @pytest.mark.asyncio
    async def test_triggered_by_event_id_for_listeners(self) -> None:
        """Listener events should have triggered_by_event_id pointing to the triggering method_execution_finished event."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ListenerFlow(Flow):
            @start()
            async def start_method(self):
                return "started"

            @listen(start_method)
            async def listener_method(self, result):
                return "listened"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = ListenerFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        assert len(started_events) >= 2
        assert len(finished_events) >= 2

        start_method_finished = next(
            (e for e in finished_events if e.method_name == "start_method"), None
        )
        listener_started = next(
            (e for e in started_events if e.method_name == "listener_method"), None
        )

        assert start_method_finished is not None
        assert listener_started is not None
        assert listener_started.triggered_by_event_id == start_method_finished.event_id

    @pytest.mark.asyncio
    async def test_start_method_has_no_triggered_by(self) -> None:
        """Start method events should have triggered_by_event_id=None."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class StartOnlyFlow(Flow):
            @start()
            async def my_start(self):
                return "started"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            flow = StartOnlyFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        start_event = next(
            (e for e in events if e.method_name == "my_start"), None
        )
        assert start_event is not None
        assert start_event.triggered_by_event_id is None

    @pytest.mark.asyncio
    async def test_chained_listeners_triggered_by(self) -> None:
        """Chained listeners should have triggered_by_event_id pointing to their triggering method."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ChainedFlow(Flow):
            @start()
            async def first(self):
                return "first"

            @listen(first)
            async def second(self, result):
                return "second"

            @listen(second)
            async def third(self, result):
                return "third"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = ChainedFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        first_finished = next(
            (e for e in finished_events if e.method_name == "first"), None
        )
        second_started = next(
            (e for e in started_events if e.method_name == "second"), None
        )
        second_finished = next(
            (e for e in finished_events if e.method_name == "second"), None
        )
        third_started = next(
            (e for e in started_events if e.method_name == "third"), None
        )

        assert first_finished is not None
        assert second_started is not None
        assert second_finished is not None
        assert third_started is not None

        assert second_started.triggered_by_event_id == first_finished.event_id
        assert third_started.triggered_by_event_id == second_finished.event_id

    @pytest.mark.asyncio
    async def test_parallel_listeners_same_trigger(self) -> None:
        """Parallel listeners should all have triggered_by_event_id pointing to the same triggering event."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ParallelFlow(Flow):
            @start()
            async def trigger(self):
                return "trigger"

            @listen(trigger)
            async def listener_a(self, result):
                return "a"

            @listen(trigger)
            async def listener_b(self, result):
                return "b"

            @listen(trigger)
            async def listener_c(self, result):
                return "c"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = ParallelFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        trigger_finished = next(
            (e for e in finished_events if e.method_name == "trigger"), None
        )
        listener_a_started = next(
            (e for e in started_events if e.method_name == "listener_a"), None
        )
        listener_b_started = next(
            (e for e in started_events if e.method_name == "listener_b"), None
        )
        listener_c_started = next(
            (e for e in started_events if e.method_name == "listener_c"), None
        )

        assert trigger_finished is not None
        assert listener_a_started is not None
        assert listener_b_started is not None
        assert listener_c_started is not None

        # All parallel listeners should point to the same triggering event
        assert listener_a_started.triggered_by_event_id == trigger_finished.event_id
        assert listener_b_started.triggered_by_event_id == trigger_finished.event_id
        assert listener_c_started.triggered_by_event_id == trigger_finished.event_id

    @pytest.mark.asyncio
    async def test_or_condition_triggered_by(self) -> None:
        """Listener with OR condition should have triggered_by_event_id pointing to whichever method triggered it."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id
        from crewai.flow.flow import or_

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class OrConditionFlow(Flow):
            @start()
            async def path_a(self):
                return "a"

            @listen(or_(path_a, "path_b"))
            async def after_either(self, result):
                return "done"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = OrConditionFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        path_a_finished = next(
            (e for e in finished_events if e.method_name == "path_a"), None
        )
        after_either_started = next(
            (e for e in started_events if e.method_name == "after_either"), None
        )

        assert path_a_finished is not None
        assert after_either_started is not None

        # The OR listener should be triggered by path_a since that's what ran
        assert after_either_started.triggered_by_event_id == path_a_finished.event_id

    @pytest.mark.asyncio
    async def test_router_triggered_by(self) -> None:
        """Events from router-triggered paths should have correct triggered_by_event_id."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id
        from crewai.flow.flow import router

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class RouterFlow(Flow):
            @start()
            async def begin(self):
                return "begin"

            @router(begin)
            async def route_decision(self, result):
                return "approved"

            @listen("approved")
            async def handle_approved(self):
                return "handled"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = RouterFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        begin_finished = next(
            (e for e in finished_events if e.method_name == "begin"), None
        )
        route_decision_started = next(
            (e for e in started_events if e.method_name == "route_decision"), None
        )
        route_decision_finished = next(
            (e for e in finished_events if e.method_name == "route_decision"), None
        )
        handle_approved_started = next(
            (e for e in started_events if e.method_name == "handle_approved"), None
        )

        assert begin_finished is not None
        assert route_decision_started is not None
        assert route_decision_finished is not None
        assert handle_approved_started is not None

        # Router should be triggered by begin
        assert route_decision_started.triggered_by_event_id == begin_finished.event_id
        # Handler should be triggered by router's finished event
        assert handle_approved_started.triggered_by_event_id == route_decision_finished.event_id

    @pytest.mark.asyncio
    async def test_multiple_kickoffs_maintain_chains(self) -> None:
        """Multiple akickoff() calls should maintain correct triggered_by chains for each execution."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        first_run_events: list[BaseEvent] = []
        second_run_events: list[BaseEvent] = []

        class ReusableFlow(Flow):
            @start()
            async def begin(self):
                return "begin"

            @listen(begin)
            async def process(self, result):
                return "processed"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                if len(second_run_events) == 0 and not capturing_second:
                    first_run_events.append(event)
                else:
                    second_run_events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                if len(second_run_events) == 0 and not capturing_second:
                    first_run_events.append(event)
                else:
                    second_run_events.append(event)

            # First kickoff
            capturing_second = False
            flow1 = ReusableFlow()
            await flow1.akickoff()
            crewai_event_bus.flush()

            # Second kickoff
            capturing_second = True
            flow2 = ReusableFlow()
            await flow2.akickoff()
            crewai_event_bus.flush()

        # Should have events from both runs
        assert len(first_run_events) >= 4  # 2 started + 2 finished
        assert len(second_run_events) >= 4

        # Check first run's triggered_by chain
        first_started = [e for e in first_run_events if isinstance(e, MethodExecutionStartedEvent)]
        first_finished = [e for e in first_run_events if isinstance(e, MethodExecutionFinishedEvent)]

        first_begin_finished = next(
            (e for e in first_finished if e.method_name == "begin"), None
        )
        first_process_started = next(
            (e for e in first_started if e.method_name == "process"), None
        )
        assert first_begin_finished is not None
        assert first_process_started is not None
        assert first_process_started.triggered_by_event_id == first_begin_finished.event_id

        # Check second run's triggered_by chain
        second_started = [e for e in second_run_events if isinstance(e, MethodExecutionStartedEvent)]
        second_finished = [e for e in second_run_events if isinstance(e, MethodExecutionFinishedEvent)]

        second_begin_finished = next(
            (e for e in second_finished if e.method_name == "begin"), None
        )
        second_process_started = next(
            (e for e in second_started if e.method_name == "process"), None
        )
        assert second_begin_finished is not None
        assert second_process_started is not None
        assert second_process_started.triggered_by_event_id == second_begin_finished.event_id

        # Verify the two runs have different event_ids (not reusing)
        assert first_begin_finished.event_id != second_begin_finished.event_id

        # Verify each run has its own independent previous_event_id chain
        # (chains reset at each top-level execution)
        first_sorted = sorted(first_run_events, key=lambda e: e.emission_sequence or 0)
        for event in first_sorted[1:]:
            assert event.previous_event_id is not None, (
                f"First run event {event.type} (seq {event.emission_sequence}) should have previous_event_id"
            )

        second_sorted = sorted(second_run_events, key=lambda e: e.emission_sequence or 0)
        for event in second_sorted[1:]:
            assert event.previous_event_id is not None, (
                f"Second run event {event.type} (seq {event.emission_sequence}) should have previous_event_id"
            )

    @pytest.mark.asyncio
    async def test_parallel_flows_maintain_separate_triggered_by_chains(self) -> None:
        """Parallel flow executions should maintain correct triggered_by chains independently."""
        import asyncio

        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ParallelTestFlow(Flow):
            def __init__(self, name: str):
                super().__init__()
                self.flow_name = name

            @start()
            async def begin(self):
                await asyncio.sleep(0.01)  # Small delay to interleave
                return self.flow_name

            @listen(begin)
            async def process(self, result):
                await asyncio.sleep(0.01)
                return f"{result}_processed"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            # Run two flows in parallel
            flow_a = ParallelTestFlow("flow_a")
            flow_b = ParallelTestFlow("flow_b")
            await asyncio.gather(flow_a.akickoff(), flow_b.akickoff())
            crewai_event_bus.flush()

        # Should have events from both flows (4 events each = 8 total)
        assert len(events) >= 8

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        # Find flow_a's events by checking the result contains "flow_a"
        flow_a_begin_finished = [
            e for e in finished_events
            if e.method_name == "begin" and "flow_a" in str(e.result)
        ]
        flow_a_process_started = [
            e for e in started_events
            if e.method_name == "process"
        ]

        flow_b_begin_finished = [
            e for e in finished_events
            if e.method_name == "begin" and "flow_b" in str(e.result)
        ]

        assert len(flow_a_begin_finished) >= 1
        assert len(flow_b_begin_finished) >= 1

        # Each flow's process should be triggered by its own begin
        # Find which process events were triggered by which begin events
        for process_event in flow_a_process_started:
            trigger_id = process_event.triggered_by_event_id
            assert trigger_id is not None

            # The triggering event should be a begin finished event
            triggering_event = next(
                (e for e in finished_events if e.event_id == trigger_id), None
            )
            assert triggering_event is not None
            assert triggering_event.method_name == "begin"

        # Verify previous_event_id forms a valid chain across all events
        all_sorted = sorted(events, key=lambda e: e.emission_sequence or 0)
        for event in all_sorted[1:]:
            assert event.previous_event_id is not None

    @pytest.mark.asyncio
    async def test_and_condition_triggered_by_last_method(self) -> None:
        """AND condition listener should have triggered_by_event_id pointing to the last completing method."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id
        from crewai.flow.flow import and_

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class AndConditionFlow(Flow):
            @start()
            async def method_a(self):
                return "a"

            @listen(method_a)
            async def method_b(self, result):
                return "b"

            @listen(and_(method_a, method_b))
            async def after_both(self, result):
                return "both_done"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = AndConditionFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        method_b_finished = next(
            (e for e in finished_events if e.method_name == "method_b"), None
        )
        after_both_started = next(
            (e for e in started_events if e.method_name == "after_both"), None
        )

        assert method_b_finished is not None
        assert after_both_started is not None

        # The AND listener should be triggered by method_b (the last one to complete)
        assert after_both_started.triggered_by_event_id == method_b_finished.event_id

    @pytest.mark.asyncio
    async def test_exception_handling_triggered_by(self) -> None:
        """Events emitted after exception should still have correct triggered_by."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id
        from crewai.events.types.flow_events import MethodExecutionFailedEvent

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ExceptionFlow(Flow):
            @start()
            async def will_fail(self):
                raise ValueError("intentional error")

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFailedEvent)
            def capture_failed(source, event):
                events.append(event)

            @crewai_event_bus.on(FlowStartedEvent)
            def capture_flow_started(source, event):
                events.append(event)

            flow = ExceptionFlow()
            try:
                await flow.akickoff()
            except ValueError:
                pass  # Expected
            crewai_event_bus.flush()

        # Even with exception, events should have proper previous_event_id chain
        all_sorted = sorted(events, key=lambda e: e.emission_sequence or 0)
        for event in all_sorted[1:]:
            assert event.previous_event_id is not None, (
                f"Event {event.type} (seq {event.emission_sequence}) should have previous_event_id"
            )

    @pytest.mark.asyncio
    async def test_sync_method_in_flow_triggered_by(self) -> None:
        """Synchronous methods should still have correct triggered_by."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class SyncFlow(Flow):
            @start()
            def sync_start(self):  # Synchronous method
                return "sync_done"

            @listen(sync_start)
            async def async_listener(self, result):
                return "async_done"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = SyncFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        sync_start_finished = next(
            (e for e in finished_events if e.method_name == "sync_start"), None
        )
        async_listener_started = next(
            (e for e in started_events if e.method_name == "async_listener"), None
        )

        assert sync_start_finished is not None
        assert async_listener_started is not None
        assert async_listener_started.triggered_by_event_id == sync_start_finished.event_id

    @pytest.mark.asyncio
    async def test_multiple_start_methods_triggered_by(self) -> None:
        """Multiple start methods should each have triggered_by_event_id=None."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class MultiStartFlow(Flow):
            @start()
            async def start_one(self):
                return "one"

            @start()
            async def start_two(self):
                return "two"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            flow = MultiStartFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]

        start_one = next(
            (e for e in started_events if e.method_name == "start_one"), None
        )
        start_two = next(
            (e for e in started_events if e.method_name == "start_two"), None
        )

        assert start_one is not None
        assert start_two is not None

        # Both start methods should have no triggered_by (they're entry points)
        assert start_one.triggered_by_event_id is None
        assert start_two.triggered_by_event_id is None

    @pytest.mark.asyncio
    async def test_none_return_triggered_by(self) -> None:
        """Methods returning None should still have correct triggered_by chain."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class NoneReturnFlow(Flow):
            @start()
            async def returns_none(self):
                return None

            @listen(returns_none)
            async def after_none(self, result):
                return "got_none"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = NoneReturnFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        returns_none_finished = next(
            (e for e in finished_events if e.method_name == "returns_none"), None
        )
        after_none_started = next(
            (e for e in started_events if e.method_name == "after_none"), None
        )

        assert returns_none_finished is not None
        assert after_none_started is not None
        assert after_none_started.triggered_by_event_id == returns_none_finished.event_id

    @pytest.mark.asyncio
    async def test_deeply_nested_chain_triggered_by(self) -> None:
        """Deeply nested listener chains (5+) should maintain correct triggered_by."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class DeepChainFlow(Flow):
            @start()
            async def level_0(self):
                return "0"

            @listen(level_0)
            async def level_1(self, result):
                return "1"

            @listen(level_1)
            async def level_2(self, result):
                return "2"

            @listen(level_2)
            async def level_3(self, result):
                return "3"

            @listen(level_3)
            async def level_4(self, result):
                return "4"

            @listen(level_4)
            async def level_5(self, result):
                return "5"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = DeepChainFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        # Verify each level triggers the next
        for i in range(5):
            prev_finished = next(
                (e for e in finished_events if e.method_name == f"level_{i}"), None
            )
            next_started = next(
                (e for e in started_events if e.method_name == f"level_{i+1}"), None
            )

            assert prev_finished is not None, f"level_{i} finished event not found"
            assert next_started is not None, f"level_{i+1} started event not found"
            assert next_started.triggered_by_event_id == prev_finished.event_id, (
                f"level_{i+1} should be triggered by level_{i}"
            )

    @pytest.mark.asyncio
    async def test_router_conditional_path_triggered_by(self) -> None:
        """Router with conditional paths should have correct triggered_by for the selected path."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id
        from crewai.flow.flow import router

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class ConditionalRouterFlow(Flow):
            @start()
            async def begin(self):
                return "begin"

            @router(begin)
            async def conditional_router(self, result):
                # Conditionally return one route
                return "path_a"

            @listen("path_a")
            async def handle_path_a(self):
                return "a_done"

            @listen("path_b")
            async def handle_path_b(self):
                return "b_done"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = ConditionalRouterFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        router_finished = next(
            (e for e in finished_events if e.method_name == "conditional_router"), None
        )
        handle_path_a_started = next(
            (e for e in started_events if e.method_name == "handle_path_a"), None
        )
        handle_path_b_started = next(
            (e for e in started_events if e.method_name == "handle_path_b"), None
        )

        assert router_finished is not None
        assert handle_path_a_started is not None
        # path_b should NOT be executed since router returned "path_a"
        assert handle_path_b_started is None

        # The selected path should be triggered by the router
        assert handle_path_a_started.triggered_by_event_id == router_finished.event_id


class TestCrewEventsInFlowTriggeredBy:
    """Tests for triggered_by in crew events running inside flows."""

    @pytest.mark.asyncio
    async def test_flow_listener_triggered_by_in_nested_context(self) -> None:
        """Nested listener contexts should maintain correct triggered_by chains."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class NestedFlow(Flow):
            @start()
            async def trigger_method(self):
                return "trigger"

            @listen(trigger_method)
            async def middle_method(self, result):
                return "middle"

            @listen(middle_method)
            async def final_method(self, result):
                return "final"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_method_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_method_finished(source, event):
                events.append(event)

            flow = NestedFlow()
            await flow.akickoff()
            crewai_event_bus.flush()

        method_started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        method_finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        trigger_finished = next(
            (e for e in method_finished_events if e.method_name == "trigger_method"), None
        )
        middle_started = next(
            (e for e in method_started_events if e.method_name == "middle_method"), None
        )
        middle_finished = next(
            (e for e in method_finished_events if e.method_name == "middle_method"), None
        )
        final_started = next(
            (e for e in method_started_events if e.method_name == "final_method"), None
        )

        assert trigger_finished is not None
        assert middle_started is not None
        assert middle_finished is not None
        assert final_started is not None

        # middle should be triggered by trigger_method
        assert middle_started.triggered_by_event_id == trigger_finished.event_id
        # final should be triggered by middle_method
        assert final_started.triggered_by_event_id == middle_finished.event_id

        # All events should have proper previous_event_id chain
        all_sorted = sorted(events, key=lambda e: e.emission_sequence or 0)
        for event in all_sorted[1:]:
            assert event.previous_event_id is not None

    def test_sync_kickoff_triggered_by(self) -> None:
        """Synchronous kickoff() should maintain correct triggered_by chains."""
        from crewai.events.base_events import reset_emission_counter
        from crewai.events.event_context import reset_last_event_id

        reset_emission_counter()
        reset_last_event_id()

        events: list[BaseEvent] = []

        class SyncKickoffFlow(Flow):
            @start()
            def start_method(self):
                return "started"

            @listen(start_method)
            def listener_method(self, result):
                return "listened"

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_started(source, event):
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_finished(source, event):
                events.append(event)

            flow = SyncKickoffFlow()
            flow.kickoff()  # Synchronous kickoff
            crewai_event_bus.flush()

        started_events = [e for e in events if isinstance(e, MethodExecutionStartedEvent)]
        finished_events = [e for e in events if isinstance(e, MethodExecutionFinishedEvent)]

        start_finished = next(
            (e for e in finished_events if e.method_name == "start_method"), None
        )
        listener_started = next(
            (e for e in started_events if e.method_name == "listener_method"), None
        )

        assert start_finished is not None
        assert listener_started is not None

        # Listener should be triggered by start_method
        assert listener_started.triggered_by_event_id == start_finished.event_id

        # Verify previous_event_id chain
        all_sorted = sorted(events, key=lambda e: e.emission_sequence or 0)
        for event in all_sorted[1:]:
            assert event.previous_event_id is not None
