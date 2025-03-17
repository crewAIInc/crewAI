"""Test Flow creation and execution basic functionality."""

import asyncio
from datetime import datetime

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.utilities.events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
    crewai_event_bus,
)
from crewai.utilities.events.flow_events import FlowPlotEvent


def test_simple_sequential_flow():
    """Test a simple flow with two steps called sequentially."""
    execution_order = []

    class SimpleFlow(Flow):
        @start()
        def step_1(self):
            execution_order.append("step_1")

        @listen(step_1)
        def step_2(self):
            execution_order.append("step_2")

    flow = SimpleFlow()
    flow.kickoff()

    assert execution_order == ["step_1", "step_2"]


def test_flow_with_multiple_starts():
    """Test a flow with multiple start methods."""
    execution_order = []

    class MultiStartFlow(Flow):
        @start()
        def step_a(self):
            execution_order.append("step_a")

        @start()
        def step_b(self):
            execution_order.append("step_b")

        @listen(step_a)
        def step_c(self):
            execution_order.append("step_c")

        @listen(step_b)
        def step_d(self):
            execution_order.append("step_d")

    flow = MultiStartFlow()
    flow.kickoff()

    assert "step_a" in execution_order
    assert "step_b" in execution_order
    assert "step_c" in execution_order
    assert "step_d" in execution_order
    assert execution_order.index("step_c") > execution_order.index("step_a")
    assert execution_order.index("step_d") > execution_order.index("step_b")


def test_cyclic_flow():
    """Test a cyclic flow that runs a finite number of iterations."""
    execution_order = []

    class CyclicFlow(Flow):
        iteration = 0
        max_iterations = 3

        @start("loop")
        def step_1(self):
            if self.iteration >= self.max_iterations:
                return  # Do not proceed further
            execution_order.append(f"step_1_{self.iteration}")

        @listen(step_1)
        def step_2(self):
            execution_order.append(f"step_2_{self.iteration}")

        @router(step_2)
        def step_3(self):
            execution_order.append(f"step_3_{self.iteration}")
            self.iteration += 1
            if self.iteration < self.max_iterations:
                return "loop"

            return "exit"

    flow = CyclicFlow()
    flow.kickoff()

    expected_order = []
    for i in range(flow.max_iterations):
        expected_order.extend([f"step_1_{i}", f"step_2_{i}", f"step_3_{i}"])

    assert execution_order == expected_order


def test_flow_with_and_condition():
    """Test a flow where a step waits for multiple other steps to complete."""
    execution_order = []

    class AndConditionFlow(Flow):
        @start()
        def step_1(self):
            execution_order.append("step_1")

        @start()
        def step_2(self):
            execution_order.append("step_2")

        @listen(and_(step_1, step_2))
        def step_3(self):
            execution_order.append("step_3")

    flow = AndConditionFlow()
    flow.kickoff()

    assert "step_1" in execution_order
    assert "step_2" in execution_order
    assert execution_order[-1] == "step_3"
    assert execution_order.index("step_3") > execution_order.index("step_1")
    assert execution_order.index("step_3") > execution_order.index("step_2")


def test_flow_with_or_condition():
    """Test a flow where a step is triggered when any of multiple steps complete."""
    execution_order = []

    class OrConditionFlow(Flow):
        @start()
        def step_a(self):
            execution_order.append("step_a")

        @start()
        def step_b(self):
            execution_order.append("step_b")

        @listen(or_(step_a, step_b))
        def step_c(self):
            execution_order.append("step_c")

    flow = OrConditionFlow()
    flow.kickoff()

    assert "step_a" in execution_order or "step_b" in execution_order
    assert "step_c" in execution_order
    assert execution_order.index("step_c") > min(
        execution_order.index("step_a"), execution_order.index("step_b")
    )


def test_flow_with_router():
    """Test a flow that uses a router method to determine the next step."""
    execution_order = []

    class RouterFlow(Flow):
        @start()
        def start_method(self):
            execution_order.append("start_method")

        @router(start_method)
        def router(self):
            execution_order.append("router")
            # Ensure the condition is set to True to follow the "step_if_true" path
            condition = True
            return "step_if_true" if condition else "step_if_false"

        @listen("step_if_true")
        def truthy(self):
            execution_order.append("step_if_true")

        @listen("step_if_false")
        def falsy(self):
            execution_order.append("step_if_false")

    flow = RouterFlow()
    flow.kickoff()

    assert execution_order == ["start_method", "router", "step_if_true"]


def test_async_flow():
    """Test an asynchronous flow."""
    execution_order = []

    class AsyncFlow(Flow):
        @start()
        async def step_1(self):
            execution_order.append("step_1")
            await asyncio.sleep(0.1)

        @listen(step_1)
        async def step_2(self):
            execution_order.append("step_2")
            await asyncio.sleep(0.1)

    flow = AsyncFlow()
    asyncio.run(flow.kickoff_async())

    assert execution_order == ["step_1", "step_2"]


def test_flow_with_exceptions():
    """Test flow behavior when exceptions occur in steps."""
    execution_order = []

    class ExceptionFlow(Flow):
        @start()
        def step_1(self):
            execution_order.append("step_1")
            raise ValueError("An error occurred in step_1")

        @listen(step_1)
        def step_2(self):
            execution_order.append("step_2")

    flow = ExceptionFlow()

    with pytest.raises(ValueError):
        flow.kickoff()

    # Ensure step_2 did not execute
    assert execution_order == ["step_1"]


def test_flow_restart():
    """Test restarting a flow after it has completed."""
    execution_order = []

    class RestartableFlow(Flow):
        @start()
        def step_1(self):
            execution_order.append("step_1")

        @listen(step_1)
        def step_2(self):
            execution_order.append("step_2")

    flow = RestartableFlow()
    flow.kickoff()
    flow.kickoff()  # Restart the flow

    assert execution_order == ["step_1", "step_2", "step_1", "step_2"]


def test_flow_with_custom_state():
    """Test a flow that maintains and modifies internal state."""

    class StateFlow(Flow):
        def __init__(self):
            super().__init__()
            self.counter = 0

        @start()
        def step_1(self):
            self.counter += 1

        @listen(step_1)
        def step_2(self):
            self.counter *= 2
            assert self.counter == 2

    flow = StateFlow()
    flow.kickoff()
    assert flow.counter == 2


def test_flow_uuid_unstructured():
    """Test that unstructured (dictionary) flow states automatically get a UUID that persists."""
    initial_id = None

    class UUIDUnstructuredFlow(Flow):
        @start()
        def first_method(self):
            nonlocal initial_id
            # Verify ID is automatically generated
            assert "id" in self.state
            assert isinstance(self.state["id"], str)
            # Store initial ID for comparison
            initial_id = self.state["id"]
            # Add some data to trigger state update
            self.state["data"] = "example"

        @listen(first_method)
        def second_method(self):
            # Ensure the ID persists after state updates
            assert "id" in self.state
            assert self.state["id"] == initial_id
            # Update state again to verify ID preservation
            self.state["more_data"] = "test"
            assert self.state["id"] == initial_id

    flow = UUIDUnstructuredFlow()
    flow.kickoff()
    # Verify ID persists after flow completion
    assert flow.state["id"] == initial_id
    # Verify UUID format (36 characters, including hyphens)
    assert len(flow.state["id"]) == 36


def test_flow_uuid_structured():
    """Test that structured (Pydantic) flow states automatically get a UUID that persists."""
    initial_id = None

    class MyStructuredState(BaseModel):
        counter: int = 0
        message: str = "initial"

    class UUIDStructuredFlow(Flow[MyStructuredState]):
        @start()
        def first_method(self):
            nonlocal initial_id
            # Verify ID is automatically generated and accessible as attribute
            assert hasattr(self.state, "id")
            assert isinstance(self.state.id, str)
            # Store initial ID for comparison
            initial_id = self.state.id
            # Update some fields to trigger state changes
            self.state.counter += 1
            self.state.message = "updated"

        @listen(first_method)
        def second_method(self):
            # Ensure the ID persists after state updates
            assert hasattr(self.state, "id")
            assert self.state.id == initial_id
            # Update state again to verify ID preservation
            self.state.counter += 1
            self.state.message = "final"
            assert self.state.id == initial_id

    flow = UUIDStructuredFlow()
    flow.kickoff()
    # Verify ID persists after flow completion
    assert flow.state.id == initial_id
    # Verify UUID format (36 characters, including hyphens)
    assert len(flow.state.id) == 36
    # Verify other state fields were properly updated
    assert flow.state.counter == 2
    assert flow.state.message == "final"


def test_router_with_multiple_conditions():
    """Test a router that triggers when any of multiple steps complete (OR condition),
    and another router that triggers only after all specified steps complete (AND condition).
    """

    execution_order = []

    class ComplexRouterFlow(Flow):
        @start()
        def step_a(self):
            execution_order.append("step_a")

        @start()
        def step_b(self):
            execution_order.append("step_b")

        @router(or_("step_a", "step_b"))
        def router_or(self):
            execution_order.append("router_or")
            return "next_step_or"

        @listen("next_step_or")
        def handle_next_step_or_event(self):
            execution_order.append("handle_next_step_or_event")

        @listen(handle_next_step_or_event)
        def branch_2_step(self):
            execution_order.append("branch_2_step")

        @router(and_(handle_next_step_or_event, branch_2_step))
        def router_and(self):
            execution_order.append("router_and")
            return "final_step"

        @listen("final_step")
        def log_final_step(self):
            execution_order.append("log_final_step")

    flow = ComplexRouterFlow()
    flow.kickoff()

    assert "step_a" in execution_order
    assert "step_b" in execution_order
    assert "router_or" in execution_order
    assert "handle_next_step_or_event" in execution_order
    assert "branch_2_step" in execution_order
    assert "router_and" in execution_order
    assert "log_final_step" in execution_order

    # Check that the AND router triggered after both relevant steps:
    assert execution_order.index("router_and") > execution_order.index(
        "handle_next_step_or_event"
    )
    assert execution_order.index("router_and") > execution_order.index("branch_2_step")

    # final_step should run after router_and
    assert execution_order.index("log_final_step") > execution_order.index("router_and")


def test_unstructured_flow_event_emission():
    """Test that the correct events are emitted during unstructured flow
    execution with all fields validated."""

    class PoemFlow(Flow):
        @start()
        def prepare_flower(self):
            self.state["flower"] = "roses"
            return "foo"

        @start()
        def prepare_color(self):
            self.state["color"] = "red"
            return "bar"

        @listen(prepare_color)
        def write_first_sentence(self):
            return f"{self.state['flower']} are {self.state['color']}"

        @listen(write_first_sentence)
        def finish_poem(self, first_sentence):
            separator = self.state.get("separator", "\n")
            return separator.join([first_sentence, "violets are blue"])

        @listen(finish_poem)
        def save_poem_to_database(self):
            # A method without args/kwargs to ensure events are sent correctly
            return "roses are red\nviolets are blue"

    flow = PoemFlow()
    received_events = []

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        received_events.append(event)

    flow.kickoff(inputs={"separator": ", "})
    assert isinstance(received_events[0], FlowStartedEvent)
    assert received_events[0].flow_name == "PoemFlow"
    assert received_events[0].inputs == {"separator": ", "}
    assert isinstance(received_events[0].timestamp, datetime)

    # All subsequent events are MethodExecutionStartedEvent
    for event in received_events[1:-1]:
        assert isinstance(event, MethodExecutionStartedEvent)
        assert event.flow_name == "PoemFlow"
        assert isinstance(event.state, dict)
        assert isinstance(event.state["id"], str)
        assert event.state["separator"] == ", "

    assert received_events[1].method_name == "prepare_flower"
    assert received_events[1].params == {}
    assert "flower" not in received_events[1].state

    assert received_events[2].method_name == "prepare_color"
    assert received_events[2].params == {}
    print("received_events[2]", received_events[2])
    assert "flower" in received_events[2].state

    assert received_events[3].method_name == "write_first_sentence"
    assert received_events[3].params == {}
    assert received_events[3].state["flower"] == "roses"
    assert received_events[3].state["color"] == "red"

    assert received_events[4].method_name == "finish_poem"
    assert received_events[4].params == {"_0": "roses are red"}
    assert received_events[4].state["flower"] == "roses"
    assert received_events[4].state["color"] == "red"

    assert received_events[5].method_name == "save_poem_to_database"
    assert received_events[5].params == {}
    assert received_events[5].state["flower"] == "roses"
    assert received_events[5].state["color"] == "red"

    assert isinstance(received_events[6], FlowFinishedEvent)
    assert received_events[6].flow_name == "PoemFlow"
    assert received_events[6].result == "roses are red\nviolets are blue"
    assert isinstance(received_events[6].timestamp, datetime)


def test_structured_flow_event_emission():
    """Test that the correct events are emitted during structured flow
    execution with all fields validated."""

    class OnboardingState(BaseModel):
        name: str = ""
        sent: bool = False

    class OnboardingFlow(Flow[OnboardingState]):
        @start()
        def user_signs_up(self):
            self.state.sent = False

        @listen(user_signs_up)
        def send_welcome_message(self):
            self.state.sent = True
            return f"Welcome, {self.state.name}!"

    flow = OnboardingFlow()
    flow.kickoff(inputs={"name": "Anakin"})

    received_events = []

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def handle_method_end(source, event):
        received_events.append(event)

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        received_events.append(event)

    flow.kickoff(inputs={"name": "Anakin"})

    assert isinstance(received_events[0], FlowStartedEvent)
    assert received_events[0].flow_name == "OnboardingFlow"
    assert received_events[0].inputs == {"name": "Anakin"}
    assert isinstance(received_events[0].timestamp, datetime)

    assert isinstance(received_events[1], MethodExecutionStartedEvent)
    assert received_events[1].method_name == "user_signs_up"

    assert isinstance(received_events[2], MethodExecutionFinishedEvent)
    assert received_events[2].method_name == "user_signs_up"

    assert isinstance(received_events[3], MethodExecutionStartedEvent)
    assert received_events[3].method_name == "send_welcome_message"
    assert received_events[3].params == {}
    assert getattr(received_events[3].state, "sent") is False

    assert isinstance(received_events[4], MethodExecutionFinishedEvent)
    assert received_events[4].method_name == "send_welcome_message"
    assert getattr(received_events[4].state, "sent") is True
    assert received_events[4].result == "Welcome, Anakin!"

    assert isinstance(received_events[5], FlowFinishedEvent)
    assert received_events[5].flow_name == "OnboardingFlow"
    assert received_events[5].result == "Welcome, Anakin!"
    assert isinstance(received_events[5].timestamp, datetime)


def test_stateless_flow_event_emission():
    """Test that the correct events are emitted stateless during flow execution
    with all fields validated."""

    class StatelessFlow(Flow):
        @start()
        def init(self):
            pass

        @listen(init)
        def process(self):
            return "Deeds will not be less valiant because they are unpraised."

    event_log = []

    def handle_event(_, event):
        event_log.append(event)

    flow = StatelessFlow()
    received_events = []

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        received_events.append(event)

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def handle_method_end(source, event):
        received_events.append(event)

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        received_events.append(event)

    flow.kickoff()

    assert isinstance(received_events[0], FlowStartedEvent)
    assert received_events[0].flow_name == "StatelessFlow"
    assert received_events[0].inputs is None
    assert isinstance(received_events[0].timestamp, datetime)

    assert isinstance(received_events[1], MethodExecutionStartedEvent)
    assert received_events[1].method_name == "init"

    assert isinstance(received_events[2], MethodExecutionFinishedEvent)
    assert received_events[2].method_name == "init"

    assert isinstance(received_events[3], MethodExecutionStartedEvent)
    assert received_events[3].method_name == "process"

    assert isinstance(received_events[4], MethodExecutionFinishedEvent)
    assert received_events[4].method_name == "process"

    assert isinstance(received_events[5], FlowFinishedEvent)
    assert received_events[5].flow_name == "StatelessFlow"
    assert (
        received_events[5].result
        == "Deeds will not be less valiant because they are unpraised."
    )
    assert isinstance(received_events[5].timestamp, datetime)


def test_flow_plotting():
    class StatelessFlow(Flow):
        @start()
        def init(self):
            return "Initializing flow..."

        @listen(init)
        def process(self):
            return "Deeds will not be less valiant because they are unpraised."

    flow = StatelessFlow()
    flow.kickoff()
    received_events = []

    @crewai_event_bus.on(FlowPlotEvent)
    def handle_flow_plot(source, event):
        received_events.append(event)

    flow.plot("test_flow")

    assert len(received_events) == 1
    assert isinstance(received_events[0], FlowPlotEvent)
    assert received_events[0].flow_name == "StatelessFlow"
    assert isinstance(received_events[0].timestamp, datetime)


def test_multiple_routers_from_same_trigger():
    """Test that multiple routers triggered by the same method all activate their listeners."""
    execution_order = []

    class MultiRouterFlow(Flow):
        def __init__(self):
            super().__init__()
            # Set diagnosed conditions to trigger all routers
            self.state["diagnosed_conditions"] = "DHA"  # Contains D, H, and A

        @start()
        def scan_medical(self):
            execution_order.append("scan_medical")
            return "scan_complete"

        @router(scan_medical)
        def diagnose_conditions(self):
            execution_order.append("diagnose_conditions")
            return "diagnosis_complete"

        @router(diagnose_conditions)
        def diabetes_router(self):
            execution_order.append("diabetes_router")
            if "D" in self.state["diagnosed_conditions"]:
                return "diabetes"
            return None

        @listen("diabetes")
        def diabetes_analysis(self):
            execution_order.append("diabetes_analysis")
            return "diabetes_analysis_complete"

        @router(diagnose_conditions)
        def hypertension_router(self):
            execution_order.append("hypertension_router")
            if "H" in self.state["diagnosed_conditions"]:
                return "hypertension"
            return None

        @listen("hypertension")
        def hypertension_analysis(self):
            execution_order.append("hypertension_analysis")
            return "hypertension_analysis_complete"

        @router(diagnose_conditions)
        def anemia_router(self):
            execution_order.append("anemia_router")
            if "A" in self.state["diagnosed_conditions"]:
                return "anemia"
            return None

        @listen("anemia")
        def anemia_analysis(self):
            execution_order.append("anemia_analysis")
            return "anemia_analysis_complete"

    flow = MultiRouterFlow()
    flow.kickoff()

    # Verify all methods were called
    assert "scan_medical" in execution_order
    assert "diagnose_conditions" in execution_order

    # Verify all routers were called
    assert "diabetes_router" in execution_order
    assert "hypertension_router" in execution_order
    assert "anemia_router" in execution_order

    # Verify all listeners were called - this is the key test for the fix
    assert "diabetes_analysis" in execution_order
    assert "hypertension_analysis" in execution_order
    assert "anemia_analysis" in execution_order

    # Verify execution order constraints
    assert execution_order.index("diagnose_conditions") > execution_order.index(
        "scan_medical"
    )

    # All routers should execute after diagnose_conditions
    assert execution_order.index("diabetes_router") > execution_order.index(
        "diagnose_conditions"
    )
    assert execution_order.index("hypertension_router") > execution_order.index(
        "diagnose_conditions"
    )
    assert execution_order.index("anemia_router") > execution_order.index(
        "diagnose_conditions"
    )

    # All analyses should execute after their respective routers
    assert execution_order.index("diabetes_analysis") > execution_order.index(
        "diabetes_router"
    )
    assert execution_order.index("hypertension_analysis") > execution_order.index(
        "hypertension_router"
    )
    assert execution_order.index("anemia_analysis") > execution_order.index(
        "anemia_router"
    )
