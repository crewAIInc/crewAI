"""Test Flow creation and execution basic functionality."""

import asyncio
import threading
from datetime import datetime
from typing import Optional

import pytest
from pydantic import BaseModel

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowPlotEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow.flow import Flow, and_, listen, or_, router, start


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
    lock = threading.Lock()
    all_events_received = threading.Event()
    expected_event_count = (
        7  # 1 FlowStarted + 5 MethodExecutionStarted + 1 FlowFinished
    )

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    flow.kickoff(inputs={"separator": ", "})

    assert all_events_received.wait(timeout=5), "Timeout waiting for all flow events"

    # Sort events by timestamp to ensure deterministic order
    # (async handlers may append out of order)
    with lock:
        received_events.sort(key=lambda e: e.timestamp)

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


def test_flow_trigger_payload_injection():
    captured_payload = []

    class TriggerFlow(Flow):
        @start()
        def start_method(self, crewai_trigger_payload=None):
            captured_payload.append(crewai_trigger_payload)
            return "started"

        @listen(start_method)
        def second_method(self):
            captured_payload.append("no_parameter")
            return "finished"

    flow = TriggerFlow()

    test_payload = "This is important trigger data"
    flow.kickoff(inputs={"crewai_trigger_payload": test_payload})

    assert captured_payload == [test_payload, "no_parameter"]


def test_flow_trigger_payload_injection_multiple_starts():
    captured_payloads = []

    class MultiStartFlow(Flow):
        @start()
        def start_method_1(self, crewai_trigger_payload=None):
            captured_payloads.append(("start_1", crewai_trigger_payload))
            return "start_1_done"

        @start()
        def start_method_2(self, crewai_trigger_payload=None):
            captured_payloads.append(("start_2", crewai_trigger_payload))
            return "start_2_done"

    flow = MultiStartFlow()

    test_payload = "Multiple start trigger data"
    flow.kickoff(inputs={"crewai_trigger_payload": test_payload})

    assert captured_payloads == [("start_1", test_payload), ("start_2", test_payload)]


def test_flow_without_trigger_payload():
    captured_payload = []

    class NormalFlow(Flow):
        @start()
        def start_method(self, crewai_trigger_payload=None):
            captured_payload.append(crewai_trigger_payload)
            return "no_trigger"

    flow = NormalFlow()

    flow.kickoff(inputs={"other_data": "some value"})

    assert captured_payload[0] is None


def test_flow_trigger_payload_with_structured_state():
    class TriggerState(BaseModel):
        id: str = "test"
        message: str = ""

    class StructuredFlow(Flow[TriggerState]):
        @start()
        def start_method(self, crewai_trigger_payload=None):
            return crewai_trigger_payload

    flow = StructuredFlow()

    test_payload = "Structured state trigger data"
    result = flow.kickoff(inputs={"crewai_trigger_payload": test_payload})

    assert result == test_payload


def test_flow_start_method_without_trigger_parameter():
    execution_order = []

    class FlowWithoutParameter(Flow):
        @start()
        def start_without_param(self):
            execution_order.append("start_executed")
            return "started"

        @listen(start_without_param)
        def second_method(self):
            execution_order.append("second_executed")
            return "finished"

    flow = FlowWithoutParameter()

    result = flow.kickoff(inputs={"crewai_trigger_payload": "some data"})

    assert execution_order == ["start_executed", "second_executed"]
    assert result == "finished"


def test_async_flow_with_trigger_payload():
    captured_payload = []

    class AsyncTriggerFlow(Flow):
        @start()
        async def async_start_method(self, crewai_trigger_payload=None):
            captured_payload.append(crewai_trigger_payload)
            await asyncio.sleep(0.01)
            return "async_started"

        @listen(async_start_method)
        async def async_second_method(self, result):
            captured_payload.append(result)
            await asyncio.sleep(0.01)
            return "async_finished"

    flow = AsyncTriggerFlow()

    test_payload = "Async trigger data"
    result = asyncio.run(
        flow.kickoff_async(inputs={"crewai_trigger_payload": test_payload})
    )

    assert captured_payload == [test_payload, "async_started"]
    assert result == "async_finished"


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

    received_events = []
    lock = threading.Lock()
    all_events_received = threading.Event()
    expected_event_count = 6  # 1 FlowStarted + 2 MethodExecutionStarted + 2 MethodExecutionFinished + 1 FlowFinished

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def handle_method_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    flow.kickoff(inputs={"name": "Anakin"})

    assert all_events_received.wait(timeout=5), "Timeout waiting for all flow events"

    # Sort events by timestamp to ensure deterministic order
    with lock:
        received_events.sort(key=lambda e: e.timestamp)

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
    assert received_events[3].state["sent"] is False

    assert isinstance(received_events[4], MethodExecutionFinishedEvent)
    assert received_events[4].method_name == "send_welcome_message"
    assert received_events[4].state["sent"] is True
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
    lock = threading.Lock()
    all_events_received = threading.Event()
    expected_event_count = 6  # 1 FlowStarted + 2 MethodExecutionStarted + 2 MethodExecutionFinished + 1 FlowFinished

    @crewai_event_bus.on(FlowStartedEvent)
    def handle_flow_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(MethodExecutionStartedEvent)
    def handle_method_start(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(MethodExecutionFinishedEvent)
    def handle_method_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    @crewai_event_bus.on(FlowFinishedEvent)
    def handle_flow_end(source, event):
        with lock:
            received_events.append(event)
            if len(received_events) == expected_event_count:
                all_events_received.set()

    flow.kickoff()

    assert all_events_received.wait(timeout=5), "Timeout waiting for all flow events"

    # Sort events by timestamp to ensure deterministic order
    with lock:
        received_events.sort(key=lambda e: e.timestamp)

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
    event_received = threading.Event()

    @crewai_event_bus.on(FlowPlotEvent)
    def handle_flow_plot(source, event):
        received_events.append(event)
        event_received.set()

    flow.plot("test_flow")

    assert event_received.wait(timeout=5), "Timeout waiting for plot event"
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


def test_flow_name():
    class MyFlow(Flow):
        name = "MyFlow"

        @start()
        def start(self):
            return "Hello, world!"

    flow = MyFlow()
    assert flow.name == "MyFlow"


def test_nested_and_or_conditions():
    """Test nested conditions like or_(and_(A, B), and_(C, D)).

    Reproduces bug from issue #3719 where nested conditions are flattened,
    causing premature execution.
    """
    execution_order = []

    class NestedConditionFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")

        @listen(method_1)
        def method_2(self):
            execution_order.append("method_2")

        @router(method_2)
        def method_3(self):
            execution_order.append("method_3")
            # Choose b_condition path
            return "b_condition"

        @listen("b_condition")
        def method_5(self):
            execution_order.append("method_5")

        @listen(method_5)
        async def method_4(self):
            execution_order.append("method_4")

        @listen(or_("a_condition", "b_condition"))
        async def method_6(self):
            execution_order.append("method_6")

        @listen(
            or_(
                and_("a_condition", method_6),
                and_(method_6, method_4),
            )
        )
        def method_7(self):
            execution_order.append("method_7")

        @listen(method_7)
        async def method_8(self):
            execution_order.append("method_8")

    flow = NestedConditionFlow()
    flow.kickoff()

    # Verify execution happened
    assert "method_1" in execution_order
    assert "method_2" in execution_order
    assert "method_3" in execution_order
    assert "method_5" in execution_order
    assert "method_4" in execution_order
    assert "method_6" in execution_order
    assert "method_7" in execution_order
    assert "method_8" in execution_order

    # Critical assertion: method_7 should only execute AFTER both method_6 AND method_4
    # Since b_condition was returned, method_6 triggers on b_condition
    # method_7 requires: (a_condition AND method_6) OR (method_6 AND method_4)
    # The second condition (method_6 AND method_4) should be the one that triggers
    assert execution_order.index("method_7") > execution_order.index("method_6")
    assert execution_order.index("method_7") > execution_order.index("method_4")

    # method_8 should execute after method_7
    assert execution_order.index("method_8") > execution_order.index("method_7")


def test_diamond_dependency_pattern():
    """Test diamond pattern where two parallel paths converge at a final step."""
    execution_order = []

    class DiamondFlow(Flow):
        @start()
        def start(self):
            execution_order.append("start")
            return "started"

        @listen(start)
        def path_a(self):
            execution_order.append("path_a")
            return "a_done"

        @listen(start)
        def path_b(self):
            execution_order.append("path_b")
            return "b_done"

        @listen(and_(path_a, path_b))
        def converge(self):
            execution_order.append("converge")
            return "converged"

    flow = DiamondFlow()
    flow.kickoff()

    # Start should execute first
    assert execution_order[0] == "start"

    # Both paths should execute after start
    assert "path_a" in execution_order
    assert "path_b" in execution_order
    assert execution_order.index("path_a") > execution_order.index("start")
    assert execution_order.index("path_b") > execution_order.index("start")

    # Converge should be last and after both paths
    assert execution_order[-1] == "converge"
    assert execution_order.index("converge") > execution_order.index("path_a")
    assert execution_order.index("converge") > execution_order.index("path_b")


def test_router_cascade_chain():
    """Test a chain of routers where each router triggers the next."""
    execution_order = []

    class RouterCascadeFlow(Flow):
        def __init__(self):
            super().__init__()
            self.state["level"] = 1

        @start()
        def begin(self):
            execution_order.append("begin")
            return "started"

        @router(begin)
        def router_level_1(self):
            execution_order.append("router_level_1")
            return "level_1_path"

        @listen("level_1_path")
        def process_level_1(self):
            execution_order.append("process_level_1")
            self.state["level"] = 2
            return "level_1_done"

        @router(process_level_1)
        def router_level_2(self):
            execution_order.append("router_level_2")
            return "level_2_path"

        @listen("level_2_path")
        def process_level_2(self):
            execution_order.append("process_level_2")
            self.state["level"] = 3
            return "level_2_done"

        @router(process_level_2)
        def router_level_3(self):
            execution_order.append("router_level_3")
            return "final_path"

        @listen("final_path")
        def finalize(self):
            execution_order.append("finalize")
            return "complete"

    flow = RouterCascadeFlow()
    flow.kickoff()

    expected_order = [
        "begin",
        "router_level_1",
        "process_level_1",
        "router_level_2",
        "process_level_2",
        "router_level_3",
        "finalize",
    ]

    assert execution_order == expected_order
    assert flow.state["level"] == 3


def test_complex_and_or_branching():
    """Test complex branching with multiple AND and OR conditions."""
    execution_order = []

    class ComplexBranchingFlow(Flow):
        @start()
        def init(self):
            execution_order.append("init")

        @listen(init)
        def branch_1a(self):
            execution_order.append("branch_1a")

        @listen(init)
        def branch_1b(self):
            execution_order.append("branch_1b")

        @listen(init)
        def branch_1c(self):
            execution_order.append("branch_1c")

        # Requires 1a AND 1b (ignoring 1c)
        @listen(and_(branch_1a, branch_1b))
        def branch_2a(self):
            execution_order.append("branch_2a")

        # Requires any of 1a, 1b, or 1c
        @listen(or_(branch_1a, branch_1b, branch_1c))
        def branch_2b(self):
            execution_order.append("branch_2b")

        # Final step requires 2a AND 2b
        @listen(and_(branch_2a, branch_2b))
        def final(self):
            execution_order.append("final")

    flow = ComplexBranchingFlow()
    flow.kickoff()

    # Verify all branches executed
    assert "init" in execution_order
    assert "branch_1a" in execution_order
    assert "branch_1b" in execution_order
    assert "branch_1c" in execution_order
    assert "branch_2a" in execution_order
    assert "branch_2b" in execution_order
    assert "final" in execution_order

    # Verify order constraints
    assert execution_order.index("branch_2a") > execution_order.index("branch_1a")
    assert execution_order.index("branch_2a") > execution_order.index("branch_1b")

    # branch_2b should trigger after at least one of 1a, 1b, or 1c
    min_branch_1_index = min(
        execution_order.index("branch_1a"),
        execution_order.index("branch_1b"),
        execution_order.index("branch_1c"),
    )
    assert execution_order.index("branch_2b") > min_branch_1_index

    # Final should be last and after both 2a and 2b
    assert execution_order[-1] == "final"
    assert execution_order.index("final") > execution_order.index("branch_2a")
    assert execution_order.index("final") > execution_order.index("branch_2b")


def test_conditional_router_paths_exclusivity():
    """Test that only the returned router path activates, not all paths."""
    execution_order = []

    class ConditionalRouterFlow(Flow):
        def __init__(self):
            super().__init__()
            self.state["condition"] = "take_path_b"

        @start()
        def begin(self):
            execution_order.append("begin")

        @router(begin)
        def decision_point(self):
            execution_order.append("decision_point")
            if self.state["condition"] == "take_path_a":
                return "path_a"
            elif self.state["condition"] == "take_path_b":
                return "path_b"
            else:
                return "path_c"

        @listen("path_a")
        def handle_path_a(self):
            execution_order.append("handle_path_a")

        @listen("path_b")
        def handle_path_b(self):
            execution_order.append("handle_path_b")

        @listen("path_c")
        def handle_path_c(self):
            execution_order.append("handle_path_c")

    flow = ConditionalRouterFlow()
    flow.kickoff()

    # Should only execute path_b, not path_a or path_c
    assert "begin" in execution_order
    assert "decision_point" in execution_order
    assert "handle_path_b" in execution_order
    assert "handle_path_a" not in execution_order
    assert "handle_path_c" not in execution_order


def test_state_consistency_across_parallel_branches():
    """Test that state remains consistent when branches execute sequentially.

    Note: Branches triggered by the same parent execute sequentially, not in parallel.
    This ensures predictable state mutations and prevents race conditions.
    """
    execution_order = []

    class StateConsistencyFlow(Flow):
        def __init__(self):
            super().__init__()
            self.state["counter"] = 0
            self.state["branch_a_value"] = None
            self.state["branch_b_value"] = None

        @start()
        def init(self):
            execution_order.append("init")
            self.state["counter"] = 10

        @listen(init)
        def branch_a(self):
            execution_order.append("branch_a")
            # Read counter value
            self.state["branch_a_value"] = self.state["counter"]
            self.state["counter"] += 1

        @listen(init)
        def branch_b(self):
            execution_order.append("branch_b")
            # Read counter value
            self.state["branch_b_value"] = self.state["counter"]
            self.state["counter"] += 5

        @listen(and_(branch_a, branch_b))
        def verify_state(self):
            execution_order.append("verify_state")

    flow = StateConsistencyFlow()
    flow.kickoff()

    # Branches execute sequentially, so branch_a runs first, then branch_b
    assert flow.state["branch_a_value"] == 10  # Sees initial value
    assert flow.state["branch_b_value"] == 11  # Sees value after branch_a increment

    # Final counter should reflect both increments sequentially
    assert flow.state["counter"] == 16  # 10 + 1 + 5


def test_deeply_nested_conditions():
    """Test deeply nested AND/OR conditions to ensure proper evaluation."""
    execution_order = []

    class DeeplyNestedFlow(Flow):
        @start()
        def a(self):
            execution_order.append("a")

        @start()
        def b(self):
            execution_order.append("b")

        @start()
        def c(self):
            execution_order.append("c")

        @start()
        def d(self):
            execution_order.append("d")

        # Nested: (a AND b) OR (c AND d)
        @listen(or_(and_(a, b), and_(c, d)))
        def result(self):
            execution_order.append("result")

    flow = DeeplyNestedFlow()
    flow.kickoff()

    # All start methods should execute
    assert "a" in execution_order
    assert "b" in execution_order
    assert "c" in execution_order
    assert "d" in execution_order

    # Result should execute after all starts
    assert "result" in execution_order
    assert execution_order.index("result") > execution_order.index("a")
    assert execution_order.index("result") > execution_order.index("b")
    assert execution_order.index("result") > execution_order.index("c")
    assert execution_order.index("result") > execution_order.index("d")


def test_mixed_sync_async_execution_order():
    """Test that execution order is preserved with mixed sync/async methods."""
    execution_order = []

    class MixedSyncAsyncFlow(Flow):
        @start()
        def sync_start(self):
            execution_order.append("sync_start")

        @listen(sync_start)
        async def async_step_1(self):
            execution_order.append("async_step_1")
            await asyncio.sleep(0.01)

        @listen(async_step_1)
        def sync_step_2(self):
            execution_order.append("sync_step_2")

        @listen(sync_step_2)
        async def async_step_3(self):
            execution_order.append("async_step_3")
            await asyncio.sleep(0.01)

        @listen(async_step_3)
        def sync_final(self):
            execution_order.append("sync_final")

    flow = MixedSyncAsyncFlow()
    asyncio.run(flow.kickoff_async())

    expected_order = [
        "sync_start",
        "async_step_1",
        "sync_step_2",
        "async_step_3",
        "sync_final",
    ]

    assert execution_order == expected_order


def test_flow_copy_state_with_unpickleable_objects():
    """Test that _copy_state handles unpickleable objects like RLock.

    Regression test for issue #3828: Flow should not crash when state contains
    objects that cannot be deep copied (like threading.RLock).
    """

    class StateWithRLock(BaseModel):
        counter: int = 0
        lock: Optional[threading.RLock] = None

    class FlowWithRLock(Flow[StateWithRLock]):
        @start()
        def step_1(self):
            self.state.counter += 1

        @listen(step_1)
        def step_2(self):
            self.state.counter += 1

    flow = FlowWithRLock(initial_state=StateWithRLock())
    flow._state.lock = threading.RLock()

    copied_state = flow._copy_state()
    assert copied_state.counter == 0
    assert copied_state.lock is not None


def test_flow_copy_state_with_nested_unpickleable_objects():
    """Test that _copy_state handles unpickleable objects nested in containers.

    Regression test for issue #3828: Verifies that unpickleable objects
    nested inside dicts/lists in state don't cause crashes.
    """

    class NestedState(BaseModel):
        data: dict = {}
        items: list = []

    class FlowWithNestedUnpickleable(Flow[NestedState]):
        @start()
        def step_1(self):
            self.state.data["lock"] = threading.RLock()
            self.state.data["value"] = 42

        @listen(step_1)
        def step_2(self):
            self.state.items.append(threading.Lock())
            self.state.items.append("normal_value")

    flow = FlowWithNestedUnpickleable(initial_state=NestedState())
    flow.kickoff()

    assert flow.state.data["value"] == 42
    assert len(flow.state.items) == 2


def test_flow_copy_state_without_unpickleable_objects():
    """Test that _copy_state still works normally with pickleable objects.

    Ensures that the fallback logic doesn't break normal deep copy behavior.
    """

    class NormalState(BaseModel):
        counter: int = 0
        data: str = ""
        nested: dict = {}

    class NormalFlow(Flow[NormalState]):
        @start()
        def step_1(self):
            self.state.counter = 5
            self.state.data = "test"
            self.state.nested = {"key": "value"}

    flow = NormalFlow(initial_state=NormalState())
    flow.state.counter = 10
    flow.state.data = "modified"
    flow.state.nested["key"] = "modified"

    copied_state = flow._copy_state()
    assert copied_state.counter == 10
    assert copied_state.data == "modified"
    assert copied_state.nested["key"] == "modified"

    flow.state.nested["key"] = "changed_after_copy"
    assert copied_state.nested["key"] == "modified"


def test_flow_copy_state_with_dict_state():
    """Test that _copy_state works with dict-based states."""

    class DictFlow(Flow[dict]):
        @start()
        def step_1(self):
            self.state["counter"] = 1

    flow = DictFlow()
    flow.state["test"] = "value"

    copied_state = flow._copy_state()
    assert copied_state["test"] == "value"

    flow.state["test"] = "modified"
    assert copied_state["test"] == "value"


class TestFlowAkickoff:
    """Tests for the native async akickoff method."""

    @pytest.mark.asyncio
    async def test_akickoff_basic(self):
        """Test basic akickoff execution."""
        execution_order = []

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                execution_order.append("step_1")
                return "step_1_result"

            @listen(step_1)
            def step_2(self, result):
                execution_order.append("step_2")
                return "final_result"

        flow = SimpleFlow()
        result = await flow.akickoff()

        assert execution_order == ["step_1", "step_2"]
        assert result == "final_result"

    @pytest.mark.asyncio
    async def test_akickoff_with_inputs(self):
        """Test akickoff with inputs."""

        class InputFlow(Flow):
            @start()
            def process_input(self):
                return self.state.get("value", "default")

        flow = InputFlow()
        result = await flow.akickoff(inputs={"value": "custom_value"})

        assert result == "custom_value"

    @pytest.mark.asyncio
    async def test_akickoff_with_async_methods(self):
        """Test akickoff with async flow methods."""
        execution_order = []

        class AsyncMethodFlow(Flow):
            @start()
            async def async_step_1(self):
                execution_order.append("async_step_1")
                await asyncio.sleep(0.01)
                return "async_result"

            @listen(async_step_1)
            async def async_step_2(self, result):
                execution_order.append("async_step_2")
                await asyncio.sleep(0.01)
                return f"final_{result}"

        flow = AsyncMethodFlow()
        result = await flow.akickoff()

        assert execution_order == ["async_step_1", "async_step_2"]
        assert result == "final_async_result"

    @pytest.mark.asyncio
    async def test_akickoff_equivalent_to_kickoff_async(self):
        """Test that akickoff produces the same results as kickoff_async."""
        execution_order_akickoff = []
        execution_order_kickoff_async = []

        class TestFlow(Flow):
            def __init__(self, execution_list):
                super().__init__()
                self._execution_list = execution_list

            @start()
            def step_1(self):
                self._execution_list.append("step_1")
                return "result_1"

            @listen(step_1)
            def step_2(self, result):
                self._execution_list.append("step_2")
                return "result_2"

        flow1 = TestFlow(execution_order_akickoff)
        result1 = await flow1.akickoff()

        flow2 = TestFlow(execution_order_kickoff_async)
        result2 = await flow2.kickoff_async()

        assert execution_order_akickoff == execution_order_kickoff_async
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_akickoff_with_multiple_starts(self):
        """Test akickoff with multiple start methods."""
        execution_order = []

        class MultiStartFlow(Flow):
            @start()
            def start_a(self):
                execution_order.append("start_a")

            @start()
            def start_b(self):
                execution_order.append("start_b")

        flow = MultiStartFlow()
        await flow.akickoff()

        assert "start_a" in execution_order
        assert "start_b" in execution_order

    @pytest.mark.asyncio
    async def test_akickoff_with_router(self):
        """Test akickoff with router method."""
        execution_order = []

        class RouterFlow(Flow):
            @start()
            def begin(self):
                execution_order.append("begin")
                return "data"

            @router(begin)
            def route(self, data):
                execution_order.append("route")
                return "PATH_A"

            @listen("PATH_A")
            def handle_path_a(self):
                execution_order.append("path_a")
                return "path_a_result"

        flow = RouterFlow()
        result = await flow.akickoff()

        assert execution_order == ["begin", "route", "path_a"]
        assert result == "path_a_result"
