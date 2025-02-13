"""Test Flow creation and execution basic functionality."""

import asyncio
from datetime import datetime

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.flow.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)


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


def test_flow_with_thread_lock():
    """Test that Flow properly handles thread locks in state."""
    import threading
    
    class LockFlow(Flow):
        def __init__(self):
            super().__init__()
            self.lock = threading.RLock()
            self.counter = 0
            
        @start()
        async def step_1(self):
            with self.lock:
                self.counter += 1
                return "step 1"
                
        @listen(step_1)
        async def step_2(self, result):
            with self.lock:
                self.counter += 1
                return result + " -> step 2"

    flow = LockFlow()
    result = flow.kickoff()
    
    assert result == "step 1 -> step 2"
    assert flow.counter == 2


def test_flow_with_nested_locks():
    """Test that Flow properly handles nested thread locks."""
    import threading
    
    class NestedLockFlow(Flow):
        def __init__(self):
            super().__init__()
            self.outer_lock = threading.RLock()
            self.inner_lock = threading.RLock()
            self.counter = 0
            
        @start()
        async def step_1(self):
            with self.outer_lock:
                with self.inner_lock:
                    self.counter += 1
                    return "step 1"
                
        @listen(step_1)
        async def step_2(self, result):
            with self.outer_lock:
                with self.inner_lock:
                    self.counter += 1
                    return result + " -> step 2"

    flow = NestedLockFlow()
    result = flow.kickoff()
    
    assert result == "step 1 -> step 2"
    assert flow.counter == 2


@pytest.mark.asyncio
async def test_flow_with_async_locks():
    """Test that Flow properly handles locks in async context."""
    import asyncio
    import threading
    
    class AsyncLockFlow(Flow):
        def __init__(self):
            super().__init__()
            self.lock = threading.RLock()
            self.async_lock = asyncio.Lock()
            self.counter = 0
            
        @start()
        async def step_1(self):
            async with self.async_lock:
                with self.lock:
                    self.counter += 1
                    return "step 1"
                
        @listen(step_1)
        async def step_2(self, result):
            async with self.async_lock:
                with self.lock:
                    self.counter += 1
                    return result + " -> step 2"

    flow = AsyncLockFlow()
    result = await flow.kickoff_async()
    
    assert result == "step 1 -> step 2"
    assert flow.counter == 2


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
            pass

    event_log = []

    def handle_event(_, event):
        event_log.append(event)

    flow = PoemFlow()
    flow.event_emitter.connect(handle_event)
    flow.kickoff(inputs={"separator": ", "})

    assert isinstance(event_log[0], FlowStartedEvent)
    assert event_log[0].flow_name == "PoemFlow"
    assert event_log[0].inputs == {"separator": ", "}
    assert isinstance(event_log[0].timestamp, datetime)

    # Asserting for concurrent start method executions in a for loop as you
    # can't guarantee ordering in asynchronous executions
    for i in range(1, 5):
        event = event_log[i]
        assert isinstance(event.state, dict)
        assert isinstance(event.state["id"], str)

        if event.method_name == "prepare_flower":
            if isinstance(event, MethodExecutionStartedEvent):
                assert event.params == {}
                assert event.state["separator"] == ", "
            elif isinstance(event, MethodExecutionFinishedEvent):
                assert event.result == "foo"
                assert event.state["flower"] == "roses"
                assert event.state["separator"] == ", "
            else:
                assert False, "Unexpected event type for prepare_flower"
        elif event.method_name == "prepare_color":
            if isinstance(event, MethodExecutionStartedEvent):
                assert event.params == {}
                assert event.state["separator"] == ", "
            elif isinstance(event, MethodExecutionFinishedEvent):
                assert event.result == "bar"
                assert event.state["color"] == "red"
                assert event.state["separator"] == ", "
            else:
                assert False, "Unexpected event type for prepare_color"
        else:
            assert False, f"Unexpected method {event.method_name} in prepare events"

    assert isinstance(event_log[5], MethodExecutionStartedEvent)
    assert event_log[5].method_name == "write_first_sentence"
    assert event_log[5].params == {}
    assert isinstance(event_log[5].state, dict)
    assert event_log[5].state["flower"] == "roses"
    assert event_log[5].state["color"] == "red"
    assert event_log[5].state["separator"] == ", "

    assert isinstance(event_log[6], MethodExecutionFinishedEvent)
    assert event_log[6].method_name == "write_first_sentence"
    assert event_log[6].result == "roses are red"

    assert isinstance(event_log[7], MethodExecutionStartedEvent)
    assert event_log[7].method_name == "finish_poem"
    assert event_log[7].params == {"_0": "roses are red"}
    assert isinstance(event_log[7].state, dict)
    assert event_log[7].state["flower"] == "roses"
    assert event_log[7].state["color"] == "red"

    assert isinstance(event_log[8], MethodExecutionFinishedEvent)
    assert event_log[8].method_name == "finish_poem"
    assert event_log[8].result == "roses are red, violets are blue"

    assert isinstance(event_log[9], MethodExecutionStartedEvent)
    assert event_log[9].method_name == "save_poem_to_database"
    assert event_log[9].params == {}
    assert isinstance(event_log[9].state, dict)
    assert event_log[9].state["flower"] == "roses"
    assert event_log[9].state["color"] == "red"

    assert isinstance(event_log[10], MethodExecutionFinishedEvent)
    assert event_log[10].method_name == "save_poem_to_database"
    assert event_log[10].result is None

    assert isinstance(event_log[11], FlowFinishedEvent)
    assert event_log[11].flow_name == "PoemFlow"
    assert event_log[11].result is None
    assert isinstance(event_log[11].timestamp, datetime)


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

    event_log = []

    def handle_event(_, event):
        event_log.append(event)

    flow = OnboardingFlow()
    flow.event_emitter.connect(handle_event)
    flow.kickoff(inputs={"name": "Anakin"})

    assert isinstance(event_log[0], FlowStartedEvent)
    assert event_log[0].flow_name == "OnboardingFlow"
    assert event_log[0].inputs == {"name": "Anakin"}
    assert isinstance(event_log[0].timestamp, datetime)

    assert isinstance(event_log[1], MethodExecutionStartedEvent)
    assert event_log[1].method_name == "user_signs_up"

    assert isinstance(event_log[2], MethodExecutionFinishedEvent)
    assert event_log[2].method_name == "user_signs_up"

    assert isinstance(event_log[3], MethodExecutionStartedEvent)
    assert event_log[3].method_name == "send_welcome_message"
    assert event_log[3].params == {}
    assert getattr(event_log[3].state, "sent") is False

    assert isinstance(event_log[4], MethodExecutionFinishedEvent)
    assert event_log[4].method_name == "send_welcome_message"
    assert getattr(event_log[4].state, "sent") is True
    assert event_log[4].result == "Welcome, Anakin!"

    assert isinstance(event_log[5], FlowFinishedEvent)
    assert event_log[5].flow_name == "OnboardingFlow"
    assert event_log[5].result == "Welcome, Anakin!"
    assert isinstance(event_log[5].timestamp, datetime)


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
    flow.event_emitter.connect(handle_event)
    flow.kickoff()

    assert isinstance(event_log[0], FlowStartedEvent)
    assert event_log[0].flow_name == "StatelessFlow"
    assert event_log[0].inputs is None
    assert isinstance(event_log[0].timestamp, datetime)

    assert isinstance(event_log[1], MethodExecutionStartedEvent)
    assert event_log[1].method_name == "init"

    assert isinstance(event_log[2], MethodExecutionFinishedEvent)
    assert event_log[2].method_name == "init"

    assert isinstance(event_log[3], MethodExecutionStartedEvent)
    assert event_log[3].method_name == "process"

    assert isinstance(event_log[4], MethodExecutionFinishedEvent)
    assert event_log[4].method_name == "process"

    assert isinstance(event_log[5], FlowFinishedEvent)
    assert event_log[5].flow_name == "StatelessFlow"
    assert (
        event_log[5].result
        == "Deeds will not be less valiant because they are unpraised."
    )
    assert isinstance(event_log[5].timestamp, datetime)
