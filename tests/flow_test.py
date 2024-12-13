"""Test Flow creation and execution basic functionality."""

import asyncio

import pytest

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
