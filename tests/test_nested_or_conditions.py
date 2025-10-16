"""Test nested or_() conditions in Flow execution (Issue #3719)."""

import pytest

from crewai.flow.flow import Flow, listen, or_, router, start


def test_nested_or_condition_triggers_once():
    """Test that nested or_() conditions only trigger listeners once.
    
    This test addresses issue #3719 where nested or_() conditions would
    cause listeners to execute multiple times instead of once.
    
    Setup:
        method_5 listens to or_(method_1, or_(method_2, method_3))
        method_7 listens to or_(method_5, method_6)
    
    Expected behavior:
        - method_5 should execute exactly once (triggered by first matching condition)
        - method_7 should execute exactly once (triggered by first matching condition)
    
    Bug behavior (before fix):
        - method_5 executed 3 times (once for method_1, method_2, and method_3)
        - method_7 executed 4 times (once for each method_5 execution + method_6)
    """
    execution_order = []

    class NestedOrFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")
            return "method_1_done"

        @listen("method_1")
        def method_2(self):
            execution_order.append("method_2")
            return "method_2_done"

        @listen("method_1")
        def method_3(self):
            execution_order.append("method_3")
            return "method_3_done"

        @listen(or_("method_1", or_("method_2", "method_3")))
        def method_5(self):
            execution_order.append("method_5")
            return "method_5_done"

        @listen("method_1")
        def method_6(self):
            execution_order.append("method_6")
            return "method_6_done"

        @listen(or_("method_5", "method_6"))
        def method_7(self):
            execution_order.append("method_7")
            return "method_7_done"

    flow = NestedOrFlow()
    flow.kickoff()

    assert execution_order.count("method_5") == 1, (
        f"method_5 should execute exactly once, but executed {execution_order.count('method_5')} times"
    )
    assert execution_order.count("method_7") == 1, (
        f"method_7 should execute exactly once, but executed {execution_order.count('method_7')} times"
    )
    
    assert "method_1" in execution_order
    assert "method_2" in execution_order
    assert "method_3" in execution_order
    assert "method_5" in execution_order
    assert "method_6" in execution_order
    assert "method_7" in execution_order


def test_simple_or_condition_triggers_once():
    """Test that simple or_() conditions only trigger once.
    
    Even without nesting, an OR condition should only trigger a listener once,
    not multiple times for each method in the OR list.
    """
    execution_order = []

    class SimpleOrFlow(Flow):
        @start()
        def method_a(self):
            execution_order.append("method_a")

        @listen("method_a")
        def method_b(self):
            execution_order.append("method_b")

        @listen("method_a")
        def method_c(self):
            execution_order.append("method_c")

        @listen(or_("method_b", "method_c"))
        def method_d(self):
            execution_order.append("method_d")

    flow = SimpleOrFlow()
    flow.kickoff()

    assert execution_order.count("method_d") == 1, (
        f"method_d should execute exactly once, but executed {execution_order.count('method_d')} times"
    )


def test_or_condition_with_three_methods():
    """Test OR condition with three methods triggers only once."""
    execution_order = []

    class ThreeMethodOrFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")

        @listen("method_1")
        def method_2(self):
            execution_order.append("method_2")

        @listen("method_1")
        def method_3(self):
            execution_order.append("method_3")

        @listen("method_1")
        def method_4(self):
            execution_order.append("method_4")

        @listen(or_("method_2", "method_3", "method_4"))
        def method_5(self):
            execution_order.append("method_5")

    flow = ThreeMethodOrFlow()
    flow.kickoff()

    assert execution_order.count("method_5") == 1, (
        f"method_5 should execute exactly once, but executed {execution_order.count('method_5')} times"
    )


def test_multiple_or_listeners_independent():
    """Test that multiple OR listeners are independent of each other."""
    execution_order = []

    class MultipleOrFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")

        @listen("method_1")
        def method_2(self):
            execution_order.append("method_2")

        @listen("method_1")
        def method_3(self):
            execution_order.append("method_3")

        @listen(or_("method_2", "method_3"))
        def method_a(self):
            execution_order.append("method_a")

        @listen(or_("method_2", "method_3"))
        def method_b(self):
            execution_order.append("method_b")

    flow = MultipleOrFlow()
    flow.kickoff()

    assert execution_order.count("method_a") == 1
    assert execution_order.count("method_b") == 1


def test_deeply_nested_or_conditions():
    """Test deeply nested or_() conditions."""
    execution_order = []

    class DeeplyNestedOrFlow(Flow):
        @start()
        def start_method(self):
            execution_order.append("start_method")

        @listen("start_method")
        def method_a(self):
            execution_order.append("method_a")

        @listen("start_method")
        def method_b(self):
            execution_order.append("method_b")

        @listen("start_method")
        def method_c(self):
            execution_order.append("method_c")

        @listen("start_method")
        def method_d(self):
            execution_order.append("method_d")

        @listen(or_(or_("method_a", "method_b"), or_("method_c", "method_d")))
        def final_method(self):
            execution_order.append("final_method")

    flow = DeeplyNestedOrFlow()
    flow.kickoff()

    assert execution_order.count("final_method") == 1, (
        f"final_method should execute exactly once, but executed {execution_order.count('final_method')} times"
    )


def test_or_condition_execution_order():
    """Test that OR listener executes after first matching condition.
    
    The listener should trigger as soon as any one of the OR conditions is met,
    not wait for all of them.
    """
    execution_order = []

    class ExecutionOrderFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")

        @listen("method_1")
        def method_2(self):
            execution_order.append("method_2")

        @listen("method_1")
        def method_3(self):
            execution_order.append("method_3")

        @listen(or_("method_2", "method_3"))
        def method_4(self):
            execution_order.append("method_4")

    flow = ExecutionOrderFlow()
    flow.kickoff()

    method_4_index = execution_order.index("method_4")
    
    assert "method_2" in execution_order[:method_4_index] or "method_3" in execution_order[:method_4_index], (
        "method_4 should execute after at least one of method_2 or method_3"
    )


def test_or_condition_with_single_method():
    """Test OR condition with a single method (edge case)."""
    execution_order = []

    class SingleMethodOrFlow(Flow):
        @start()
        def method_1(self):
            execution_order.append("method_1")

        @listen(or_("method_1"))
        def method_2(self):
            execution_order.append("method_2")

    flow = SingleMethodOrFlow()
    flow.kickoff()

    assert execution_order == ["method_1", "method_2"]
    assert execution_order.count("method_2") == 1


def test_cyclic_flow_with_or_condition():
    """Test that OR conditions work correctly in cyclic flows.
    
    Within a single flow execution, an OR listener should only trigger once
    even if multiple methods in the OR condition complete.
    """
    execution_order = []

    class CyclicOrFlow(Flow):
        @start()
        def step_1(self):
            execution_order.append("step_1")

        @listen("step_1")
        def step_2(self):
            execution_order.append("step_2")

        @listen("step_1")
        def step_3(self):
            execution_order.append("step_3")

        @listen(or_("step_2", "step_3"))
        def step_4(self):
            execution_order.append("step_4")

    flow = CyclicOrFlow()
    flow.kickoff()

    assert execution_order.count("step_4") == 1, (
        f"step_4 should execute once (not once for each OR condition), but executed {execution_order.count('step_4')} times"
    )
