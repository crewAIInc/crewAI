"""Test @listen decorator for method vs output disambiguation."""

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, router, start


def test_listen_with_explicit_method():
    """Test @listen with explicit method parameter."""
    execution_order = []

    class ExplicitFlow(Flow):
        @start()
        def method_to_listen_for(self):
            execution_order.append("method_to_listen_for")
            return "method_output"

        @listen(method="method_to_listen_for")
        def explicit_method_listener(self):
            execution_order.append("explicit_method_listener")

    flow = ExplicitFlow()
    flow.kickoff()

    assert "method_to_listen_for" in execution_order
    assert "explicit_method_listener" in execution_order
    assert execution_order.index("explicit_method_listener") > execution_order.index("method_to_listen_for")


def test_listen_with_explicit_output():
    """Test @listen with explicit output parameter."""
    execution_order = []

    class ExplicitOutputFlow(Flow):
        @start()
        def start_method(self):
            execution_order.append("start_method")

        @router(start_method)
        def router_method(self):
            execution_order.append("router_method")
            return "output_value"

        @listen(output="output_value")
        def output_listener(self):
            execution_order.append("output_listener")

    flow = ExplicitOutputFlow()
    flow.kickoff()

    assert "start_method" in execution_order
    assert "router_method" in execution_order
    assert "output_listener" in execution_order
    assert execution_order.index("output_listener") > execution_order.index("router_method")


def test_ambiguous_case_with_explicit_parameters():
    """Test case where method name matches a possible output value."""
    import logging
    import asyncio
    import time
    logging.basicConfig(level=logging.DEBUG)
    
    execution_order = []
    
    class AmbiguousFlow(Flow):
        @start()
        def start_method(self):
            print("Executing start_method")
            execution_order.append("start_method")
            return "start output"
            
        @router(start_method)
        def router_method(self):
            print("Executing router_method")
            execution_order.append("router_method")
            return "ambiguous_name"
            
        def ambiguous_name(self):
            print("This method should not be called directly")
            execution_order.append("ambiguous_name_direct_call")
            return "should not happen"
            
        @listen(method="ambiguous_name")  # Listen to method name explicitly
        def method_listener(self):
            print("Executing method_listener")
            execution_order.append("method_listener")
            
        @listen(output="ambiguous_name")  # Listen to output string explicitly
        def output_listener(self):
            print("Executing output_listener")
            execution_order.append("output_listener")
    
    print("Creating flow instance")
    flow = AmbiguousFlow()
    
    async def run_with_timeout():
        task = asyncio.create_task(flow.kickoff_async())
        
        try:
            await asyncio.wait_for(task, timeout=5.0)  # 5 second timeout
        except asyncio.TimeoutError:
            print("Test timed out - likely an infinite loop")
            return False
        return True
    
    print("Starting flow kickoff with timeout")
    success = asyncio.run(run_with_timeout())
    
    print(f"Execution order: {execution_order}")
    
    if success:
        assert "start_method" in execution_order
        assert "router_method" in execution_order
        assert "output_listener" in execution_order
        
        assert "method_listener" not in execution_order
        
        assert "ambiguous_name_direct_call" not in execution_order
        
        assert execution_order.index("output_listener") > execution_order.index("router_method")
    else:
        pytest.fail("Test timed out - likely an infinite loop in the flow execution")


def test_listen_with_backward_compatibility():
    """Test that the old way of using @listen still works."""
    execution_order = []

    class BackwardCompatFlow(Flow):
        @start()
        def start_method(self):
            execution_order.append("start_method")

        @router(start_method)
        def router_method(self):
            execution_order.append("router_method")
            return "success"

        @listen("start_method")  # Old way - listen to method by name
        def method_listener(self):
            execution_order.append("method_listener")

        @listen("success")  # Old way - listen to output string
        def output_listener(self):
            execution_order.append("output_listener")

    flow = BackwardCompatFlow()
    flow.kickoff()

    assert "start_method" in execution_order
    assert "router_method" in execution_order
    assert "method_listener" in execution_order
    assert "output_listener" in execution_order


def test_listen_with_invalid_parameters():
    """Test that invalid parameters raise exceptions."""
    with pytest.raises(ValueError):
        @listen(method="method_name", output="output_value")
        def invalid_listener(self):
            pass

    with pytest.raises(ValueError):
        @listen()
        def no_param_listener(self):
            pass
