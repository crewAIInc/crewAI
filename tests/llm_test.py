from time import sleep

import pytest

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import LLM
from crewai.tools import tool
from crewai.utilities.token_counter_callback import TokenCalcHandler


# TODO: This test fails without print statement, which makes me think that something is happening asynchronously that we need to eventually fix and dive deeper into at a later date
@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm1 = LLM(model="gpt-4o-mini")
    llm2 = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    result1 = llm1.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    print("result1:", result1)
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()
    print("usage_metrics_1:", usage_metrics_1)

    result2 = llm2.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    sleep(5)
    print("result2:", result2)
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()
    print("usage_metrics_2:", usage_metrics_2)

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_string_input():
    llm = LLM(model="gpt-4o-mini")

    # Test the call method with a string input
    result = llm.call("Return the name of a random city in the world.")
    assert isinstance(result, str)
    assert len(result.strip()) > 0  # Ensure the response is not empty


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_string_input_and_callbacks():
    llm = LLM(model="gpt-4o-mini")
    calc_handler = TokenCalcHandler(token_cost_process=TokenProcess())

    # Test the call method with a string input and callbacks
    result = llm.call(
        "Tell me a joke.",
        callbacks=[calc_handler],
    )
    usage_metrics = calc_handler.token_cost_process.get_summary()

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert usage_metrics.successful_requests == 1


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_message_list():
    llm = LLM(model="gpt-4o-mini")
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    # Test the call method with a list of messages
    result = llm.call(messages)
    assert isinstance(result, str)
    assert "Paris" in result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_tool_and_string_input():
    llm = LLM(model="gpt-4o-mini")

    def get_current_year() -> str:
        """Returns the current year as a string."""
        from datetime import datetime

        return str(datetime.now().year)

    # Create tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_current_year",
            "description": "Returns the current year as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    # Available functions mapping
    available_functions = {"get_current_year": get_current_year}

    # Test the call method with a string input and tool
    result = llm.call(
        "What is the current year?",
        tools=[tool_schema],
        available_functions=available_functions,
    )

    assert isinstance(result, str)
    assert result == get_current_year()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_call_with_tool_and_message_list():
    llm = LLM(model="gpt-4o-mini")

    def square_number(number: int) -> int:
        """Returns the square of a number."""
        return number * number

    # Create tool schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": "square_number",
            "description": "Returns the square of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "The number to square"}
                },
                "required": ["number"],
            },
        },
    }

    # Available functions mapping
    available_functions = {"square_number": square_number}

    messages = [{"role": "user", "content": "What is the square of 5?"}]

    # Test the call method with messages and tool
    result = llm.call(
        messages,
        tools=[tool_schema],
        available_functions=available_functions,
    )

    assert isinstance(result, int)
    assert result == 25
