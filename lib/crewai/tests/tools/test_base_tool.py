import asyncio
from typing import Callable
from unittest.mock import patch

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools import BaseTool, tool


def test_creating_a_tool_using_annotation():
    @tool("Name of my tool")
    def my_tool(question: str) -> str:
        """Clear description for what this tool is useful for, your agent will need this information to use it."""
        return question

    # Assert all the right attributes were defined
    assert my_tool.name == "Name of my tool"
    assert "Tool Name: Name of my tool" in my_tool.description
    assert "Tool Arguments:" in my_tool.description
    assert '"question"' in my_tool.description
    assert '"type": "string"' in my_tool.description
    assert "Tool Description: Clear description for what this tool is useful for" in my_tool.description
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        my_tool.func("What is the meaning of life?") == "What is the meaning of life?"
    )

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert "Tool Name: Name of my tool" in converted_tool.description
    assert "Tool Arguments:" in converted_tool.description
    assert '"question"' in converted_tool.description
    assert converted_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        converted_tool.func("What is the meaning of life?")
        == "What is the meaning of life?"
    )


def test_creating_a_tool_using_baseclass():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert my_tool.name == "Name of my tool"

    assert "Tool Name: Name of my tool" in my_tool.description
    assert "Tool Arguments:" in my_tool.description
    assert '"question"' in my_tool.description
    assert '"type": "string"' in my_tool.description
    assert "Tool Description: Clear description for what this tool is useful for" in my_tool.description
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert my_tool.run("What is the meaning of life?") == "What is the meaning of life?"

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert "Tool Name: Name of my tool" in converted_tool.description
    assert "Tool Arguments:" in converted_tool.description
    assert '"question"' in converted_tool.description
    assert converted_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        converted_tool._run("What is the meaning of life?")
        == "What is the meaning of life?"
    )


def test_setting_cache_function():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."
        cache_function: Callable = lambda: False

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert not my_tool.cache_function()


def test_default_cache_function_is_true():
    class MyCustomTool(BaseTool):
        name: str = "Name of my tool"
        description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

        def _run(self, question: str) -> str:
            return question

    my_tool = MyCustomTool()
    # Assert all the right attributes were defined
    assert my_tool.cache_function()


def test_result_as_answer_in_tool_decorator():
    @tool("Tool with result as answer", result_as_answer=True)
    def my_tool_with_result_as_answer(question: str) -> str:
        """This tool will return its result as the final answer."""
        return question

    assert my_tool_with_result_as_answer.result_as_answer is True

    converted_tool = my_tool_with_result_as_answer.to_structured_tool()
    assert converted_tool.result_as_answer is True

    @tool("Tool with default result_as_answer")
    def my_tool_with_default(question: str) -> str:
        """This tool uses the default result_as_answer value."""
        return question

    assert my_tool_with_default.result_as_answer is False

    converted_tool = my_tool_with_default.to_structured_tool()
    assert converted_tool.result_as_answer is False


class SyncTool(BaseTool):
    """Test implementation with a synchronous _run method"""

    name: str = "sync_tool"
    description: str = "A synchronous tool for testing"

    def _run(self, input_text: str) -> str:
        """Process input text synchronously."""
        return f"Processed {input_text} synchronously"


class AsyncTool(BaseTool):
    """Test implementation with an asynchronous _run method"""

    name: str = "async_tool"
    description: str = "An asynchronous tool for testing"

    async def _run(self, input_text: str) -> str:
        """Process input text asynchronously."""
        await asyncio.sleep(0.1)  # Simulate async operation
        return f"Processed {input_text} asynchronously"


def test_sync_run_returns_direct_result():
    """Test that _run in a synchronous tool returns a direct result, not a coroutine."""
    tool = SyncTool()
    result = tool._run(input_text="hello")

    assert not asyncio.iscoroutine(result)
    assert result == "Processed hello synchronously"

    run_result = tool.run(input_text="hello")
    assert run_result == "Processed hello synchronously"


def test_async_run_returns_coroutine():
    """Test that _run in an asynchronous tool returns a coroutine object."""
    tool = AsyncTool()
    result = tool._run(input_text="hello")

    assert asyncio.iscoroutine(result)
    result.close()  # Clean up the coroutine


def test_run_calls_asyncio_run_for_async_tools():
    """Test that asyncio.run is called when using async tools."""
    async_tool = AsyncTool()

    with patch("asyncio.run") as mock_run:
        mock_run.return_value = "Processed test asynchronously"
        async_result = async_tool.run(input_text="test")

        mock_run.assert_called_once()
        assert async_result == "Processed test asynchronously"


def test_run_does_not_call_asyncio_run_for_sync_tools():
    """Test that asyncio.run is NOT called when using sync tools."""
    sync_tool = SyncTool()

    with patch("asyncio.run") as mock_run:
        sync_result = sync_tool.run(input_text="test")

        mock_run.assert_not_called()
        assert sync_result == "Processed test synchronously"


@pytest.mark.vcr()
def test_max_usage_count_is_respected():
    class IteratingTool(BaseTool):
        name: str = "iterating_tool"
        description: str = "A tool that iterates a given number of times"

        def _run(self, input_text: str):
            return f"Iteration {input_text}"

    tool = IteratingTool(max_usage_count=5)

    agent = Agent(
        role="Iterating Agent",
        goal="Call the iterating tool 5 times",
        backstory="You are an agent that iterates a given number of times",
        tools=[tool],
    )

    task = Task(
        description="Call the iterating tool 5 times",
        expected_output="A list of the iterations",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
    )

    crew.kickoff()
    assert tool.max_usage_count == 5
    assert tool.current_usage_count == 5
