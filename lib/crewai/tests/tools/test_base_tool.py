import asyncio
from typing import Callable
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

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
    assert "Tool Name: name_of_my_tool" in my_tool.description
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

    assert "Tool Name: name_of_my_tool" in converted_tool.description
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

    assert "Tool Name: name_of_my_tool" in my_tool.description
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

    assert "Tool Name: name_of_my_tool" in converted_tool.description
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


# =============================================================================
# Schema Validation in run() Tests
# =============================================================================


class CodeExecutorInput(BaseModel):
    code: str = Field(description="The code to execute")
    language: str = Field(default="python", description="Programming language")


class CodeExecutorTool(BaseTool):
    name: str = "code_executor"
    description: str = "Execute code snippets"
    args_schema: type[BaseModel] = CodeExecutorInput

    def _run(self, code: str, language: str = "python") -> str:
        return f"Executed {language}: {code}"


class TestBaseToolRunValidation:
    """Tests for args_schema validation in BaseTool.run()."""

    def test_run_with_valid_kwargs_passes_validation(self) -> None:
        """Valid keyword arguments should pass schema validation and execute."""
        t = CodeExecutorTool()
        result = t.run(code="print('hello')")
        assert result == "Executed python: print('hello')"

    def test_run_with_all_kwargs_passes_validation(self) -> None:
        """All keyword arguments including optional ones should pass."""
        t = CodeExecutorTool()
        result = t.run(code="console.log('hi')", language="javascript")
        assert result == "Executed javascript: console.log('hi')"

    def test_run_with_missing_required_kwarg_raises(self) -> None:
        """Missing required kwargs should raise ValueError from schema validation."""
        t = CodeExecutorTool()
        with pytest.raises(ValueError, match="validation failed"):
            t.run(language="python")

    def test_run_with_wrong_field_name_raises(self) -> None:
        """Kwargs not matching any schema field should trigger validation error
        for missing required fields."""
        t = CodeExecutorTool()
        with pytest.raises(ValueError, match="validation failed"):
            t.run(wrong_arg="value")

    def test_run_with_positional_args_skips_validation(self) -> None:
        """Positional-arg calls should bypass schema validation (backwards compat)."""
        class SimpleTool(BaseTool):
            name: str = "simple"
            description: str = "A simple tool"

            def _run(self, question: str) -> str:
                return question

        t = SimpleTool()
        result = t.run("What is life?")
        assert result == "What is life?"

    def test_run_strips_extra_kwargs_from_llm(self) -> None:
        """Extra kwargs not in the schema should be silently stripped,
        preventing unexpected-keyword crashes in _run."""
        t = CodeExecutorTool()
        result = t.run(code="1+1", extra_hallucinated_field="junk")
        assert result == "Executed python: 1+1"

    def test_run_increments_usage_after_validation(self) -> None:
        """Usage count should still increment after validated execution."""
        t = CodeExecutorTool()
        assert t.current_usage_count == 0
        t.run(code="x = 1")
        assert t.current_usage_count == 1

    def test_run_does_not_increment_usage_on_validation_error(self) -> None:
        """Usage count should NOT increment when validation fails."""
        t = CodeExecutorTool()
        assert t.current_usage_count == 0
        with pytest.raises(ValueError):
            t.run(wrong="bad")
        assert t.current_usage_count == 0


class TestToolDecoratorRunValidation:
    """Tests for args_schema validation in Tool.run() (decorator-based tools)."""

    def test_decorator_tool_run_validates_kwargs(self) -> None:
        """Decorator-created tools should also validate kwargs against schema."""
        @tool("execute_code")
        def execute_code(code: str, language: str = "python") -> str:
            """Execute a code snippet."""
            return f"Executed {language}: {code}"

        result = execute_code.run(code="x = 1")
        assert result == "Executed python: x = 1"

    def test_decorator_tool_run_rejects_missing_required(self) -> None:
        """Decorator tools should reject missing required args via validation."""
        @tool("execute_code")
        def execute_code(code: str) -> str:
            """Execute a code snippet."""
            return f"Executed: {code}"

        with pytest.raises(ValueError, match="validation failed"):
            execute_code.run(wrong_arg="value")

    def test_decorator_tool_positional_args_still_work(self) -> None:
        """Positional args to decorator tools should bypass validation."""
        @tool("greet")
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        result = greet.run("World")
        assert result == "Hello, World!"


# =============================================================================
# Async arun() Schema Validation Tests
# =============================================================================


class AsyncCodeExecutorTool(BaseTool):
    name: str = "async_code_executor"
    description: str = "Execute code snippets asynchronously"
    args_schema: type[BaseModel] = CodeExecutorInput

    async def _arun(self, code: str, language: str = "python") -> str:
        return f"Async executed {language}: {code}"

    def _run(self, code: str, language: str = "python") -> str:
        return f"Executed {language}: {code}"


class TestBaseToolArunValidation:
    """Tests for args_schema validation in BaseTool.arun()."""

    @pytest.mark.asyncio
    async def test_arun_with_valid_kwargs_passes_validation(self) -> None:
        """Valid keyword arguments should pass schema validation in arun."""
        t = AsyncCodeExecutorTool()
        result = await t.arun(code="print('hello')")
        assert result == "Async executed python: print('hello')"

    @pytest.mark.asyncio
    async def test_arun_with_missing_required_kwarg_raises(self) -> None:
        """Missing required kwargs should raise ValueError in arun."""
        t = AsyncCodeExecutorTool()
        with pytest.raises(ValueError, match="validation failed"):
            await t.arun(language="python")

    @pytest.mark.asyncio
    async def test_arun_with_wrong_field_name_raises(self) -> None:
        """Kwargs not matching schema fields should trigger validation error in arun."""
        t = AsyncCodeExecutorTool()
        with pytest.raises(ValueError, match="validation failed"):
            await t.arun(wrong_arg="value")

    @pytest.mark.asyncio
    async def test_arun_strips_extra_kwargs(self) -> None:
        """Extra kwargs not in the schema should be stripped in arun."""
        t = AsyncCodeExecutorTool()
        result = await t.arun(code="1+1", extra_field="junk")
        assert result == "Async executed python: 1+1"

    @pytest.mark.asyncio
    async def test_arun_does_not_increment_usage_on_validation_error(self) -> None:
        """Usage count should NOT increment when arun validation fails."""
        t = AsyncCodeExecutorTool()
        assert t.current_usage_count == 0
        with pytest.raises(ValueError):
            await t.arun(wrong="bad")
        assert t.current_usage_count == 0


class TestToolDecoratorArunValidation:
    """Tests for args_schema validation in Tool.arun() (decorator-based async tools)."""

    @pytest.mark.asyncio
    async def test_async_decorator_tool_arun_validates_kwargs(self) -> None:
        """Async decorator tools should validate kwargs in arun."""
        @tool("async_execute")
        async def async_execute(code: str, language: str = "python") -> str:
            """Execute code asynchronously."""
            return f"Async {language}: {code}"

        result = await async_execute.arun(code="x = 1")
        assert result == "Async python: x = 1"

    @pytest.mark.asyncio
    async def test_async_decorator_tool_arun_rejects_missing_required(self) -> None:
        """Async decorator tools should reject missing required args in arun."""
        @tool("async_execute")
        async def async_execute(code: str) -> str:
            """Execute code asynchronously."""
            return f"Async: {code}"

        with pytest.raises(ValueError, match="validation failed"):
            await async_execute.arun(wrong_arg="value")
