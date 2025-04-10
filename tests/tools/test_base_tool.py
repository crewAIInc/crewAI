import asyncio
import inspect
import unittest
from typing import Any, Callable, Dict, List
from unittest.mock import patch

from crewai.tools import BaseTool, tool

# Import the necessary classes
from crewai.tools.base_tool import BaseTool


def test_creating_a_tool_using_annotation():
    @tool("Name of my tool")
    def my_tool(question: str) -> str:
        """Clear description for what this tool is useful for, your agent will need this information to use it."""
        return question

    # Assert all the right attributes were defined
    assert my_tool.name == "Name of my tool"
    assert (
        my_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert (
        my_tool.func("What is the meaning of life?") == "What is the meaning of life?"
    )

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert (
        converted_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
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

    assert (
        my_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    assert my_tool.args_schema.model_json_schema()["properties"] == {
        "question": {"title": "Question", "type": "string"}
    }
    assert my_tool.run("What is the meaning of life?") == "What is the meaning of life?"

    converted_tool = my_tool.to_structured_tool()
    assert converted_tool.name == "Name of my tool"

    assert (
        converted_tool.description
        == "Tool Name: Name of my tool\nTool Arguments: {'question': {'description': None, 'type': 'str'}}\nTool Description: Clear description for what this tool is useful for, your agent will need this information to use it."
    )
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


class BaseToolCoroutineTest(unittest.TestCase):
    def test_sync_run_returns_direct_result(self):
        """Test that _run in a synchronous tool returns a direct result, not a coroutine."""
        tool = SyncTool()
        
        # Call _run directly
        result = tool._run(input_text="hello")
        
        # Verify it's not a coroutine
        self.assertFalse(asyncio.iscoroutine(result))
        self.assertEqual(result, "Processed hello synchronously")
        
        # Verify that run method returns the expected result
        run_result = tool.run(input_text="hello")
        self.assertEqual(run_result, "Processed hello synchronously")
    
    def test_async_run_returns_coroutine(self):
        """Test that _run in an asynchronous tool returns a coroutine object."""
        tool = AsyncTool()
        
        # Call _run directly
        result = tool._run(input_text="hello")
        
        # Verify it's a coroutine
        self.assertTrue(asyncio.iscoroutine(result))
        
        # Cancel the coroutine to avoid warnings
        result.close()
    

class BaseToolCoroutineTest(unittest.TestCase):
    def test_run_calls_asyncio_run_for_async_tools(self):
        """Test that asyncio.run is called when using async tools."""
        # Test with async tool
        async_tool = AsyncTool()
        
        # Patch asyncio.run for this specific test
        with patch('asyncio.run') as mock_run:
            # Setup the mock to pass through to the real implementation
            mock_run.return_value = "Processed test asynchronously"
            
            # Call the tool's run method
            async_result = async_tool.run(input_text="test")
            
            # Verify asyncio.run was called exactly once
            mock_run.assert_called_once()
            
            # Verify the result is correct
            self.assertEqual(async_result, "Processed test asynchronously")
    
    def test_run_does_not_call_asyncio_run_for_sync_tools(self):
        """Test that asyncio.run is NOT called when using sync tools."""
        # Test with sync tool
        sync_tool = SyncTool()
        
        # Patch asyncio.run for this specific test
        with patch('asyncio.run') as mock_run:
            # Setup the mock to pass through to the real implementation

            # Call the tool's run method
            sync_result = sync_tool.run(input_text="test")
            
            # Verify asyncio.run was NOT called
            mock_run.assert_not_called()
            
            # Verify the result is correct
            self.assertEqual(sync_result, "Processed test synchronously")
