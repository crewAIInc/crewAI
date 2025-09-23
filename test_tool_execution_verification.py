#!/usr/bin/env python3
"""
Test suite for the Tool Execution Verification System
"""

import sys
import time
import unittest
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crewai.utilities.tool_execution_verifier import (
    ExecutionToken,
    ToolExecutionWrapper,
    execution_registry,
    verify_observation_token,
)


class TestWebSearchTool:
    """Mock web search tool for testing"""
    def __init__(self):
        self.call_count = 0

    def run(self, query: str) -> str:
        """Simulate a real web search"""
        self.call_count += 1
        time.sleep(0.001)  # Simulate minimal network delay
        return f"Search results for '{query}': Found 5 relevant articles"

class TestToolExecutionVerification(unittest.TestCase):
    """Test cases for the token-based execution verification system"""

    def setUp(self):
        """Set up test fixtures"""
        # Clear the registry for each test
        execution_registry._pending.clear()
        execution_registry._completed.clear()

    def test_legitimate_tool_execution(self):
        """Test that legitimate tool execution works and can be verified"""
        # Arrange
        tool = TestWebSearchTool()
        tool_wrapper = ToolExecutionWrapper(tool.run, "TestWebSearchTool")
        token = execution_registry.request_execution("TestWebSearchTool", "test_agent", "test_task")

        # Act
        result = tool_wrapper.execute_with_token(token, "AI in Healthcare")

        # Assert
        self.assertIn("Search results for 'AI in Healthcare'", result)
        self.assertEqual(tool.call_count, 1)

        # Verify the execution
        is_valid = verify_observation_token(token.token_id)
        self.assertTrue(is_valid)

    def test_fabricated_observation_detection(self):
        """Test that fabricated observations are correctly detected"""
        # Arrange
        fake_token_id = "test-token-id-for-validation"  # noqa: S105

        # Act
        is_valid = verify_observation_token(fake_token_id)

        # Assert
        self.assertFalse(is_valid)

    def test_invalid_token_execution(self):
        """Test that execution with invalid token fails"""
        # Arrange
        tool = TestWebSearchTool()
        tool_wrapper = ToolExecutionWrapper(tool.run, "TestWebSearchTool")
        fake_token = ExecutionToken(
            token_id="invalid-test-token-id",  # noqa: S106
            tool_name="TestWebSearchTool",
            agent_id="test_agent",
            task_id="test_task",
            timestamp=time.time()
        )

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            tool_wrapper.execute_with_token(fake_token, "AI in Healthcare")

        self.assertIn("Invalid or expired execution token", str(context.exception))
        self.assertEqual(tool.call_count, 0)

    def test_multiple_concurrent_executions(self):
        """Test multiple concurrent tool executions"""
        # Arrange
        tools = [TestWebSearchTool() for _ in range(5)]
        wrappers = [ToolExecutionWrapper(tool.run, f"TestWebSearchTool{i}") for i, tool in enumerate(tools)]
        tokens = [execution_registry.request_execution(f"TestWebSearchTool{i}", "test_agent", f"test_task_{i}")
                 for i in range(5)]

        # Act
        results = []
        for i, (wrapper, token) in enumerate(zip(wrappers, tokens, strict=False)):
            result = wrapper.execute_with_token(token, f"Query {i}")
            results.append((result, token))

        # Assert
        for i, (result, token) in enumerate(results):
            self.assertIn(f"Search results for 'Query {i}'", result)
            self.assertEqual(tools[i].call_count, 1)

            # Verify each execution
            is_valid = verify_observation_token(token.token_id)
            self.assertTrue(is_valid)

    def test_tool_execution_failure_handling(self):
        """Test that failed tool executions are properly recorded"""
        # Arrange
        def failing_tool(query: str) -> str:
            raise Exception("Tool execution failed")

        tool_wrapper = ToolExecutionWrapper(failing_tool, "FailingTool")
        token = execution_registry.request_execution("FailingTool", "test_agent", "test_task")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            tool_wrapper.execute_with_token(token, "test query")

        self.assertIn("Tool execution failed", str(context.exception))

        # Verify the execution (should still be tracked even though it failed)
        is_valid = verify_observation_token(token.token_id)
        self.assertFalse(is_valid)  # Failed executions are tracked but not verified as successful

    def test_token_uniqueness(self):
        """Test that tokens are unique"""
        # Arrange
        tokens = set()

        # Act
        for i in range(100):
            token = execution_registry.request_execution(f"Tool{i}", "agent1", f"task{i}")
            self.assertNotIn(token.token_id, tokens)
            tokens.add(token.token_id)

        # Assert
        self.assertEqual(len(tokens), 100)

if __name__ == '__main__':
    unittest.main(verbosity=2)
