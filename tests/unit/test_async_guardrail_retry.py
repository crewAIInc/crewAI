"""Tests for async guardrail retry fix (issue #7252)."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from crewai.agent.core import Agent


class TestAsyncGuardrailRetry:
    """Guardrail retries must work correctly in async context."""

    def test_execute_and_build_output_handles_coroutine_result(self):
        """When executor.invoke returns a coroutine, it should be awaited properly."""
        agent = Agent(
            role="Test agent",
            goal="Test",
            backstory="Test",
            llm="openai/gpt-4o-mini",
        )

        # Mock the executor to return a coroutine (simulating async context)
        async def fake_invoke(inputs):
            return {"output": "test result"}

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = fake_invoke({})
        mock_executor.state = MagicMock()
        mock_executor.state.todos = MagicMock()
        mock_executor.state.todos.items = []
        mock_executor.state.messages = []
        mock_executor.state.plan = None
        mock_executor.state.replan_count = 0
        mock_executor.state.last_replan_reason = None

        # The method should handle the coroutine and return proper output
        with patch("crewai.agent.core.tool_failure_collector") as mock_collector:
            mock_collector.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_collector.return_value.__exit__ = MagicMock(return_value=False)
            result = agent._execute_and_build_output(mock_executor, {})

        # Should return a LiteAgentOutput, not raise AttributeError
        assert hasattr(result, "raw")

    def test_execute_and_build_output_handles_dict_result(self):
        """When executor.invoke returns a dict (normal sync), it should work."""
        agent = Agent(
            role="Test agent",
            goal="Test",
            backstory="Test",
            llm="openai/gpt-4o-mini",
        )

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "test result"}
        mock_executor.state = MagicMock()
        mock_executor.state.todos = MagicMock()
        mock_executor.state.todos.items = []
        mock_executor.state.messages = []
        mock_executor.state.plan = None
        mock_executor.state.replan_count = 0
        mock_executor.state.last_replan_reason = None

        with patch("crewai.agent.core.tool_failure_collector") as mock_collector:
            mock_collector.return_value.__enter__ = MagicMock(
                return_value=MagicMock()
            )
            mock_collector.return_value.__exit__ = MagicMock(return_value=False)
            result = agent._execute_and_build_output(mock_executor, {})

        assert hasattr(result, "raw")
