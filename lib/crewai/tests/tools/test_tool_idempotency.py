"""Tests for tool execution idempotency on task retry."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_usage import ToolUsage


class TestToolsHandlerIdempotency:
    """Unit tests for ToolsHandler idempotency store."""

    def test_store_and_retrieve(self) -> None:
        handler = ToolsHandler()
        handler.set_idempotent_result("pay", {"amount": 10}, "payment-sent")
        assert handler.get_idempotent_result("pay", {"amount": 10}) == "payment-sent"

    def test_different_args_different_keys(self) -> None:
        handler = ToolsHandler()
        handler.set_idempotent_result("pay", {"amount": 10}, "result-10")
        handler.set_idempotent_result("pay", {"amount": 20}, "result-20")
        assert handler.get_idempotent_result("pay", {"amount": 10}) == "result-10"
        assert handler.get_idempotent_result("pay", {"amount": 20}) == "result-20"

    def test_missing_key_returns_none(self) -> None:
        handler = ToolsHandler()
        assert handler.get_idempotent_result("pay", {"amount": 10}) is None

    def test_reset_clears_store(self) -> None:
        handler = ToolsHandler()
        handler.set_idempotent_result("pay", {"amount": 10}, "sent")
        handler.reset_idempotency_store()
        assert handler.get_idempotent_result("pay", {"amount": 10}) is None

    def test_on_tool_use_stores_idempotent_result(self) -> None:
        handler = ToolsHandler()
        from crewai.tools.tool_calling import ToolCalling

        calling = ToolCalling(tool_name="pay", arguments={"amount": 10})
        handler.on_tool_use(calling, "payment-sent", should_cache=False)
        assert handler.get_idempotent_result("pay", {"amount": 10}) == "payment-sent"


class TestToolUsageIdempotency:
    """Integration tests: ToolUsage checks idempotency before executing."""

    @pytest.mark.asyncio
    async def test_ause_skips_execution_when_idempotent_result_exists(self) -> None:
        """If ToolsHandler already has an idempotent result, tool should not be invoked again."""
        handler = ToolsHandler()
        handler.set_idempotent_result("add", {"x": 1, "y": 2}, "3")

        tool = MagicMock(spec=CrewStructuredTool)
        tool.name = "add"
        tool.description = "Add numbers"
        tool.args_schema.model_json_schema.return_value = {
            "properties": {"x": {}, "y": {}}
        }

        from crewai.tools.tool_calling import ToolCalling
        calling = ToolCalling(tool_name="add", arguments={"x": 1, "y": 2})

        tool_usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        tool_usage.action = MagicMock()
        tool_usage.action.tool = "add"
        tool_usage.action.tool_input = {"x": 1, "y": 2}

        result = await tool_usage._ause(tool_string="", tool=tool, calling=calling)
        assert result == "3"
        tool.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_ause_executes_when_no_idempotent_result(self) -> None:
        """If no idempotent result, tool should execute normally."""
        handler = ToolsHandler()

        tool = MagicMock(spec=CrewStructuredTool)
        tool.name = "add"
        tool.description = "Add numbers"
        tool.args_schema.model_json_schema.return_value = {
            "properties": {"x": {}, "y": {}}
        }
        tool.ainvoke.return_value = 3

        from crewai.tools.tool_calling import ToolCalling
        calling = ToolCalling(tool_name="add", arguments={"x": 1, "y": 2})

        tool_usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        tool_usage.action = MagicMock()
        tool_usage.action.tool = "add"
        tool_usage.action.tool_input = {"x": 1, "y": 2}

        await tool_usage._ause(tool_string="", tool=tool, calling=calling)
        tool.ainvoke.assert_called_once()

    def test_use_skips_execution_when_idempotent_result_exists(self) -> None:
        """Sync path also skips execution when idempotent result exists."""
        handler = ToolsHandler()
        handler.set_idempotent_result("add", {"x": 1, "y": 2}, "3")

        tool = MagicMock(spec=CrewStructuredTool)
        tool.name = "add"
        tool.description = "Add numbers"
        tool.args_schema.model_json_schema.return_value = {
            "properties": {"x": {}, "y": {}}
        }

        from crewai.tools.tool_calling import ToolCalling
        calling = ToolCalling(tool_name="add", arguments={"x": 1, "y": 2})

        tool_usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        tool_usage.action = MagicMock()
        tool_usage.action.tool = "add"
        tool_usage.action.tool_input = {"x": 1, "y": 2}

        result = tool_usage._use(tool_string="", tool=tool, calling=calling)
        assert result == "3"
        tool.invoke.assert_not_called()
