"""Tests for tool execution idempotency on task retry."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities.idempotency_backend import (
    IdempotencyBackend,
    IdempotencyInProgressError,
    MemoryIdempotencyBackend,
)


class TestMemoryIdempotencyBackend:
    """Unit tests for MemoryIdempotencyBackend."""

    def test_set_and_get(self) -> None:
        backend = MemoryIdempotencyBackend()
        backend.set("key1", "value1")
        assert backend.get("key1") == "value1"

    def test_get_missing_returns_none(self) -> None:
        backend = MemoryIdempotencyBackend()
        assert backend.get("missing") is None

    def test_claim_succeeds_when_key_absent(self) -> None:
        backend = MemoryIdempotencyBackend()
        claimed, result = backend.claim("key1")
        assert claimed is True
        assert result is None

    def test_claim_returns_result_when_key_exists(self) -> None:
        backend = MemoryIdempotencyBackend()
        backend.set("key1", "value1")
        claimed, result = backend.claim("key1")
        assert claimed is False
        assert result == "value1"

    def test_claim_blocks_second_caller_on_in_progress(self) -> None:
        backend = MemoryIdempotencyBackend()
        # First caller claims
        claimed1, result1 = backend.claim("key1")
        assert claimed1 is True
        assert result1 is None
        # Second caller sees in-progress
        claimed2, result2 = backend.claim("key1")
        assert claimed2 is False
        assert result2 is None  # in-progress, not yet completed

    def test_claim_then_set_makes_result_available(self) -> None:
        backend = MemoryIdempotencyBackend()
        backend.claim("key1")  # claim
        backend.set("key1", "final_value")  # publish result
        claimed, result = backend.claim("key1")
        assert claimed is False
        assert result == "final_value"

    def test_get_returns_none_for_in_progress(self) -> None:
        backend = MemoryIdempotencyBackend()
        backend.claim("key1")
        assert backend.get("key1") is None  # in-progress treated as not available

    def test_clear_removes_all(self) -> None:
        backend = MemoryIdempotencyBackend()
        backend.set("key1", "value1")
        backend.set("key2", "value2")
        backend.clear()
        assert backend.get("key1") is None
        assert backend.get("key2") is None


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

    def test_on_tool_use_with_empty_dict_arguments_stores_result(self) -> None:
        """No-argument tool calls (arguments={}) must be stored for idempotency."""
        handler = ToolsHandler()
        from crewai.tools.tool_calling import ToolCalling

        calling = ToolCalling(tool_name="noop", arguments={})
        handler.on_tool_use(calling, "done", should_cache=False)
        assert handler.get_idempotent_result("noop", {}) == "done"

    def test_on_tool_use_with_none_arguments_skips_store(self) -> None:
        """None arguments should skip idempotency storage (no key to build)."""
        handler = ToolsHandler()
        from crewai.tools.tool_calling import ToolCalling

        calling = ToolCalling(tool_name="noop", arguments=None)
        handler.on_tool_use(calling, "done", should_cache=False)
        # Should not have stored anything
        assert handler.get_idempotent_result("noop", {}) is None

    def test_claim_idempotent_result_returns_stored_result(self) -> None:
        handler = ToolsHandler()
        handler.set_idempotent_result("add", {"x": 1}, "42")
        result = handler.claim_idempotent_result("add", {"x": 1})
        assert result == "42"

    def test_claim_idempotent_result_returns_none_when_not_stored(self) -> None:
        handler = ToolsHandler()
        result = handler.claim_idempotent_result("add", {"x": 1})
        assert result is None  # claim succeeded, caller must execute

    def test_claim_idempotent_blocks_in_progress_execution(self) -> None:
        handler = ToolsHandler()
        # First claim succeeds (caller must execute)
        result1 = handler.claim_idempotent_result("add", {"x": 1})
        assert result1 is None
        with pytest.raises(IdempotencyInProgressError, match="already in progress"):
            handler.claim_idempotent_result("add", {"x": 1})

    def test_tool_name_sanitization_consistency(self) -> None:
        """Write and read paths must use the same sanitised tool name."""
        handler = ToolsHandler()
        from crewai.tools.tool_calling import ToolCalling

        # Tool name with spaces and mixed case — should be normalized
        calling = ToolCalling(
            tool_name="My Tool !", arguments={"x": 1}
        )
        handler.on_tool_use(calling, "result", should_cache=False)
        # Read back with the same raw name — sanitization must match
        assert (
            handler.get_idempotent_result("My Tool !", {"x": 1}) == "result"
        )

    def test_custom_backend_injection(self) -> None:
        """Users can inject a custom IdempotencyBackend."""
        handler = ToolsHandler()

        class CountingBackend(IdempotencyBackend):
            def __init__(self) -> None:
                self.gets = 0
                self.sets = 0
                self._store: dict[str, str] = {}

            def get(self, key: str) -> str | None:
                self.gets += 1
                return self._store.get(key)

            def set(self, key: str, value: str) -> None:
                self.sets += 1
                self._store[key] = value

            def claim(self, key: str) -> tuple[bool, str | None]:
                existing = self._store.get(key)
                if existing is not None:
                    return (False, existing)
                self._store[key] = "claimed"
                return (True, None)

            def clear(self) -> None:
                self._store.clear()

            def release(self, key: str) -> None:
                self._store.pop(key, None)

        backend = CountingBackend()
        handler.set_idempotency_backend(backend)
        handler.set_idempotent_result("tool", {}, "result")
        assert backend.sets == 1
        assert handler.get_idempotent_result("tool", {}) == "result"
        assert backend.gets == 1


class TestToolUsageIdempotency:
    """Integration tests: ToolUsage checks idempotency before executing."""

    @pytest.mark.asyncio
    async def test_ause_skips_execution_when_idempotent_result_exists(self) -> None:
        """If ToolsHandler already has an idempotent result, tool should not be invoked again."""
        handler = ToolsHandler()
        handler.set_idempotent_result("add", {"x": 1, "y": 2}, "3")

        tool = MagicMock()
        tool.name = "add"
        tool.description = "Add numbers"
        tool.max_usage_count = None
        tool.current_usage_count = 0
        tool.format_output_for_agent.side_effect = str
        tool.args_schema = MagicMock()
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

        tool = MagicMock()
        tool.name = "add"
        tool.description = "Add numbers"
        tool.max_usage_count = None
        tool.current_usage_count = 0
        tool.format_output_for_agent.side_effect = str
        tool.args_schema = MagicMock()
        tool.args_schema.model_json_schema.return_value = {
            "properties": {"x": {}, "y": {}}
        }
        tool.ainvoke = AsyncMock(return_value=3)

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

        tool = MagicMock()
        tool.name = "add"
        tool.description = "Add numbers"
        tool.max_usage_count = None
        tool.current_usage_count = 0
        tool.format_output_for_agent.side_effect = str
        tool.args_schema = MagicMock()
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

    @pytest.mark.asyncio
    async def test_ause_with_no_args_tool(self) -> None:
        """No-argument tool calls (arguments={}) must go through idempotency check."""
        handler = ToolsHandler()
        # Pre-store a result for a no-arg tool call
        handler.set_idempotent_result("noop", {}, "done")

        tool = MagicMock()
        tool.name = "noop"
        tool.description = "No-op tool"
        tool.max_usage_count = None
        tool.current_usage_count = 0
        tool.format_output_for_agent.side_effect = str
        tool.args_schema = MagicMock()
        tool.args_schema.model_json_schema.return_value = {"properties": {}}

        from crewai.tools.tool_calling import ToolCalling
        calling = ToolCalling(tool_name="noop", arguments={})

        tool_usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        tool_usage.action = MagicMock()
        tool_usage.action.tool = "noop"
        tool_usage.action.tool_input = {}

        result = await tool_usage._ause(tool_string="", tool=tool, calling=calling)
        assert result == "done"
        tool.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_ause_with_none_arguments_skips_idempotency(self) -> None:
        """None arguments should skip idempotency check entirely."""
        handler = ToolsHandler()
        handler.set_idempotent_result("noop", {}, "stale")

        tool = MagicMock()
        tool.name = "noop"
        tool.description = "No-op tool"
        tool.max_usage_count = None
        tool.current_usage_count = 0
        tool.format_output_for_agent.side_effect = str
        tool.args_schema = MagicMock()
        tool.args_schema.model_json_schema.return_value = {"properties": {}}
        tool.ainvoke = AsyncMock(return_value="fresh")

        from crewai.tools.tool_calling import ToolCalling
        calling = ToolCalling(tool_name="noop", arguments=None)

        tool_usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        tool_usage.action = MagicMock()
        tool_usage.action.tool = "noop"
        tool_usage.action.tool_input = None

        result = await tool_usage._ause(tool_string="", tool=tool, calling=calling)
        assert result == "fresh"  # executed fresh, not from idempotency
        tool.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_tool_calls_do_not_repeat_an_in_progress_side_effect():
    import asyncio

    from crewai.tools.tool_calling import ToolCalling

    handler = ToolsHandler()
    started = asyncio.Event()
    release = asyncio.Event()
    tool = MagicMock()
    tool.name = "pay"
    tool.description = "Submit a payment"
    tool.max_usage_count = None
    tool.current_usage_count = 0
    tool.format_output_for_agent.side_effect = str
    tool.args_schema.model_json_schema.return_value = {"properties": {"amount": {}}}

    async def submit_payment(**kwargs):
        started.set()
        await release.wait()
        return "payment-sent"

    tool.ainvoke = AsyncMock(side_effect=submit_payment)
    calling = ToolCalling(tool_name="pay", arguments={"amount": 10})

    def make_usage():
        usage = ToolUsage(
            tools_handler=handler,
            tools=[tool],
            task=None,
            function_calling_llm=MagicMock(),
        )
        usage.action = MagicMock()
        usage.action.tool = "pay"
        usage.action.tool_input = {"amount": 10}
        return usage

    owner = asyncio.create_task(
        make_usage()._ause(tool_string="", tool=tool, calling=calling)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        # A duplicate call must stop before invoking the side effect, for both
        # entry points, while the original execution retains its claim.
        duplicate = await asyncio.wait_for(
            make_usage()._ause(tool_string="", tool=tool, calling=calling), timeout=0.2
        )
        assert "already in progress" in duplicate
        assert "already in progress" in make_usage()._use(
            tool_string="", tool=tool, calling=calling
        )
        tool.invoke.assert_not_called()
        assert tool.ainvoke.call_count == 1
    finally:
        release.set()
        await owner

    # A task retry starts a new action sequence while reusing the backend.
    handler.last_used_tool = None
    assert await make_usage()._ause(tool_string="", tool=tool, calling=calling) == "payment-sent"
    assert tool.ainvoke.call_count == 1


def _formal_tool_usage(func, handler):
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.tools.tool_calling import ToolCalling

    tool = CrewStructuredTool.from_function(func, name="pay", description="Submit payment")
    usage = ToolUsage(tools_handler=handler, tools=[tool], task=None, function_calling_llm=MagicMock())
    usage.action = MagicMock(tool="pay", tool_input={"amount": 10})
    calling = ToolCalling(tool_name="pay", arguments={"amount": 10})
    return usage, tool, calling


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_claim_lifecycle_cache_and_usage_limit(asynchronous):
    from crewai.agents.cache.cache_handler import CacheHandler

    def pay(amount: int) -> str:
        return "payment-sent"

    handler = ToolsHandler(cache=CacheHandler())
    handler.cache.add(tool="pay", input='{"amount": 10}', output="cached-payment")
    usage, tool, calling = _formal_tool_usage(pay, handler)
    async def execute():
        return await usage.ause(calling, "") if asynchronous else usage.use(calling, "")
    assert await execute() == "cached-payment"
    assert handler.claim_idempotent_result("pay", {"amount": 10}) == "cached-payment"

    handler = ToolsHandler()
    usage, tool, calling = _formal_tool_usage(pay, handler)
    tool.max_usage_count = 0
    assert "usage limit" in await execute()
    tool.max_usage_count = None
    assert await execute() == "payment-sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
async def test_unknown_tool_outcome_is_reported_without_duplicate_execution(asynchronous, error_type):
    calls = []
    def pay(amount: int) -> str:
        calls.append(amount)
        raise error_type("response lost after side effect")
    handler = ToolsHandler()
    usage, tool, calling = _formal_tool_usage(pay, handler)
    result = await usage.ause(calling, "") if asynchronous else usage.use(calling, "")
    assert "outcome" in result
    assert calls == [10]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_declared_retryable_failure_does_not_become_success(asynchronous):
    from crewai.tools.tool_failure import ToolFailure
    calls = []
    def pay(amount: int):
        calls.append(amount)
        if len(calls) == 1:
            return ToolFailure(message="unavailable before payment", retryable=True)
        return "payment-sent"
    handler = ToolsHandler()
    usage, tool, calling = _formal_tool_usage(pay, handler)
    async def execute():
        return await usage.ause(calling, "") if asynchronous else usage.use(calling, "")
    assert await execute() == "unavailable before payment"
    handler.last_used_tool = None
    assert await execute() == "payment-sent"
    assert calls == [10, 10]


def test_native_entry_reuses_successful_tool_outcome():
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    calls = []
    def pay(amount: int) -> str:
        calls.append(amount)
        return "payment-sent"
    handler = ToolsHandler()
    _, tool, _ = _formal_tool_usage(pay, handler)
    executor = CrewAgentExecutor(tools=[tool], original_tools=[], tools_handler=handler)
    for call_id in ("first", "retry"):
        result = executor._execute_single_native_tool_call(
            call_id=call_id, func_name="pay", func_args={"amount": 10},
            available_functions={"pay": pay}, original_tool=tool,
        )
        assert result["result"] == "payment-sent"
    assert calls == [10]


def test_task_preparation_retains_retries_but_isolates_distinct_tasks():
    from crewai import Agent, Task
    from crewai.llm import LLM
    agent = Agent(role="r", goal="g", backstory="b", llm=LLM(model="gpt-4o-mini", api_key="test"))
    task = Task(description="pay", expected_output="receipt")
    next_task = Task(description="pay again", expected_output="another receipt")
    agent._prepare_task_execution(task, None)
    handler = agent.tools_handler
    handler.set_idempotent_result("pay", {"amount": 10}, "first-receipt")
    agent._prepare_task_execution(task, None)
    assert handler.get_idempotent_result("pay", {"amount": 10}) == "first-receipt"
    agent._prepare_task_execution(next_task, None)
    assert handler.get_idempotent_result("pay", {"amount": 10}) is None


def test_agent_task_retry_does_not_repeat_a_successful_side_effect():
    from crewai import Agent, Task
    from crewai.llm import LLM
    from crewai.tools.base_tool import BaseTool

    calls = []
    class PaymentTool(BaseTool):
        name: str = "pay"
        description: str = "Submit a payment."
        def _run(self, amount: int) -> str:
            calls.append(amount)
            return "payment-sent"

    class RetryLLM(LLM):
        def __new__(cls):
            return object.__new__(cls)
        def __init__(self):
            super().__init__(model="gpt-4o")
            self.steps = iter([
                'Thought: pay\nAction: pay\nAction Input: {"amount": 10}',
                RuntimeError("interrupted after payment"),
                'Thought: retry task\nAction: pay\nAction Input: {"amount": 10}',
                "Thought: done\nFinal Answer: receipt",
            ])
        def call(self, messages, **kwargs):
            step = next(self.steps)
            if isinstance(step, Exception):
                raise step
            return step
        def supports_function_calling(self):
            return False

    agent = Agent(role="payer", goal="pay", backstory="b", llm=RetryLLM(),
                  tools=[PaymentTool()], max_retry_limit=1)
    task = Task(description="pay ten", expected_output="receipt", agent=agent)
    assert agent.execute_task(task) == "receipt"
    assert calls == [10]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_terminal_tool_failure_keeps_its_original_result(asynchronous):
    from crewai.tools.tool_failure import ToolFailure

    calls = []
    failure = ToolFailure(message="payment refused", retryable=False)
    def pay(amount: int):
        calls.append(amount)
        return failure
    handler = ToolsHandler()
    usage, tool, calling = _formal_tool_usage(pay, handler)
    async def execute():
        handler.last_used_tool = None
        return await usage.ause(calling, "") if asynchronous else usage.use(calling, "")
    assert await execute() == "payment refused"
    assert await execute() == "payment refused"
    assert usage.last_failure is failure
    assert calls == [10]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_tool_arguments_leave_no_execution_in_progress(asynchronous):
    calls = []
    def pay(amount: int) -> str:
        calls.append(amount)
        return "payment-sent"
    handler = ToolsHandler()
    usage, tool, calling = _formal_tool_usage(pay, handler)
    calling.arguments = {"amount": "not-an-integer"}
    result = await usage.ause(calling, "") if asynchronous else usage.use(calling, "")
    assert "Arguments validation failed" in result
    assert "already in progress" not in result
    assert calls == []
    assert handler.claim_idempotent_result("pay", calling.arguments) is None
