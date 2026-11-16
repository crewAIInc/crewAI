import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event

import pytest

from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool, ToolUsageLimitExceededError


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Return the supplied value."

    def _run(self, value: int) -> int:
        return value


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("mixed_async", [False, True])
def test_concurrent_invocations_respect_the_usage_budget(
    converted: bool, mixed_async: bool
) -> None:
    """Only five callers may execute, even when they contend for the last slot."""

    def echo(value: int) -> int:
        """Return the supplied value."""
        return value

    original_interval = sys.getswitchinterval()
    try:
        # Exercise real thread interleavings without replacing the tool's
        # counter, limit check, or invocation methods.
        sys.setswitchinterval(0.000001)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for _ in range(200):
                original = EchoTool(max_usage_count=5)
                tool = (
                    original.to_structured_tool()
                    if converted
                    else CrewStructuredTool.from_function(echo, max_usage_count=5)
                )
                ready = Barrier(16)

                def invoke(value: int) -> int | None:
                    ready.wait(timeout=5)
                    try:
                        if mixed_async and value % 2:
                            return asyncio.run(tool.ainvoke({"value": value}))
                        return tool.invoke({"value": value})
                    except ToolUsageLimitExceededError:
                        return None

                results = list(pool.map(invoke, range(16)))
                assert sum(result is not None for result in results) == 5
                assert tool.current_usage_count == 5
                if converted:
                    assert original.current_usage_count == 5
    finally:
        sys.setswitchinterval(original_interval)


@pytest.mark.parametrize("shared_tool", [False, True])
def test_usage_claim_does_not_serialize_tool_execution(shared_tool: bool) -> None:
    executing = Barrier(2)

    def echo(value: int) -> int:
        """Wait for another tool call before returning."""
        executing.wait(timeout=5)
        return value

    tool = CrewStructuredTool.from_function(echo, max_usage_count=2)
    other = (
        tool
        if shared_tool
        else CrewStructuredTool.from_function(echo, max_usage_count=1)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tool.invoke, {"value": 1})
        second = pool.submit(other.invoke, {"value": 2})
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert results == [1, 2]
    assert tool.current_usage_count == (2 if shared_tool else 1)
    assert other.current_usage_count == (2 if shared_tool else 1)


@pytest.mark.asyncio
async def test_cancelled_tool_cleans_up_and_keeps_its_spent_slot() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def wait() -> str:
        """Wait until cancelled."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()
        return "done"

    tool = CrewStructuredTool.from_function(wait, max_usage_count=1)
    task = asyncio.create_task(tool.ainvoke({}))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cleaned_up.is_set()
    assert tool.current_usage_count == 1
    with pytest.raises(ToolUsageLimitExceededError):
        await tool.ainvoke({})


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("copy_method", ["deepcopy", "model_copy"])
def test_tool_remains_deep_copyable_with_an_independent_budget(
    converted: bool, copy_method: str
) -> None:
    def echo(value: int) -> int:
        """Return the supplied value."""
        return value

    base = EchoTool(max_usage_count=2)
    original = (
        base.to_structured_tool()
        if converted
        else CrewStructuredTool.from_function(echo, max_usage_count=2)
    )
    assert original.invoke({"value": 1}) == 1
    copied = (
        deepcopy(original)
        if copy_method == "deepcopy"
        else original.model_copy(deep=True)
    )
    # Converted callables bind to the copied BaseTool instead of the original.
    if not converted:
        assert copied == original
    assert copied.current_usage_count == 1

    assert copied.invoke({"value": 2}) == 2
    assert original.current_usage_count == 1
    if converted:
        assert base.current_usage_count == 1
    assert original.invoke({"value": 3}) == 3
    assert original.current_usage_count == copied.current_usage_count == 2
    with pytest.raises(ToolUsageLimitExceededError):
        copied.invoke({"value": 4})
    with pytest.raises(ToolUsageLimitExceededError):
        original.invoke({"value": 5})


@pytest.mark.parametrize("copy_method", [None, "deepcopy", "model_copy"])
def test_usage_claims_on_separate_instances_do_not_contend(
    copy_method: str | None,
) -> None:
    checking_budget = Event()
    release_budget = Event()

    class WaitingTool(CrewStructuredTool):
        def has_reached_max_usage_count(self) -> bool:
            if self is waiting:
                checking_budget.set()
                assert release_budget.wait(timeout=10)
            return super().has_reached_max_usage_count()

    def echo(value: int) -> int:
        """Return the supplied value."""
        return value

    waiting = WaitingTool.from_function(echo, max_usage_count=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(waiting.invoke, {"value": 1})
        try:
            assert checking_budget.wait(timeout=5)
            if copy_method == "deepcopy":
                independent = deepcopy(waiting)
            elif copy_method == "model_copy":
                independent = waiting.model_copy(deep=True)
            else:
                independent = CrewStructuredTool.from_function(echo, max_usage_count=1)
            second = pool.submit(independent.invoke, {"value": 2})
            assert second.result(timeout=5) == 2
        finally:
            release_budget.set()
        assert first.result(timeout=5) == 1

    assert waiting.current_usage_count == independent.current_usage_count == 1


@pytest.mark.parametrize("copy_method", ["deepcopy", "model_copy"])
def test_copied_converted_tool_executes_with_its_updated_usage_count(
    copy_method: str,
) -> None:
    class CountingTool(BaseTool):
        name: str = "counting"
        description: str = "Return this tool's current usage count."

        def _run(self) -> int:
            return self.current_usage_count

    base = CountingTool(max_usage_count=2)
    original = base.to_structured_tool()
    assert original.invoke({}) == 1

    copied = (
        deepcopy(original)
        if copy_method == "deepcopy"
        else original.model_copy(deep=True)
    )
    assert copied.invoke({}) == 2
    assert copied.current_usage_count == 2
    assert original.current_usage_count == base.current_usage_count == 1
