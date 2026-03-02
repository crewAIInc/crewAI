"""Tests for A2A delegation sync wrappers when a running event loop exists.

Covers the fix for https://github.com/crewAIInc/crewAI/issues/4671 where
``execute_a2a_delegation()`` raised ``RuntimeError`` when called from an
environment that already has a running event loop (e.g. Jupyter notebooks).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.utils.delegation import execute_a2a_delegation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_task_state_result(**overrides: Any) -> TaskStateResult:
    """Return a minimal ``TaskStateResult`` dict for mocking."""
    base: TaskStateResult = {
        "status": "completed",
        "result": "mocked result",
        "error": None,
        "history": [],
        "agent_card": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


_DELEGATION_KWARGS: dict[str, Any] = dict(
    endpoint="http://localhost:9999/.well-known/agent-card.json",
    auth=None,
    timeout=30,
    task_description="test task",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteA2ADelegationEventLoop:
    """Verify execute_a2a_delegation works with and without a running loop."""

    @patch("crewai.a2a.utils.delegation.aexecute_a2a_delegation")
    def test_works_without_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Normal case: no running event loop, should succeed directly."""
        expected = _minimal_task_state_result()
        mock_async_fn.return_value = expected

        result = execute_a2a_delegation(**_DELEGATION_KWARGS)

        assert result["status"] == "completed"
        assert result["result"] == "mocked result"
        mock_async_fn.assert_called_once()

    @patch("crewai.a2a.utils.delegation.aexecute_a2a_delegation")
    def test_works_inside_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Regression test for #4671: must not raise when a loop is running.

        Simulates the Jupyter notebook environment by calling the sync wrapper
        from within an already-running event loop.
        """
        expected = _minimal_task_state_result()
        mock_async_fn.return_value = expected

        result_holder: list[Any] = []
        error_holder: list[Exception] = []

        async def _call_sync_from_async() -> None:
            """Call the sync function from within a running event loop."""
            try:
                # This must NOT raise RuntimeError anymore
                res = execute_a2a_delegation(**_DELEGATION_KWARGS)
                result_holder.append(res)
            except Exception as exc:
                error_holder.append(exc)

        # Run inside an event loop, simulating a Jupyter notebook
        asyncio.run(_call_sync_from_async())

        assert not error_holder, f"Unexpected error: {error_holder[0]}"
        assert len(result_holder) == 1
        assert result_holder[0]["status"] == "completed"
        assert result_holder[0]["result"] == "mocked result"

    @patch("crewai.a2a.utils.delegation.aexecute_a2a_delegation")
    def test_propagates_errors_from_async_fn(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Errors from the underlying async function should propagate."""
        mock_async_fn.side_effect = ConnectionError("remote agent down")

        with pytest.raises(ConnectionError, match="remote agent down"):
            execute_a2a_delegation(**_DELEGATION_KWARGS)

    @patch("crewai.a2a.utils.delegation.aexecute_a2a_delegation")
    def test_propagates_errors_inside_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Errors should propagate even when called from a running loop."""
        mock_async_fn.side_effect = ConnectionError("remote agent down")

        error_holder: list[Exception] = []

        async def _call_sync_from_async() -> None:
            try:
                execute_a2a_delegation(**_DELEGATION_KWARGS)
            except Exception as exc:
                error_holder.append(exc)

        asyncio.run(_call_sync_from_async())

        assert len(error_holder) == 1
        assert isinstance(error_holder[0], ConnectionError)
        assert "remote agent down" in str(error_holder[0])
