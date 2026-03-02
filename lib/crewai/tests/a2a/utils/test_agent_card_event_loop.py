"""Tests for fetch_agent_card sync wrapper when a running event loop exists.

Covers the same class of issue as #4671 — ``fetch_agent_card()`` must work
even when called from within an already-running event loop (e.g. Jupyter).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from a2a.types import AgentCapabilities, AgentCard

from crewai.a2a.utils.agent_card import fetch_agent_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_agent_card() -> AgentCard:
    """Return a minimal ``AgentCard`` for mocking."""
    return AgentCard(
        name="Test Agent",
        description="A test agent",
        url="http://localhost:9999",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchAgentCardEventLoop:
    """Verify fetch_agent_card works with and without a running loop."""

    @patch("crewai.a2a.utils.agent_card.afetch_agent_card")
    def test_works_without_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Normal case: no running event loop, should succeed directly."""
        expected = _mock_agent_card()
        mock_async_fn.return_value = expected

        result = fetch_agent_card(
            endpoint="http://localhost:9999",
            use_cache=False,
        )

        assert result.name == "Test Agent"
        mock_async_fn.assert_called_once()

    @patch("crewai.a2a.utils.agent_card.afetch_agent_card")
    def test_works_inside_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Regression test: must not raise when a loop is already running."""
        expected = _mock_agent_card()
        mock_async_fn.return_value = expected

        result_holder: list[Any] = []
        error_holder: list[Exception] = []

        async def _call_sync_from_async() -> None:
            try:
                res = fetch_agent_card(
                    endpoint="http://localhost:9999",
                    use_cache=False,
                )
                result_holder.append(res)
            except Exception as exc:
                error_holder.append(exc)

        asyncio.run(_call_sync_from_async())

        assert not error_holder, f"Unexpected error: {error_holder[0]}"
        assert len(result_holder) == 1
        assert result_holder[0].name == "Test Agent"

    @patch("crewai.a2a.utils.agent_card.afetch_agent_card")
    def test_propagates_errors_inside_running_event_loop(
        self, mock_async_fn: AsyncMock
    ) -> None:
        """Errors should propagate even when called from a running loop."""
        mock_async_fn.side_effect = ConnectionError("cannot reach agent")

        error_holder: list[Exception] = []

        async def _call_sync_from_async() -> None:
            try:
                fetch_agent_card(
                    endpoint="http://localhost:9999",
                    use_cache=False,
                )
            except Exception as exc:
                error_holder.append(exc)

        asyncio.run(_call_sync_from_async())

        assert len(error_holder) == 1
        assert isinstance(error_holder[0], ConnectionError)
        assert "cannot reach agent" in str(error_holder[0])
