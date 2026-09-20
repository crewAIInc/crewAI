"""Offline regressions for SnowflakeSearchTool error handling.

The optional ``snowflake`` package is required to construct the tool, so a
minimal stand-in is injected only when the real dependency is absent. CI runs
with the real package installed and never uses the stand-in.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

try:
    from snowflake.connector.errors import DatabaseError
except ImportError:
    _snowflake = types.ModuleType("snowflake")
    _connector = types.ModuleType("snowflake.connector")
    _errors = types.ModuleType("snowflake.connector.errors")

    class DatabaseError(Exception):
        """Stand-in for snowflake.connector.errors.DatabaseError."""

    class OperationalError(DatabaseError):
        """Stand-in for snowflake.connector.errors.OperationalError."""

    _errors.DatabaseError = DatabaseError  # type: ignore[attr-defined]
    _errors.OperationalError = OperationalError  # type: ignore[attr-defined]
    _connector.connect = MagicMock()  # type: ignore[attr-defined]
    _connector.errors = _errors  # type: ignore[attr-defined]
    _snowflake.connector = _connector  # type: ignore[attr-defined]
    sys.modules.setdefault("snowflake", _snowflake)
    sys.modules.setdefault("snowflake.connector", _connector)
    sys.modules.setdefault("snowflake.connector.errors", _errors)

    from snowflake.connector.errors import DatabaseError

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool


@pytest.fixture
def mock_snowflake_connection():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",)]
    mock_cursor.fetchall.return_value = [(1,)]
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def snowflake_tool():
    config = SnowflakeConfig(
        account="test_account",
        user="test_user",
        password="test_password",
    )
    with patch("snowflake.connector.connect"):
        tool = SnowflakeSearchTool(config=config)
    tool.enable_caching = False
    return tool


@pytest.mark.asyncio
async def test_database_error_is_retried_without_being_masked(
    snowflake_tool, mock_snowflake_connection
):
    """A failing query surfaces the Snowflake error, not a NameError."""
    snowflake_tool.max_retries = 2
    snowflake_tool.retry_delay = 0
    cursor = mock_snowflake_connection.cursor.return_value
    cursor.execute.side_effect = DatabaseError("query failed")

    with patch.object(
        snowflake_tool, "_create_connection", return_value=mock_snowflake_connection
    ):
        with pytest.raises(DatabaseError, match="query failed"):
            await snowflake_tool._execute_query("SELECT 1")

    assert cursor.execute.call_count == 2
    assert cursor.close.call_count == 2


@pytest.mark.asyncio
async def test_cursor_creation_error_is_not_masked(
    snowflake_tool, mock_snowflake_connection
):
    """A cursor failure keeps its Snowflake error instead of UnboundLocalError."""
    snowflake_tool.max_retries = 1
    mock_snowflake_connection.cursor.side_effect = DatabaseError("cursor unavailable")

    with patch.object(
        snowflake_tool, "_create_connection", return_value=mock_snowflake_connection
    ):
        with pytest.raises(DatabaseError, match="cursor unavailable"):
            await snowflake_tool._execute_query("SELECT 1")

    assert mock_snowflake_connection in snowflake_tool._connection_pool
