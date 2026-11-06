import asyncio
from unittest.mock import MagicMock, patch

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool
import pytest


# Unit Test Fixtures
@pytest.fixture
def mock_snowflake_connection():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.description = [("col1",), ("col2",)]
    mock_cursor.fetchall.return_value = [(1, "value1"), (2, "value2")]
    mock_cursor.execute.return_value = None
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


@pytest.fixture
def mock_config():
    return SnowflakeConfig(
        account="test_account",
        user="test_user",
        password="test_password",
        warehouse="test_warehouse",
        database="test_db",
        snowflake_schema="test_schema",
    )


@pytest.fixture
def snowflake_tool(mock_config):
    with patch("snowflake.connector.connect"):
        tool = SnowflakeSearchTool(config=mock_config)
        yield tool


# Unit Tests
@pytest.mark.asyncio
async def test_successful_query_execution(snowflake_tool, mock_snowflake_connection):
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        results = await snowflake_tool._run(
            query="SELECT * FROM test_table", timeout=300
        )

        assert len(results) == 2
        assert results[0]["col1"] == 1
        assert results[0]["col2"] == "value1"
        mock_snowflake_connection.cursor.assert_called_once()


@pytest.mark.asyncio
async def test_connection_pooling(snowflake_tool, mock_snowflake_connection):
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        await asyncio.gather(
            snowflake_tool._run("SELECT 1"),
            snowflake_tool._run("SELECT 2"),
            snowflake_tool._run("SELECT 3"),
        )

        # Should reuse connections from pool
        assert mock_create_conn.call_count <= snowflake_tool.pool_size


@pytest.mark.asyncio
async def test_cleanup_on_deletion(snowflake_tool, mock_snowflake_connection):
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        await snowflake_tool._get_connection()

        async with snowflake_tool._pool_lock:
            snowflake_tool._connection_pool.append(mock_snowflake_connection)

        # Trigger cleanup
        snowflake_tool.__del__()

        mock_snowflake_connection.close.assert_called_once()


def test_config_validation():
    # Test missing required fields
    with pytest.raises(ValueError):
        SnowflakeConfig()

    with pytest.raises(ValueError):
        SnowflakeConfig(
            account="invalid//account", user="test_user", password="test_pass"
        )

    with pytest.raises(ValueError):
        SnowflakeConfig(account="test_account", user="test_user")


def test_arun_does_not_raise_not_implemented() -> None:
    """``await tool.arun(...)`` must delegate to ``_run`` instead of raising."""
    from unittest.mock import AsyncMock

    with (
        patch(
            "crewai_tools.tools.snowflake_search_tool.snowflake_search_tool.SNOWFLAKE_AVAILABLE",
            True,
        ),
        patch.object(SnowflakeSearchTool, "_initialize_snowflake", lambda self: None),
    ):
        tool = SnowflakeSearchTool(
            config=SnowflakeConfig(account="acct", user="u", password="p"),
        )
    tool._execute_query = AsyncMock(return_value=[{"VALUE": 1}])
    # _claim_usage touches the usage counter / lock; stub it so the test
    # focuses purely on the _arun -> _run delegation.
    with patch.object(tool, "_claim_usage", return_value=None):
        result = asyncio.run(tool.arun(query="SELECT 1 AS value"))
    assert result == [{"VALUE": 1}]


def test_arun_delegates_to_run() -> None:
    """``_arun`` returns the same rows as ``_run`` for the same query."""
    from unittest.mock import AsyncMock

    with (
        patch(
            "crewai_tools.tools.snowflake_search_tool.snowflake_search_tool.SNOWFLAKE_AVAILABLE",
            True,
        ),
        patch.object(SnowflakeSearchTool, "_initialize_snowflake", lambda self: None),
    ):
        tool = SnowflakeSearchTool(
            config=SnowflakeConfig(account="acct", user="u", password="p"),
        )
    tool._execute_query = AsyncMock(return_value=[{"VALUE": 1}])
    run_result = asyncio.run(tool._run(query="SELECT 1 AS value"))
    arun_result = asyncio.run(tool._arun(query="SELECT 1 AS value"))
    assert arun_result == run_result == [{"VALUE": 1}]
