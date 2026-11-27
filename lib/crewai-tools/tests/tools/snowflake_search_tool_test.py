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


class _FakeCursor:
    """Cursor that reports the database selected by the last USE statement."""

    def __init__(self) -> None:
        self._database: str | None = None
        self._description: list[tuple[str]] | None = None

    def execute(self, sql: str, timeout: int | None = None) -> None:
        if sql.startswith("USE DATABASE "):
            self._database = sql.rsplit(" ", 1)[-1]
            self._description = None
        else:
            self._description = [("value",)]

    @property
    def description(self) -> list[tuple[str]] | None:
        return self._description

    def fetchall(self) -> list[tuple[str | None]]:
        return [(self._database,)]

    def close(self) -> None:
        pass


class _FakeConnection:
    def __init__(self) -> None:
        self._cursor = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self._cursor


def test_overrides_do_not_reuse_results_cached_for_another_database(
    snowflake_tool,
):
    """The cache key must reflect the per-call database override."""
    import crewai_tools.tools.snowflake_search_tool.snowflake_search_tool as sf

    sf._query_cache.clear()
    connection = _FakeConnection()

    async def _get_connection():
        return connection

    with patch.object(snowflake_tool, "_get_connection", new=_get_connection):
        first = asyncio.run(
            snowflake_tool._run(query="SELECT value", database="db_one")
        )
        second = asyncio.run(
            snowflake_tool._run(query="SELECT value", database="db_two")
        )

    sf._query_cache.clear()

    assert first == [{"value": "db_one"}]
    assert second == [{"value": "db_two"}]
