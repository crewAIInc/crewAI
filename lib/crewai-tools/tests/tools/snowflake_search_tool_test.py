import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool
from crewai_tools.tools.snowflake_search_tool.snowflake_search_tool import (
    _validate_snowflake_identifier,
)
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


@pytest.mark.parametrize(
    ("name", "allow_qualified"),
    [
        ("analytics", False),
        ("ANALYTICS", False),
        ("_tmp", False),
        ("db_1$", False),
        ("analytics", True),
        ("analytics.public", True),
        ("DB1.SCHEMA_1", True),
    ],
)
def test_validate_snowflake_identifier_accepts_safe_names(
    name: str, allow_qualified: bool
) -> None:
    assert (
        _validate_snowflake_identifier(name, allow_qualified=allow_qualified) == name
    )


@pytest.mark.parametrize(
    "name",
    [
        "analytics; DROP DATABASE prod",
        "analytics;DROP DATABASE prod",
        "analytics --",
        "analytics/*comment*/",
        "analytics public",
        "1analytics",
        "",
        ".",
        "analytics.public.extra",
        'analytics"',
        "analytics'",
    ],
)
def test_validate_snowflake_identifier_rejects_injection(name: str) -> None:
    with pytest.raises(ValueError, match="valid identifier"):
        _validate_snowflake_identifier(name)


def test_validate_snowflake_identifier_rejects_qualified_when_disallowed() -> None:
    with pytest.raises(ValueError, match="valid identifier"):
        _validate_snowflake_identifier("analytics.public")


def _fake_snowflake_tool() -> MagicMock:
    """Bind SnowflakeSearchTool._run without constructing a real connection."""
    tool = MagicMock()
    tool._execute_query = AsyncMock(return_value=[])
    return tool


@pytest.mark.asyncio
async def test_run_uses_validated_database_and_schema() -> None:
    tool = _fake_snowflake_tool()
    await SnowflakeSearchTool._run(
        tool,
        query="SELECT 1",
        database="analytics",
        snowflake_schema="public",
    )

    assert tool._execute_query.await_args_list[0].args[0] == "USE DATABASE analytics"
    assert tool._execute_query.await_args_list[1].args[0] == "USE SCHEMA public"
    assert tool._execute_query.await_args_list[2].args == ("SELECT 1", 300)


@pytest.mark.asyncio
async def test_run_accepts_qualified_schema() -> None:
    tool = _fake_snowflake_tool()
    await SnowflakeSearchTool._run(
        tool,
        query="SELECT 1",
        snowflake_schema="analytics.public",
    )

    assert (
        tool._execute_query.await_args_list[0].args[0] == "USE SCHEMA analytics.public"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("database", "snowflake_schema"),
    [
        ("analytics; DROP DATABASE prod", None),
        ("analytics--comment", None),
        (None, "public; DROP SCHEMA secret"),
        (None, "public.extra.evil"),
    ],
)
async def test_run_rejects_injected_database_or_schema(
    database: str | None, snowflake_schema: str | None
) -> None:
    tool = _fake_snowflake_tool()
    with pytest.raises(ValueError, match="valid identifier"):
        await SnowflakeSearchTool._run(
            tool,
            query="SELECT 1",
            database=database,
            snowflake_schema=snowflake_schema,
        )
    tool._execute_query.assert_not_called()
