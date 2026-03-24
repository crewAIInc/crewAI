import asyncio
from unittest.mock import MagicMock, patch

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool
from crewai_tools.tools.snowflake_search_tool.snowflake_search_tool import (
    SnowflakeSearchToolInput,
    _validate_identifier,
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

        # Execute multiple queries
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

        # Add connection to pool
        await snowflake_tool._get_connection()

        # Return connection to pool
        async with snowflake_tool._pool_lock:
            snowflake_tool._connection_pool.append(mock_snowflake_connection)

        # Trigger cleanup
        snowflake_tool.__del__()

        mock_snowflake_connection.close.assert_called_once()


def test_config_validation():
    # Test missing required fields
    with pytest.raises(ValueError):
        SnowflakeConfig()

    # Test invalid account format
    with pytest.raises(ValueError):
        SnowflakeConfig(
            account="invalid//account", user="test_user", password="test_pass"
        )

    # Test missing authentication
    with pytest.raises(ValueError):
        SnowflakeConfig(account="test_account", user="test_user")


# SQL Injection Prevention Tests
class TestIdentifierValidation:
    """Test that SQL injection payloads are rejected by identifier validation."""

    def test_valid_identifiers(self):
        """Valid Snowflake identifiers should pass validation."""
        valid_names = ["my_database", "PRODUCTION", "schema_v2", "DB$123", "_private"]
        for name in valid_names:
            assert _validate_identifier(name, "test") == name

    def test_sql_injection_semicolon(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier("test_db; DROP TABLE users; --", "database")

    def test_sql_injection_quotes(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier('test" OR 1=1 --', "database")

    def test_sql_injection_spaces(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier("test db", "database")

    def test_sql_injection_parentheses(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier("test()", "schema")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier("", "database")

    def test_starts_with_number_rejected(self):
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            _validate_identifier("1database", "database")


class TestInputSchemaValidation:
    """Test that SnowflakeSearchToolInput rejects injection payloads at the Pydantic level."""

    def test_valid_input(self):
        inp = SnowflakeSearchToolInput(
            query="SELECT 1", database="my_db", snowflake_schema="public"
        )
        assert inp.database == "my_db"
        assert inp.snowflake_schema == "public"

    def test_input_rejects_injection_in_database(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", database="db; DROP TABLE x; --"
            )

    def test_input_rejects_injection_in_schema(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", snowflake_schema="schema; DROP TABLE x; --"
            )

    def test_none_values_allowed(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1")
        assert inp.database is None
        assert inp.snowflake_schema is None


class TestConfigIdentifierValidation:
    """Test that SnowflakeConfig also rejects injection payloads in database/schema."""

    def test_config_rejects_injection_in_database(self):
        with pytest.raises(ValueError):
            SnowflakeConfig(
                account="test_account",
                user="test_user",
                password="test_pass",
                database="db; DROP TABLE users; --",
            )

    def test_config_rejects_injection_in_schema(self):
        with pytest.raises(ValueError):
            SnowflakeConfig(
                account="test_account",
                user="test_user",
                password="test_pass",
                snowflake_schema="schema; DROP TABLE users; --",
            )

    def test_config_accepts_valid_identifiers(self):
        config = SnowflakeConfig(
            account="test_account",
            user="test_user",
            password="test_pass",
            database="PRODUCTION_DB",
            snowflake_schema="analytics_v2",
        )
        assert config.database == "PRODUCTION_DB"
        assert config.snowflake_schema == "analytics_v2"


@pytest.mark.asyncio
async def test_run_rejects_injection_in_database(snowflake_tool, mock_snowflake_connection):
    """Defense-in-depth: even if Pydantic is bypassed, _run validates identifiers."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            await snowflake_tool._run(
                query="SELECT 1",
                database="db; DROP TABLE users; --",
            )


@pytest.mark.asyncio
async def test_run_rejects_injection_in_schema(snowflake_tool, mock_snowflake_connection):
    """Defense-in-depth: even if Pydantic is bypassed, _run validates identifiers."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection
        with pytest.raises(ValueError, match="Invalid Snowflake identifier"):
            await snowflake_tool._run(
                query="SELECT 1",
                snowflake_schema="schema; DROP TABLE users; --",
            )


@pytest.mark.asyncio
async def test_run_uses_quoted_identifiers(snowflake_tool, mock_snowflake_connection):
    """Valid identifiers should be double-quoted in the USE statements."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection
        await snowflake_tool._run(
            query="SELECT 1", database="my_db", snowflake_schema="my_schema"
        )
        cursor = mock_snowflake_connection.cursor.return_value
        calls = [str(c) for c in cursor.execute.call_args_list]
        assert any('"my_db"' in c for c in calls), f"Expected quoted database in calls: {calls}"
        assert any('"my_schema"' in c for c in calls), f"Expected quoted schema in calls: {calls}"
