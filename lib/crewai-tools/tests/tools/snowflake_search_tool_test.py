import asyncio
from unittest.mock import MagicMock, patch

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool
from crewai_tools.tools.snowflake_search_tool.snowflake_search_tool import (
    SnowflakeSearchToolInput,
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
class TestSnowflakeSearchToolInputValidation:
    """Tests for SQL injection prevention via input schema validation."""

    def test_valid_database_identifier(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1", database="my_database")
        assert inp.database == "my_database"

    def test_valid_schema_identifier(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1", snowflake_schema="public")
        assert inp.snowflake_schema == "public"

    def test_valid_identifier_with_dollar_sign(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1", database="my$db")
        assert inp.database == "my$db"

    def test_database_with_sql_injection_semicolon(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", database="test_db; DROP TABLE users; --"
            )

    def test_schema_with_sql_injection_semicolon(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", snowflake_schema="public; DROP TABLE users; --"
            )

    def test_database_with_sql_injection_spaces(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", database="test_db DROP TABLE"
            )

    def test_schema_with_sql_injection_quotes(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", snowflake_schema="public\"--"
            )

    def test_database_with_sql_injection_dash_comment(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(
                query="SELECT 1", database="test--comment"
            )

    def test_database_starting_with_number(self):
        with pytest.raises(ValueError):
            SnowflakeSearchToolInput(query="SELECT 1", database="1invalid")

    def test_none_database_is_allowed(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1", database=None)
        assert inp.database is None

    def test_none_schema_is_allowed(self):
        inp = SnowflakeSearchToolInput(query="SELECT 1", snowflake_schema=None)
        assert inp.snowflake_schema is None


class TestSnowflakeSearchToolValidateIdentifier:
    """Tests for the _validate_identifier runtime check."""

    def test_valid_identifiers(self):
        assert SnowflakeSearchTool._validate_identifier("my_db", "database") == "my_db"
        assert SnowflakeSearchTool._validate_identifier("PROD_DB", "database") == "PROD_DB"
        assert SnowflakeSearchTool._validate_identifier("schema$1", "schema") == "schema$1"
        assert SnowflakeSearchTool._validate_identifier("_private", "schema") == "_private"

    def test_rejects_semicolons(self):
        with pytest.raises(ValueError, match="Invalid database"):
            SnowflakeSearchTool._validate_identifier("db; DROP TABLE users;--", "database")

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid schema"):
            SnowflakeSearchTool._validate_identifier("public schema", "schema")

    def test_rejects_quotes(self):
        with pytest.raises(ValueError, match="Invalid database"):
            SnowflakeSearchTool._validate_identifier('db"--', "database")

    def test_rejects_leading_number(self):
        with pytest.raises(ValueError, match="Invalid database"):
            SnowflakeSearchTool._validate_identifier("1db", "database")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="Invalid database"):
            SnowflakeSearchTool._validate_identifier("", "database")


@pytest.mark.asyncio
async def test_run_uses_quoted_identifiers(snowflake_tool, mock_snowflake_connection):
    """Verify that _run wraps database/schema in double quotes in the SQL."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        await snowflake_tool._run(
            query="SELECT 1",
            database="my_db",
            snowflake_schema="my_schema",
        )

        calls = mock_snowflake_connection.cursor().execute.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert 'USE DATABASE "my_db"' in sql_statements
        assert 'USE SCHEMA "my_schema"' in sql_statements


@pytest.mark.asyncio
async def test_run_rejects_malicious_database(snowflake_tool, mock_snowflake_connection):
    """Verify that _run raises ValueError for SQL injection attempts in database."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        with pytest.raises(ValueError, match="Invalid database"):
            await snowflake_tool._run(
                query="SELECT 1",
                database="test_db; DROP TABLE users; --",
            )


@pytest.mark.asyncio
async def test_run_rejects_malicious_schema(snowflake_tool, mock_snowflake_connection):
    """Verify that _run raises ValueError for SQL injection attempts in schema."""
    with patch.object(snowflake_tool, "_create_connection") as mock_create_conn:
        mock_create_conn.return_value = mock_snowflake_connection

        with pytest.raises(ValueError, match="Invalid schema"):
            await snowflake_tool._run(
                query="SELECT 1",
                snowflake_schema="public; DROP TABLE users; --",
            )
