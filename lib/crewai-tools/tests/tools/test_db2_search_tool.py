"""Tests for DB2VectorSearchTool.

All tests are fully unit-tested — no real IBM DB2 instance is required.
ibm_db and ibm_db_dbi are mocked at import time so the suite runs without
those optional packages installed.
"""
from __future__ import annotations

import decimal
import datetime
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Stub ibm_db / ibm_db_dbi before any crewai_tools import, so the
# ImportString validator on DB2VectorSearchTool does not fail.
# ---------------------------------------------------------------------------

def _make_ibm_db_stub() -> ModuleType:
    mod = ModuleType("ibm_db")
    mod.connect = MagicMock()
    mod.close = MagicMock()
    return mod


def _make_ibm_db_dbi_stub() -> ModuleType:
    mod = ModuleType("ibm_db_dbi")

    class FakeConnection:
        def __init__(self, conn):
            self._conn = conn
            self.cursor = MagicMock(return_value=MagicMock())

        def close(self):
            pass

    mod.Connection = FakeConnection
    return mod


# Inject stubs before importing tool module
_ibm_db_stub = _make_ibm_db_stub()
_ibm_db_dbi_stub = _make_ibm_db_dbi_stub()
sys.modules.setdefault("ibm_db", _ibm_db_stub)
sys.modules.setdefault("ibm_db_dbi", _ibm_db_dbi_stub)

from crewai_tools.tools.db2_search_tool.db2_search_tool import (  # noqa: E402
    DB2JSONEncoder,
    DB2ToolSchema,
    DB2VectorSearchTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(
    *,
    table_name: str = "documents",
    vector_column: str = "embedding",
    return_columns: list[str] | None = None,
    limit: int = 3,
    distance_metric: str = "COSINE",
    max_distance: float | None = None,
    embedding_model: str = "text-embedding-3-large",
    custom_embedding_fn=None,
) -> DB2VectorSearchTool:
    """Return a DB2VectorSearchTool with mocked ibm_db packages."""
    return DB2VectorSearchTool(
        connection_string="DATABASE=TESTDB;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=user;PWD=pass;",
        table_name=table_name,
        vector_column=vector_column,
        return_columns=return_columns or ["content"],
        limit=limit,
        distance_metric=distance_metric,
        max_distance=max_distance,
        embedding_model=embedding_model,
        db2_package=_ibm_db_stub,
        db2_dbi_package=_ibm_db_dbi_stub,
        custom_embedding_fn=custom_embedding_fn,
    )


def _fake_embedding(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _make_cursor_with_rows(rows: list[tuple]) -> MagicMock:
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    return cursor


# ---------------------------------------------------------------------------
# DB2ToolSchema validation
# ---------------------------------------------------------------------------

class TestDB2ToolSchema:
    def test_valid_query_only(self):
        schema = DB2ToolSchema(query="find documents about AI")
        assert schema.query == "find documents about AI"
        assert schema.filter_by is None
        assert schema.filter_value is None

    def test_valid_query_with_filter_pair(self):
        schema = DB2ToolSchema(query="search", filter_by="category", filter_value="tech")
        assert schema.filter_by == "category"
        assert schema.filter_value == "tech"

    def test_filter_by_without_filter_value_raises(self):
        with pytest.raises(ValueError, match="filter_by and filter_value must be provided together"):
            DB2ToolSchema(query="search", filter_by="category")

    def test_filter_value_without_filter_by_raises(self):
        with pytest.raises(ValueError, match="filter_by and filter_value must be provided together"):
            DB2ToolSchema(query="search", filter_value="tech")

    def test_blank_filter_by_raises(self):
        with pytest.raises(ValueError, match="filter_by must be a non-empty column name"):
            DB2ToolSchema(query="search", filter_by="   ", filter_value="tech")

    def test_none_filter_by_and_none_filter_value_is_valid(self):
        schema = DB2ToolSchema(query="hello", filter_by=None, filter_value=None)
        assert schema.filter_by is None
        assert schema.filter_value is None


# ---------------------------------------------------------------------------
# DB2VectorSearchTool field validation
# ---------------------------------------------------------------------------

class TestDB2VectorSearchToolConfig:
    _conn = "DATABASE=MYDB;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=u;PWD=p;"

    def test_default_values(self):
        tool = DB2VectorSearchTool(
            connection_string=self._conn,
            db2_package=_ibm_db_stub,
            db2_dbi_package=_ibm_db_dbi_stub,
        )
        assert tool.return_columns == ["content"]
        assert tool.limit == 3
        assert tool.distance_metric == "COSINE"
        assert tool.max_distance is None

    def test_empty_return_columns_raises(self):
        with pytest.raises(ValueError, match="return_columns cannot be empty"):
            DB2VectorSearchTool(
                connection_string=self._conn,
                return_columns=[],
                db2_package=_ibm_db_stub,
                db2_dbi_package=_ibm_db_dbi_stub,
            )

    def test_limit_out_of_range_raises(self):
        with pytest.raises(ValueError):
            DB2VectorSearchTool(
                connection_string=self._conn,
                limit=0,
                db2_package=_ibm_db_stub,
                db2_dbi_package=_ibm_db_dbi_stub,
            )
        with pytest.raises(ValueError):
            DB2VectorSearchTool(
                connection_string=self._conn,
                limit=101,
                db2_package=_ibm_db_stub,
                db2_dbi_package=_ibm_db_dbi_stub,
            )

    def test_negative_max_distance_raises(self):
        with pytest.raises(ValueError):
            DB2VectorSearchTool(
                connection_string=self._conn,
                max_distance=-1.0,
                db2_package=_ibm_db_stub,
                db2_dbi_package=_ibm_db_dbi_stub,
            )

    def test_multiple_return_columns(self):
        tool = DB2VectorSearchTool(
            connection_string=self._conn,
            return_columns=["title", "body", "author"],
            db2_package=_ibm_db_stub,
            db2_dbi_package=_ibm_db_dbi_stub,
        )
        assert tool.return_columns == ["title", "body", "author"]


# ---------------------------------------------------------------------------
# DB2JSONEncoder
# ---------------------------------------------------------------------------

class TestDB2JSONEncoder:
    def test_encodes_decimal(self):
        result = json.dumps(decimal.Decimal("3.14"), cls=DB2JSONEncoder)
        assert result == "3.14"

    def test_encodes_datetime(self):
        dt = datetime.datetime(2024, 1, 15, 12, 0, 0)
        result = json.dumps(dt, cls=DB2JSONEncoder)
        assert "2024-01-15" in result

    def test_encodes_date(self):
        d = datetime.date(2024, 6, 1)
        result = json.dumps(d, cls=DB2JSONEncoder)
        assert "2024-06-01" in result

    def test_encodes_bytes(self):
        result = json.dumps(b"\x00\xff", cls=DB2JSONEncoder)
        assert "<binary_data>" in result

    def test_raises_for_unknown_type(self):
        class Unknown:
            pass
        with pytest.raises(TypeError):
            json.dumps(Unknown(), cls=DB2JSONEncoder)


# ---------------------------------------------------------------------------
# _validate_identifier (SQL injection guard)
# ---------------------------------------------------------------------------

class TestValidateIdentifier:
    def test_valid_simple_name(self):
        tool = _make_tool()
        assert tool._validate_identifier("documents") == "documents"
        assert tool._validate_identifier("my_table_1") == "my_table_1"

    def test_valid_schema_qualified_with_period(self):
        tool = _make_tool()
        assert tool._validate_identifier("myschema.documents", allow_period=True) == "myschema.documents"

    def test_period_without_allow_period_raises(self):
        tool = _make_tool()
        with pytest.raises(ValueError, match="Security Alert"):
            tool._validate_identifier("schema.table", allow_period=False)

    @pytest.mark.parametrize("bad_name", [
        "'; DROP TABLE documents; --",
        "table--",
        "col name",
        "col;name",
        "col OR 1=1",
        "",
        "1table",          # must start with a letter
        "123",             # must start with a letter
        ".documents",      # leading period
        "schema..table",   # double period
        "schema.table.extra",  # more than one period
        ".....",           # only dots — previously passed old regex
    ])
    def test_injection_strings_raise(self, bad_name: str):
        tool = _make_tool()
        with pytest.raises(ValueError, match="Security Alert"):
            tool._validate_identifier(bad_name)

    def test_allow_period_rejects_digit_led_schema(self):
        tool = _make_tool()
        with pytest.raises(ValueError, match="Security Alert"):
            tool._validate_identifier("1schema.table", allow_period=True)

    def test_allow_period_rejects_digit_led_table(self):
        tool = _make_tool()
        with pytest.raises(ValueError, match="Security Alert"):
            tool._validate_identifier("schema.1table", allow_period=True)


# ---------------------------------------------------------------------------
# _generate_embedding
# ---------------------------------------------------------------------------

class TestGenerateEmbedding:
    def test_uses_custom_embedding_fn(self):
        called_with = []

        def my_embed(text: str) -> list[float]:
            called_with.append(text)
            return [0.5, 0.5]

        tool = _make_tool(custom_embedding_fn=my_embed)
        result = tool._generate_embedding("hello world")
        assert result == [0.5, 0.5]
        assert called_with == ["hello world"]

    def test_falls_back_to_openai_with_api_key(self):
        tool = _make_tool()
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2])
        ]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"openai": mock_openai}):
                result = tool._generate_embedding("test query")

        assert result == [0.1, 0.2]

    def test_raises_when_no_openai_key_and_no_custom_fn(self):
        tool = _make_tool()

        with patch.dict("os.environ", {}, clear=True):
            # Remove OPENAI_API_KEY specifically if it exists
            import os
            env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict("os.environ", env_without_key, clear=True):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    tool._generate_embedding("test")


# ---------------------------------------------------------------------------
# _run — empty / whitespace query guard
# ---------------------------------------------------------------------------

class TestRunQueryValidation:
    def test_empty_query_returns_error_json(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        result = json.loads(tool._run(query=""))
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_whitespace_only_query_returns_error_json(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        result = json.loads(tool._run(query="   "))
        assert result["success"] is False

    def test_none_query_returns_error_json(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        result = json.loads(tool._run(query=None))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# _run — connection failure
# ---------------------------------------------------------------------------

class TestRunConnectionFailure:
    def test_connection_error_returns_error_json(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        with patch.object(tool, "_connect", side_effect=Exception("Connection refused")):
            result = json.loads(tool._run(query="find AI docs"))
        assert result["success"] is False
        assert "Failed to connect to DB2" in result["error"]


# ---------------------------------------------------------------------------
# _run — invalid distance metric
# ---------------------------------------------------------------------------

class TestRunInvalidMetric:
    def test_invalid_metric_returns_error_json(self):
        tool = _make_tool(
            custom_embedding_fn=_fake_embedding,
            distance_metric="INVALID_METRIC",
        )
        mock_cursor = _make_cursor_with_rows([])
        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                tool.cursor = mock_cursor
                result = json.loads(tool._run(query="test"))
        assert result["success"] is False
        assert "Invalid distance metric" in result["error"]


# ---------------------------------------------------------------------------
# _run — successful search (core happy path)
# ---------------------------------------------------------------------------

class TestRunSuccessful:
    def _setup_connected_tool(self, rows: list[tuple], **kwargs) -> DB2VectorSearchTool:
        tool = _make_tool(custom_embedding_fn=_fake_embedding, **kwargs)
        mock_cursor = _make_cursor_with_rows(rows)
        tool.cursor = mock_cursor
        return tool, mock_cursor

    def test_returns_results_as_json(self):
        rows = [("Some document text", 0.12)]
        tool, cursor = self._setup_connected_tool(rows)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(tool._run(query="find documents about AI"))

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["distance"] == pytest.approx(0.12)
        assert result["results"][0]["data"]["content"] == "Some document text"

    def test_multiple_return_columns_mapped_correctly(self):
        rows = [("Title A", "Body text A", 0.05)]
        tool, cursor = self._setup_connected_tool(
            rows, return_columns=["title", "body"]
        )

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(tool._run(query="search"))

        data = result["results"][0]["data"]
        assert data["title"] == "Title A"
        assert data["body"] == "Body text A"

    def test_empty_db_result_returns_empty_list(self):
        tool, _ = self._setup_connected_tool([])

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(tool._run(query="nothing"))

        assert result["success"] is True
        assert result["results"] == []

    def test_max_distance_filters_far_results(self):
        # Row 0 is close (0.2), Row 1 is too far (0.9)
        rows = [("Close doc", 0.2), ("Far doc", 0.9)]
        tool, _ = self._setup_connected_tool(rows, max_distance=0.5)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(tool._run(query="test"))

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["data"]["content"] == "Close doc"

    def test_filter_by_and_filter_value_added_to_params(self):
        rows = [("Filtered doc", 0.1)]
        tool, cursor = self._setup_connected_tool(rows)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(
                    tool._run(query="test", filter_by="category", filter_value="AI")
                )

        assert result["success"] is True
        # The second param in the execute call must be the filter value
        execute_args = cursor.execute.call_args
        params_tuple = execute_args[0][1]
        assert "AI" in params_tuple

    def test_sql_contains_correct_metric(self):
        rows = [("doc", 0.1)]
        tool, cursor = self._setup_connected_tool(rows, distance_metric="EUCLIDEAN")

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                tool._run(query="test")

        executed_sql = cursor.execute.call_args[0][0]
        assert "EUCLIDEAN" in executed_sql

    def test_sql_contains_correct_limit(self):
        rows = []
        tool, cursor = self._setup_connected_tool(rows, limit=7)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                tool._run(query="test")

        executed_sql = cursor.execute.call_args[0][0]
        assert "7" in executed_sql

    def test_sql_contains_where_clause_when_filter_provided(self):
        rows = []
        tool, cursor = self._setup_connected_tool(rows)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                tool._run(query="test", filter_by="dept", filter_value="HR")

        executed_sql = cursor.execute.call_args[0][0]
        assert "WHERE dept = ?" in executed_sql

    def test_sql_has_no_where_clause_without_filter(self):
        rows = []
        tool, cursor = self._setup_connected_tool(rows)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                tool._run(query="test")

        executed_sql = cursor.execute.call_args[0][0]
        assert "WHERE" not in executed_sql

    def test_json_encoder_handles_decimal_in_results(self):
        rows = [(decimal.Decimal("42.50"), 0.1)]
        tool, _ = self._setup_connected_tool(rows, return_columns=["price"])

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(tool._run(query="test"))

        assert result["success"] is True
        assert result["results"][0]["data"]["price"] == pytest.approx(42.5)

    def test_disconnect_called_after_successful_run(self):
        rows = [("doc", 0.1)]
        tool, cursor = self._setup_connected_tool(rows)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect") as mock_disconnect:
                tool._run(query="test")

        mock_disconnect.assert_called_once()

    def test_disconnect_called_on_unexpected_error(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect") as mock_disconnect:
                # cursor is None → will raise AttributeError inside _run
                tool.cursor = None
                # Override _connect to set cursor to a raising mock
                def bad_cursor_setup():
                    c = MagicMock()
                    c.execute.side_effect = RuntimeError("Unexpected DB error")
                    tool.cursor = c

                tool._connect = bad_cursor_setup
                result = json.loads(tool._run(query="test"))

        assert result["success"] is False
        mock_disconnect.assert_called()


# ---------------------------------------------------------------------------
# _run — SQL injection via filter_by rejected
# ---------------------------------------------------------------------------

class TestRunSQLInjectionPrevention:
    @pytest.mark.parametrize("bad_col", [
        "col; DROP TABLE documents; --",
        "col OR 1=1",
        "col name",
        # NOTE: empty string is falsy — _run skips the WHERE clause entirely
        # so it does NOT trigger _validate_identifier.  The schema-level guard
        # (DB2ToolSchema._validate_filter_pair) catches the empty string case.
    ])
    def test_injection_in_filter_by_returns_error(self, bad_col: str):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        mock_cursor = _make_cursor_with_rows([])
        tool.cursor = mock_cursor

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(
                    tool._run(query="test", filter_by=bad_col, filter_value="val")
                )

        assert result["success"] is False

    def test_empty_filter_by_bypasses_where_clause(self):
        """Empty string is falsy in Python — _run skips WHERE rather than injecting.
        The actual guard lives in DB2ToolSchema (schema-level validation).
        """
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        mock_cursor = _make_cursor_with_rows([])
        tool.cursor = mock_cursor

        with patch.object(tool, "_connect"):
            with patch.object(tool, "_disconnect"):
                result = json.loads(
                    tool._run(query="test", filter_by="", filter_value="val")
                )

        # The query succeeds (no WHERE clause injected) — success is True
        assert result["success"] is True
        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "WHERE" not in executed_sql


# ---------------------------------------------------------------------------
# _connect / _disconnect lifecycle
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    def test_connect_builds_connection_objects(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        mock_conn = MagicMock()
        _ibm_db_stub.connect.return_value = mock_conn

        tool._connect()

        assert tool.connection is mock_conn
        assert tool.cursor is not None

    def test_connect_opens_fresh_connection_each_call(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        mock_conn = MagicMock()
        local_connect = MagicMock(return_value=mock_conn)
        tool.db2_package = MagicMock()
        tool.db2_package.connect = local_connect
        tool.db2_package.close = MagicMock()

        tool._connect()
        tool._connect()  # connect-per-call: each invocation opens a new connection

        assert local_connect.call_count == 2

    def test_disconnect_resets_all_handles(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        mock_conn = MagicMock()
        _ibm_db_stub.connect.return_value = mock_conn

        tool._connect()
        tool._disconnect()

        assert tool.connection is None
        assert tool.dbi_connection is None
        assert tool.cursor is None

    def test_disconnect_is_safe_when_already_disconnected(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        # Should not raise even with no open connection
        tool._disconnect()

    def test_del_calls_disconnect(self):
        tool = _make_tool(custom_embedding_fn=_fake_embedding)
        with patch.object(tool, "_disconnect") as mock_disconnect:
            tool.__del__()
        mock_disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

class TestToolMetadata:
    def test_tool_name(self):
        tool = _make_tool()
        assert tool.name == "DB2VectorSearchTool"

    def test_tool_description(self):
        tool = _make_tool()
        assert "DB2" in tool.description

    def test_args_schema_is_db2_tool_schema(self):
        tool = _make_tool()
        assert tool.args_schema is DB2ToolSchema

    def test_package_dependencies_listed(self):
        tool = _make_tool()
        assert "ibm_db" in tool.package_dependencies

    def test_env_vars_declared(self):
        tool = _make_tool()
        env_var_names = {ev.name for ev in tool.env_vars}
        assert "OPENAI_API_KEY" in env_var_names
        assert "DB2_CONNECTION_STRING" in env_var_names
