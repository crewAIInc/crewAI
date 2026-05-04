"""Security tests for NL2SQLTool.

Uses an in-memory SQLite database so no external service is needed.
SQLite does not have information_schema, so we patch the schema-introspection
helpers to avoid bootstrap failures and focus purely on the security logic.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module if SQLAlchemy is not installed
pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, text  # noqa: E402

from crewai_tools.tools.nl2sql.nl2sql_tool import NL2SQLTool  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQLITE_URI = "sqlite://"  # in-memory


def _make_tool(allow_dml: bool = False, **kwargs) -> NL2SQLTool:
    """Return a NL2SQLTool wired to an in-memory SQLite DB.

    Schema-introspection is patched out so we can create the tool without a
    real PostgreSQL information_schema.
    """
    with (
        patch.object(NL2SQLTool, "_fetch_available_tables", return_value=[]),
        patch.object(NL2SQLTool, "_fetch_all_available_columns", return_value=[]),
    ):
        return NL2SQLTool(db_uri=SQLITE_URI, allow_dml=allow_dml, **kwargs)


# ---------------------------------------------------------------------------
# Read-only enforcement (allow_dml=False)
# ---------------------------------------------------------------------------


class TestReadOnlyMode:
    def test_select_allowed_by_default(self):
        tool = _make_tool()
        # SQLite supports SELECT without information_schema
        result = tool.execute_sql("SELECT 1 AS val")
        assert result == [{"val": 1}]

    @pytest.mark.parametrize(
        "stmt",
        [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET col = 1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD col TEXT",
            "CREATE TABLE t (id INTEGER)",
            "TRUNCATE TABLE t",
            "GRANT SELECT ON t TO user1",
            "REVOKE SELECT ON t FROM user1",
            "EXEC sp_something",
            "EXECUTE sp_something",
            "CALL proc()",
        ],
    )
    def test_write_statements_blocked_by_default(self, stmt: str):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(stmt)

    def test_explain_allowed(self):
        tool = _make_tool()
        # Should not raise
        tool._validate_query("EXPLAIN SELECT 1")

    def test_read_only_cte_allowed(self):
        tool = _make_tool()
        tool._validate_query("WITH cte AS (SELECT 1) SELECT * FROM cte")

    def test_show_allowed(self):
        tool = _make_tool()
        tool._validate_query("SHOW TABLES")

    def test_describe_allowed(self):
        tool = _make_tool()
        tool._validate_query("DESCRIBE users")


# ---------------------------------------------------------------------------
# DML enabled (allow_dml=True)
# ---------------------------------------------------------------------------


class TestDMLEnabled:
    def test_insert_allowed_when_dml_enabled(self):
        tool = _make_tool(allow_dml=True)
        # Should not raise
        tool._validate_query("INSERT INTO t VALUES (1)")

    def test_delete_allowed_when_dml_enabled(self):
        tool = _make_tool(allow_dml=True)
        tool._validate_query("DELETE FROM t WHERE id = 1")

    def test_drop_allowed_when_dml_enabled(self):
        tool = _make_tool(allow_dml=True)
        tool._validate_query("DROP TABLE t")

    def test_dml_actually_persists(self):
        """End-to-end: INSERT commits when allow_dml=True."""
        # Use a file-based SQLite so we can verify persistence across sessions
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        uri = f"sqlite:///{db_path}"
        try:
            tool = _make_tool(allow_dml=True)
            tool.db_uri = uri

            engine = create_engine(uri)
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY)"))
                conn.commit()

            tool.execute_sql("INSERT INTO items VALUES (42)")

            with engine.connect() as conn:
                rows = conn.execute(text("SELECT id FROM items")).fetchall()
            assert (42,) in rows
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Parameterised query — SQL injection prevention
# ---------------------------------------------------------------------------


class TestParameterisedQueries:
    def test_table_name_is_parameterised(self):
        """_fetch_all_available_columns must not interpolate table_name into SQL."""
        tool = _make_tool()
        captured_calls = []

        def recording_execute_sql(self_inner, sql_query, params=None):
            captured_calls.append((sql_query, params))
            return []

        with patch.object(NL2SQLTool, "execute_sql", recording_execute_sql):
            tool._fetch_all_available_columns("users'; DROP TABLE users; --")

        assert len(captured_calls) == 1
        sql, params = captured_calls[0]
        # The raw SQL must NOT contain the injected string
        assert "DROP" not in sql
        # The table name must be passed as a parameter
        assert params is not None
        assert params.get("table_name") == "users'; DROP TABLE users; --"
        # The SQL template must use the :param syntax
        assert ":table_name" in sql

    def test_injection_string_not_in_sql_template(self):
        """The f-string vulnerability is gone — table name never lands in the SQL."""
        tool = _make_tool()
        injection = "'; DROP TABLE users; --"
        captured = {}

        def spy(self_inner, sql_query, params=None):
            captured["sql"] = sql_query
            captured["params"] = params
            return []

        with patch.object(NL2SQLTool, "execute_sql", spy):
            tool._fetch_all_available_columns(injection)

        assert injection not in captured["sql"]
        assert captured["params"]["table_name"] == injection


# ---------------------------------------------------------------------------
# session.commit() not called for read-only queries
# ---------------------------------------------------------------------------


class TestNoCommitForReadOnly:
    def test_select_does_not_commit(self):
        tool = _make_tool(allow_dml=False)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["val"]
        mock_result.fetchall.return_value = [(1,)]
        mock_session.execute.return_value = mock_result

        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql("SELECT 1")

        mock_session.commit.assert_not_called()

    def test_write_with_dml_enabled_does_commit(self):
        tool = _make_tool(allow_dml=True)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = False
        mock_session.execute.return_value = mock_result

        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql("INSERT INTO t VALUES (1)")

        mock_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Environment-variable escape hatch
# ---------------------------------------------------------------------------


class TestEnvVarEscapeHatch:
    def test_env_var_enables_dml(self):
        with patch.dict(os.environ, {"CREWAI_NL2SQL_ALLOW_DML": "true"}):
            tool = _make_tool(allow_dml=False)
        assert tool.allow_dml is True

    def test_env_var_case_insensitive(self):
        with patch.dict(os.environ, {"CREWAI_NL2SQL_ALLOW_DML": "TRUE"}):
            tool = _make_tool(allow_dml=False)
        assert tool.allow_dml is True

    def test_env_var_absent_keeps_default(self):
        env = {k: v for k, v in os.environ.items() if k != "CREWAI_NL2SQL_ALLOW_DML"}
        with patch.dict(os.environ, env, clear=True):
            tool = _make_tool(allow_dml=False)
        assert tool.allow_dml is False

    def test_env_var_false_does_not_enable_dml(self):
        with patch.dict(os.environ, {"CREWAI_NL2SQL_ALLOW_DML": "false"}):
            tool = _make_tool(allow_dml=False)
        assert tool.allow_dml is False

    def test_dml_write_blocked_without_env_var(self):
        env = {k: v for k, v in os.environ.items() if k != "CREWAI_NL2SQL_ALLOW_DML"}
        with patch.dict(os.environ, env, clear=True):
            tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("DROP TABLE sensitive_data")


# ---------------------------------------------------------------------------
# _run() propagates ValueError from _validate_query
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_run_raises_on_blocked_query(self):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._run("DELETE FROM users")

    def test_run_returns_results_for_select(self):
        tool = _make_tool(allow_dml=False)
        result = tool._run("SELECT 1 AS n")
        assert result == [{"n": 1}]


# ---------------------------------------------------------------------------
# Multi-statement / semicolon injection prevention
# ---------------------------------------------------------------------------


class TestSemicolonInjection:
    def test_multi_statement_blocked_in_read_only_mode(self):
        """SELECT 1; DROP TABLE users must be rejected when allow_dml=False."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="multi-statement"):
            tool._validate_query("SELECT 1; DROP TABLE users")

    def test_multi_statement_blocked_even_with_only_selects(self):
        """Two SELECT statements are still rejected in read-only mode."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="multi-statement"):
            tool._validate_query("SELECT 1; SELECT 2")

    def test_trailing_semicolon_allowed_single_statement(self):
        """A single statement with a trailing semicolon should pass."""
        tool = _make_tool(allow_dml=False)
        # Should not raise — the part after the semicolon is empty
        tool._validate_query("SELECT 1;")

    def test_multi_statement_allowed_when_dml_enabled(self):
        """Multiple statements are permitted when allow_dml=True."""
        tool = _make_tool(allow_dml=True)
        # Should not raise
        tool._validate_query("SELECT 1; INSERT INTO t VALUES (1)")

    def test_multi_statement_write_still_blocked_individually(self):
        """Even with allow_dml=False, a single write statement is blocked."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("DROP TABLE users")


# ---------------------------------------------------------------------------
# Writable CTEs (WITH … DELETE/INSERT/UPDATE)
# ---------------------------------------------------------------------------


class TestWritableCTE:
    def test_writable_cte_delete_blocked_in_read_only(self):
        """WITH d AS (DELETE FROM users RETURNING *) SELECT * FROM d — blocked."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH deleted AS (DELETE FROM users RETURNING *) SELECT * FROM deleted"
            )

    def test_writable_cte_insert_blocked_in_read_only(self):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH ins AS (INSERT INTO t VALUES (1) RETURNING id) SELECT * FROM ins"
            )

    def test_writable_cte_update_blocked_in_read_only(self):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH upd AS (UPDATE t SET x=1 RETURNING id) SELECT * FROM upd"
            )

    def test_writable_cte_allowed_when_dml_enabled(self):
        tool = _make_tool(allow_dml=True)
        # Should not raise
        tool._validate_query(
            "WITH deleted AS (DELETE FROM users RETURNING *) SELECT * FROM deleted"
        )

    def test_plain_read_only_cte_still_allowed(self):
        tool = _make_tool(allow_dml=False)
        # No write commands in the CTE body — must pass
        tool._validate_query("WITH cte AS (SELECT id FROM users) SELECT * FROM cte")

    def test_cte_with_comment_column_not_false_positive(self):
        """Column named 'comment' should NOT trigger writable CTE detection."""
        tool = _make_tool(allow_dml=False)
        # 'comment' is a column name, not a SQL command
        tool._validate_query(
            "WITH cte AS (SELECT comment FROM posts) SELECT * FROM cte"
        )

    def test_cte_with_set_column_not_false_positive(self):
        """Column named 'set' should NOT trigger writable CTE detection."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query(
            "WITH cte AS (SELECT set, reset FROM config) SELECT * FROM cte"
        )


# ---------------------------------------------------------------------------
# EXPLAIN ANALYZE executes the underlying query
# ---------------------------------------------------------------------------


    def test_cte_with_write_main_query_blocked(self):
        """WITH cte AS (SELECT 1) DELETE FROM users — main query must be caught."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH cte AS (SELECT 1) DELETE FROM users"
            )

    def test_cte_with_write_main_query_allowed_with_dml(self):
        """Main query write after CTE should pass when allow_dml=True."""
        tool = _make_tool(allow_dml=True)
        tool._validate_query(
            "WITH cte AS (SELECT id FROM users) INSERT INTO archive SELECT * FROM cte"
        )

    def test_cte_with_newline_before_paren_blocked(self):
        """AS followed by newline then ( should still detect writable CTE."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH cte AS\n(DELETE FROM users RETURNING *) SELECT * FROM cte"
            )

    def test_cte_with_tab_before_paren_blocked(self):
        """AS followed by tab then ( should still detect writable CTE."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH cte AS\t(DELETE FROM users RETURNING *) SELECT * FROM cte"
            )


class TestExplainAnalyze:
    def test_explain_analyze_delete_blocked_in_read_only(self):
        """EXPLAIN ANALYZE DELETE actually runs the delete — block it."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN ANALYZE DELETE FROM users")

    def test_explain_analyse_delete_blocked_in_read_only(self):
        """British spelling ANALYSE is also caught."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN ANALYSE DELETE FROM users")

    def test_explain_analyze_drop_blocked_in_read_only(self):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN ANALYZE DROP TABLE users")

    def test_explain_analyze_select_allowed_in_read_only(self):
        """EXPLAIN ANALYZE on a SELECT is safe — must be permitted."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query("EXPLAIN ANALYZE SELECT * FROM users")

    def test_explain_without_analyze_allowed(self):
        tool = _make_tool(allow_dml=False)
        tool._validate_query("EXPLAIN SELECT * FROM users")

    def test_explain_analyze_delete_allowed_when_dml_enabled(self):
        tool = _make_tool(allow_dml=True)
        tool._validate_query("EXPLAIN ANALYZE DELETE FROM users")

    def test_explain_paren_analyze_delete_blocked_in_read_only(self):
        """EXPLAIN (ANALYZE) DELETE actually runs the delete — block it."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN (ANALYZE) DELETE FROM users")

    def test_explain_paren_analyze_verbose_delete_blocked_in_read_only(self):
        """EXPLAIN (ANALYZE, VERBOSE) DELETE actually runs the delete — block it."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN (ANALYZE, VERBOSE) DELETE FROM users")

    def test_explain_paren_verbose_select_allowed_in_read_only(self):
        """EXPLAIN (VERBOSE) SELECT is safe — no ANALYZE means no execution."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query("EXPLAIN (VERBOSE) SELECT * FROM users")


# ---------------------------------------------------------------------------
# Multi-statement commit covers ALL statements (not just the first)
# ---------------------------------------------------------------------------


class TestMultiStatementCommit:
    def test_select_then_insert_triggers_commit(self):
        """SELECT 1; INSERT … — commit must happen because INSERT is a write."""
        tool = _make_tool(allow_dml=True)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = False
        mock_session.execute.return_value = mock_result
        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql("SELECT 1; INSERT INTO t VALUES (1)")

        mock_session.commit.assert_called_once()

    def test_select_only_multi_statement_does_not_commit(self):
        """Two SELECTs must not trigger a commit even when allow_dml=True."""
        tool = _make_tool(allow_dml=True)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["v"]
        mock_result.fetchall.return_value = [(1,)]
        mock_session.execute.return_value = mock_result
        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql("SELECT 1; SELECT 2")

    def test_writable_cte_triggers_commit(self):
        """WITH d AS (DELETE ...) must trigger commit when allow_dml=True."""
        tool = _make_tool(allow_dml=True)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["id"]
        mock_result.fetchall.return_value = [(1,)]
        mock_session.execute.return_value = mock_result
        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql(
                "WITH d AS (DELETE FROM users RETURNING *) SELECT * FROM d"
            )
            mock_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Extended _WRITE_COMMANDS coverage
# ---------------------------------------------------------------------------


class TestExtendedWriteCommands:
    @pytest.mark.parametrize(
        "stmt",
        [
            "UPSERT INTO t VALUES (1)",
            "LOAD DATA INFILE 'f.csv' INTO TABLE t",
            "COPY t FROM '/tmp/f.csv'",
            "VACUUM ANALYZE t",
            "ANALYZE t",
            "ANALYSE t",
            "REINDEX TABLE t",
            "CLUSTER t USING idx",
            "REFRESH MATERIALIZED VIEW v",
            "COMMENT ON TABLE t IS 'desc'",
            "SET search_path = myschema",
            "RESET search_path",
        ],
    )
    def test_extended_write_commands_blocked_by_default(self, stmt: str):
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(stmt)


# ---------------------------------------------------------------------------
# EXPLAIN ANALYZE VERBOSE handling
# ---------------------------------------------------------------------------


class TestExplainAnalyzeVerbose:
    def test_explain_analyze_verbose_select_allowed(self):
        """EXPLAIN ANALYZE VERBOSE SELECT should be allowed (read-only)."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query("EXPLAIN ANALYZE VERBOSE SELECT * FROM users")

    def test_explain_analyze_verbose_delete_blocked(self):
        """EXPLAIN ANALYZE VERBOSE DELETE should be blocked."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query("EXPLAIN ANALYZE VERBOSE DELETE FROM users")

    def test_explain_verbose_select_allowed(self):
        """EXPLAIN VERBOSE SELECT (no ANALYZE) should be allowed."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query("EXPLAIN VERBOSE SELECT * FROM users")


# ---------------------------------------------------------------------------
# CTE with string literal parens
# ---------------------------------------------------------------------------


class TestCTEStringLiteralParens:
    def test_cte_string_paren_does_not_bypass(self):
        """Parens inside string literals should not confuse the paren walker."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH cte AS (SELECT '(' FROM t) DELETE FROM users"
            )

    def test_cte_string_paren_read_only_allowed(self):
        """Read-only CTE with string literal parens should be allowed."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query(
            "WITH cte AS (SELECT '(' FROM t) SELECT * FROM cte"
        )


# ---------------------------------------------------------------------------
# EXPLAIN ANALYZE commit logic
# ---------------------------------------------------------------------------


class TestExplainAnalyzeCommit:
    def test_explain_analyze_delete_triggers_commit(self):
        """EXPLAIN ANALYZE DELETE should trigger commit when allow_dml=True."""
        tool = _make_tool(allow_dml=True)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["QUERY PLAN"]
        mock_result.fetchall.return_value = [("Delete on users",)]
        mock_session.execute.return_value = mock_result
        mock_session_cls = MagicMock(return_value=mock_session)

        with (
            patch("crewai_tools.tools.nl2sql.nl2sql_tool.create_engine"),
            patch(
                "crewai_tools.tools.nl2sql.nl2sql_tool.sessionmaker",
                return_value=mock_session_cls,
            ),
        ):
            tool.execute_sql("EXPLAIN ANALYZE DELETE FROM users")
            mock_session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# AS( inside string literals must not confuse CTE detection
# ---------------------------------------------------------------------------


class TestCTEStringLiteralAS:
    def test_as_paren_inside_string_does_not_bypass(self):
        """'AS (' inside a string literal must not be treated as a CTE body."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="read-only mode"):
            tool._validate_query(
                "WITH cte AS (SELECT 'AS (' FROM t) DELETE FROM users"
            )

    def test_as_paren_inside_string_read_only_ok(self):
        """Read-only CTE with 'AS (' in a string should be allowed."""
        tool = _make_tool(allow_dml=False)
        tool._validate_query(
            "WITH cte AS (SELECT 'AS (' FROM t) SELECT * FROM cte"
        )


# ---------------------------------------------------------------------------
# Unknown command after CTE should be blocked
# ---------------------------------------------------------------------------


class TestCTEUnknownCommand:
    def test_unknown_command_after_cte_blocked(self):
        """WITH cte AS (SELECT 1) FOOBAR should be blocked as unknown."""
        tool = _make_tool(allow_dml=False)
        with pytest.raises(ValueError, match="unrecognised"):
            tool._validate_query("WITH cte AS (SELECT 1) FOOBAR")
