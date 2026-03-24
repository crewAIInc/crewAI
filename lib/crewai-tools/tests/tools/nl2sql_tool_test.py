from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.nl2sql.nl2sql_tool import NL2SQLTool


class TestNL2SQLToolValidateIdentifier:
    """Tests for SQL injection prevention via identifier validation."""

    def test_valid_identifiers(self):
        assert NL2SQLTool._validate_identifier("users", "table_name") == "users"
        assert NL2SQLTool._validate_identifier("MY_TABLE", "table_name") == "MY_TABLE"
        assert NL2SQLTool._validate_identifier("table$1", "table_name") == "table$1"
        assert NL2SQLTool._validate_identifier("_private", "table_name") == "_private"

    def test_rejects_sql_injection_with_semicolon(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("users; DROP TABLE users;--", "table_name")

    def test_rejects_sql_injection_with_quotes(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("users'--", "table_name")

    def test_rejects_sql_injection_with_spaces(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("users DROP TABLE", "table_name")

    def test_rejects_leading_number(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("1table", "table_name")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("", "table_name")

    def test_rejects_parentheses(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("users()", "table_name")

    def test_rejects_dash_comment(self):
        with pytest.raises(ValueError, match="Invalid table_name"):
            NL2SQLTool._validate_identifier("users--comment", "table_name")


@patch("crewai_tools.tools.nl2sql.nl2sql_tool.SQLALCHEMY_AVAILABLE", True)
class TestNL2SQLToolFetchColumns:
    """Tests that _fetch_all_available_columns validates table names."""

    def _make_tool(self):
        """Create an NL2SQLTool instance bypassing model_post_init DB calls."""
        with patch.object(NL2SQLTool, "model_post_init"):
            tool = NL2SQLTool(
                db_uri="sqlite:///:memory:",
                name="NL2SQLTool",
                description="test",
            )
        return tool

    def test_rejects_malicious_table_name(self):
        tool = self._make_tool()
        with pytest.raises(ValueError, match="Invalid table_name"):
            tool._fetch_all_available_columns("users'; DROP TABLE users;--")

    def test_accepts_valid_table_name(self):
        tool = self._make_tool()
        with patch.object(NL2SQLTool, "execute_sql", return_value=[]) as mock_exec:
            result = tool._fetch_all_available_columns("valid_table")
            mock_exec.assert_called_once()
            call_sql = mock_exec.call_args[0][0]
            assert "valid_table" in call_sql
            assert result == []
