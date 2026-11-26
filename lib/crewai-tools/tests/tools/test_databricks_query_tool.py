"""Tests for the Databricks SQL query tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from crewai_tools.tools.databricks_query_tool.databricks_query_tool import (
    DatabricksQueryTool,
)


SCHEMA_COLUMNS = ["alpha", "beta", "gamma", "delta", "epsilon"]


class _FakeStatementExecution:
    def __init__(self, result: Any) -> None:
        self._result = result

    def execute_statement(self, **kwargs: Any) -> Any:
        return SimpleNamespace(statement_id="statement-1")

    def get_statement(self, statement_id: str) -> Any:
        return self._result


class _FakeWorkspaceClient:
    def __init__(self, result: Any) -> None:
        self.statement_execution = _FakeStatementExecution(result)


def _make_tool(monkeypatch: pytest.MonkeyPatch, result: Any) -> DatabricksQueryTool:
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    tool = DatabricksQueryTool(default_warehouse_id="warehouse-1")
    tool._workspace_client = _FakeWorkspaceClient(result)  # type: ignore[assignment]
    return tool


def _result_with_rows(rows: list[list[Any]]) -> Any:
    return SimpleNamespace(
        status=SimpleNamespace(state="SUCCEEDED"),
        manifest=SimpleNamespace(
            schema=SimpleNamespace(
                columns=[SimpleNamespace(name=name) for name in SCHEMA_COLUMNS]
            )
        ),
        result=SimpleNamespace(data_array=[rows], data=None),
    )


def _header_columns(output: str) -> list[str]:
    return [part.strip() for part in output.splitlines()[0].split("|")]


def test_result_columns_follow_schema_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Columns must keep the order reported by the result schema."""
    tool = _make_tool(monkeypatch, _result_with_rows([[1, 2, 3, 4, 5]]))

    output = tool._run(query="SELECT * FROM my_table")

    assert _header_columns(output) == SCHEMA_COLUMNS


def test_extra_values_are_appended_after_schema_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Values without a schema column go to the end, not into the middle."""
    tool = _make_tool(monkeypatch, _result_with_rows([[1, 2, 3, 4, 5, 6]]))

    output = tool._run(query="SELECT * FROM my_table")

    assert _header_columns(output) == [*SCHEMA_COLUMNS, "Column_5"]
