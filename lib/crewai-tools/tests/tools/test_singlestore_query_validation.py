"""Unit tests for SingleStoreSearchTool._validate_query.

These do not need a SingleStore server — they construct the class without
calling __init__ so we can exercise the read-only gate in isolation.
"""

import pytest

from crewai_tools.tools.singlestore_search_tool.singlestore_search_tool import (
    SingleStoreSearchTool,
)


def _tool() -> SingleStoreSearchTool:
    return object.__new__(SingleStoreSearchTool)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select * from employees",
        "  SELECT id FROM t  ",
        "SELECT * FROM t;",
        "SHOW TABLES",
        "show columns from employees",
    ],
)
def test_validate_query_allows_single_select_or_show(query: str) -> None:
    ok, message = _tool()._validate_query(query)
    assert ok is True
    assert message == "Valid query"


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1; DROP TABLE employees",
        "select * from users; drop table users",
        "SHOW TABLES; DELETE FROM employees",
        "SHOW TABLES; DROP TABLE employees;--",
    ],
)
def test_validate_query_rejects_stacked_statements(query: str) -> None:
    ok, message = _tool()._validate_query(query)
    assert ok is False
    assert "Multiple SQL statements" in message


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM employees",
        "DROP TABLE employees",
        "INSERT INTO employees VALUES (1)",
        "UPDATE employees SET salary = 0",
        "",
        123,
    ],
)
def test_validate_query_rejects_writes_and_non_strings(query: object) -> None:
    ok, message = _tool()._validate_query(query)  # type: ignore[arg-type]
    assert ok is False
    assert message
