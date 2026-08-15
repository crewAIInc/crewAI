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


@pytest.mark.parametrize(
    "query,keyword",
    [
        ("SELECT * FROM employees INTO OUTFILE '/tmp/data.csv'", "OUTFILE"),
        ("SELECT * FROM employees INTO DUMPFILE '/tmp/data.bin'", "DUMPFILE"),
        ("SELECT * FROM employees INTO FS '/mnt/data'", "FS"),
        ("SELECT * FROM employees INTO LINK 's3://bucket'", "LINK"),
        ("SELECT * FROM employees INTO S3 's3://bucket/key'", "S3"),
        ("SELECT * FROM employees INTO HDFS '/path'", "HDFS"),
        ("SELECT * FROM employees INTO AZURE 'wasb://container'", "AZURE"),
        ("SELECT * FROM employees INTO GCS 'gs://bucket'", "GCS"),
        ("SELECT * FROM employees INTO KAFKA 'kafka:topic'", "KAFKA"),
        ("select id from t into outfile '/tmp/out.csv'", "OUTFILE"),
        ("SELECT * FROM t INTO STAGE 'result.csv'", "STAGE"),
        ("select id from employees into stage 'out.csv'", "STAGE"),
    ],
)
def test_validate_query_rejects_select_into_write_targets(query: str, keyword: str) -> None:
    """SELECT … INTO <write-target> is rejected even though it starts with SELECT."""
    ok, message = _tool()._validate_query(query)
    assert ok is False
    assert keyword in message


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM employees FOR UPDATE",
        "SELECT id FROM t WHERE id = 1 FOR UPDATE",
        "select * from users for update",
        "SELECT * FROM employees LOCK IN SHARE MODE",
        "SELECT * FROM employees LOCK IN SHARE MODE",
        "select * from t lock in share mode",
    ],
)
def test_validate_query_rejects_locking_clauses(query: str) -> None:
    """FOR UPDATE and LOCK IN SHARE MODE acquire row locks that can deadlock."""
    ok, message = _tool()._validate_query(query)
    assert ok is False
    assert ("FOR UPDATE" in message) or ("LOCK" in message)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 3.14 INTO @pi",
        "SELECT 1 INTO @result",
        "select name from t into @x",
    ],
)
def test_validate_query_rejects_select_into_variable(query: str) -> None:
    """SELECT … INTO @<variable> assigns a session variable and is not read-only."""
    ok, message = _tool()._validate_query(query)
    assert ok is False
    assert "variable" in message


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1 INTO/**/S3",
        "SELECT * FROM t FOR/**/UPDATE",
        "SELECT 1 INTO /* comment */ OUTFILE",
        "SELECT * FROM t /**/ LOCK IN SHARE MODE",
        # Double-dash comments between restricted keyword pairs
        "SELECT 1 INTO -- bypass\nS3",
        "SELECT * FROM t FOR -- comment\nUPDATE",
        "SELECT 1 INTO --comment OUTFILE",
        # Hash comments between restricted keyword pairs
        "SELECT 1 INTO # bypass\nS3",
        "SELECT * FROM t FOR # comment\nUPDATE",
        "SELECT 1 INTO #comment OUTFILE",
        # Double-dash / hash mid-query
        "SELECT * FROM employees -- rest ignored",
        "SHOW TABLES # ignored rest",
        "SELECT id FROM t WHERE x=1 --trail",
    ],
)
def test_validate_query_rejects_sql_comments(query: str) -> None:
    """SQL comments can separate keywords and bypass pattern checks."""
    ok, message = _tool()._validate_query(query)
    assert ok is False
    assert "comments" in message
