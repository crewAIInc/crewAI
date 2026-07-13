import asyncio
import json
from unittest.mock import MagicMock

import pytest

from crewai_tools import (
    ChDBAttachFileTool,
    ChDBDescribeTableTool,
    ChDBGetSampleDataTool,
    ChDBListDatabasesTool,
    ChDBListFunctionsTool,
    ChDBListTablesTool,
    ChDBRunSelectQueryTool,
    chdb_tools,
)


OK_ENVELOPE = {
    "ok": True,
    "result": {"rows": [{"x": 1}], "row_count": 1, "truncated": False},
}
ERROR_ENVELOPE = {
    "ok": False,
    "error": {"code": 62, "type": "SYNTAX_ERROR", "message": "Syntax error"},
}


# --- unit tests against a mocked engine (no chdb install required) ----------


def _mock_engine(envelope=OK_ENVELOPE, read_only=True):
    engine = MagicMock()
    engine.read_only = read_only
    engine.call.return_value = envelope

    async def acall(name, arguments):
        return envelope

    engine.acall = acall
    return engine


def test_query_tool_dispatches_canonical_call():
    engine = _mock_engine()
    tool = ChDBRunSelectQueryTool(engine=engine)

    out = tool.run(sql="SELECT 1 AS x")

    engine.call.assert_called_once_with(
        "run_select_query", {"sql": "SELECT 1 AS x", "params": None}
    )
    assert json.loads(out) == OK_ENVELOPE


def test_query_tool_passes_params():
    engine = _mock_engine()
    tool = ChDBRunSelectQueryTool(engine=engine)

    tool.run(sql="SELECT {id:Int64}", params={"id": 42})

    engine.call.assert_called_once_with(
        "run_select_query", {"sql": "SELECT {id:Int64}", "params": {"id": 42}}
    )


def test_error_envelope_is_returned_not_raised():
    tool = ChDBRunSelectQueryTool(engine=_mock_engine(ERROR_ENVELOPE))

    out = json.loads(tool.run(sql="SELEC bad"))

    assert out["ok"] is False
    assert out["error"]["type"] == "SYNTAX_ERROR"


def test_arun_uses_async_engine_path():
    tool = ChDBRunSelectQueryTool(engine=_mock_engine())

    out = asyncio.run(tool.arun(sql="SELECT 1"))

    assert json.loads(out) == OK_ENVELOPE


@pytest.mark.parametrize(
    ("tool_class", "canonical_name", "kwargs", "expected_arguments"),
    [
        (ChDBListDatabasesTool, "list_databases", {}, {}),
        (ChDBListTablesTool, "list_tables", {"database": "db1"}, {"database": "db1"}),
        (
            ChDBDescribeTableTool,
            "describe_table",
            {"target": "events"},
            {"target": "events", "database": None},
        ),
        (
            ChDBGetSampleDataTool,
            "get_sample_data",
            {"target": "events", "limit": 3},
            {"target": "events", "database": None, "limit": 3},
        ),
        (
            ChDBListFunctionsTool,
            "list_functions",
            {"like": "%array%"},
            {"like": "%array%", "limit": None},
        ),
        (
            ChDBAttachFileTool,
            "attach_file",
            {"name": "t", "path": "/tmp/t.csv"},
            {"name": "t", "path": "/tmp/t.csv", "format": None},
        ),
    ],
)
def test_each_tool_dispatches_its_canonical_name(
    tool_class, canonical_name, kwargs, expected_arguments
):
    engine = _mock_engine()
    tool = tool_class(engine=engine)

    assert tool.name == canonical_name
    tool.run(**kwargs)
    engine.call.assert_called_once_with(canonical_name, expected_arguments)


def test_close_does_not_close_injected_engine():
    engine = _mock_engine()
    tool = ChDBRunSelectQueryTool(engine=engine)

    tool.close()

    engine.close.assert_not_called()


def test_suite_shares_injected_engine_and_gates_attach_file():
    read_only_suite = chdb_tools(engine=_mock_engine(read_only=True))
    writable_engine = _mock_engine(read_only=False)
    writable_suite = chdb_tools(engine=writable_engine)

    assert [t.name for t in read_only_suite] == [
        "run_select_query",
        "list_databases",
        "list_tables",
        "describe_table",
        "get_sample_data",
        "list_functions",
    ]
    assert [t.name for t in writable_suite][-1] == "attach_file"
    assert all(t.engine is writable_engine for t in writable_suite)


def test_import_error_message_mentions_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("chdb"):
            raise ImportError("No module named 'chdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tool = ChDBRunSelectQueryTool()

    with pytest.raises(ImportError, match=r"crewai-tools\[chdb\]"):
        tool.run(sql="SELECT 1")


# --- integration tests against the real engine -------------------------------

chdb = pytest.importorskip("chdb")


def test_end_to_end_select():
    tool = ChDBRunSelectQueryTool()
    try:
        out = json.loads(tool.run(sql="SELECT 1 + 1 AS answer"))
        assert out["ok"] is True
        assert out["result"]["rows"] == [{"answer": 2}]
    finally:
        tool.close()


def test_param_binding_end_to_end():
    tool = ChDBRunSelectQueryTool()
    try:
        out = json.loads(
            tool.run(sql="SELECT {n:Int64} * 2 AS doubled", params={"n": 21})
        )
        assert out["ok"] is True
        # 64-bit integers are serialized as strings so exact values survive JSON.
        assert out["result"]["rows"] == [{"doubled": "42"}]
    finally:
        tool.close()


def test_read_only_rejects_writes_with_envelope():
    tool = ChDBRunSelectQueryTool()
    try:
        out = json.loads(
            tool.run(sql="CREATE TABLE t (x Int64) ENGINE = MergeTree ORDER BY x")
        )
        assert out["ok"] is False
        assert "readonly" in out["error"]["message"].lower()
    finally:
        tool.close()


def test_writable_suite_attach_then_query(tmp_path):
    csv = tmp_path / "people.csv"
    csv.write_text("name,age\nada,36\ngrace,45\n")

    tools = {t.name: t for t in chdb_tools(read_only=False)}
    try:
        attached = json.loads(
            tools["attach_file"].run(name="people", path=str(csv))
        )
        assert attached["ok"] is True

        queried = json.loads(
            tools["run_select_query"].run(
                sql="SELECT max(age) AS oldest FROM people"
            )
        )
        assert queried["ok"] is True
        # 64-bit integers are serialized as strings so exact values survive JSON.
        assert queried["result"]["rows"] == [{"oldest": "45"}]
    finally:
        tools["run_select_query"].engine.close()


def test_read_only_suite_with_attachments(tmp_path):
    csv = tmp_path / "cities.csv"
    csv.write_text("city,pop\nparis,2100000\n")

    tools = chdb_tools(attachments={"cities": str(csv)})
    try:
        out = json.loads(tools[0].run(sql="SELECT city FROM cities"))
        assert out["ok"] is True
        assert out["result"]["rows"] == [{"city": "paris"}]
    finally:
        tools[0].engine.close()


def test_introspection_tools_end_to_end():
    engine_owner = ChDBListDatabasesTool()
    try:
        databases = json.loads(engine_owner.run())
        assert databases["ok"] is True

        shared = engine_owner._get_engine()
        described = json.loads(
            ChDBDescribeTableTool(engine=shared).run(
                target="numbers(3)",
            )
        )
        assert described["ok"] is True

        functions = json.loads(
            ChDBListFunctionsTool(engine=shared).run(like="%arrayJoin%")
        )
        assert functions["ok"] is True
    finally:
        engine_owner.close()
