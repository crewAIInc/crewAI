from __future__ import annotations

import json

import pytest

from crewai_tools import (
    OracleHybridSearchTool,
    OracleTextSearchTool,
    OracleVectorSearchTool,
)
from tests.tools.oracle_db.conftest import CursorStub

pytest_plugins = ("tests.tools.oracle_db.conftest",)


def test_oracle_text_search_tool_formats_contains_query(oracle_connection_mock):
    cursor = CursorStub()

    def execute_side_effect(cursor_obj, sql, *args, **kwargs):
        assert "CONTAINS(text, :query, 1)" in sql
        assert 'fuzzy("oracle") ACCUM fuzzy("database")' == kwargs["query"]
        cursor_obj.description = [("SCORE",), ("TEXT",), ("CATEGORY",)]
        cursor_obj._fetchall_result = [(92, "Oracle Database text result", "docs")]

    cursor.execute_side_effect = execute_side_effect
    oracle_connection_mock.cursor.return_value = cursor

    tool = OracleTextSearchTool(
        client=oracle_connection_mock,
        table_name="docs_table",
        text_column="text",
        metadata_columns=["category"],
        fuzzy=True,
    )

    result = json.loads(tool._run("oracle database"))

    assert result["results"][0]["content"] == "Oracle Database text result"
    assert result["results"][0]["metadata"]["category"] == "docs"
    assert result["results"][0]["score"] == 92


def test_oracle_text_search_tool_validates_identifiers():
    tool = OracleTextSearchTool(
        client=object(),
        table_name="docs-table",
        text_column="text",
    )

    with pytest.raises(ValueError, match="table_name"):
        tool._run("oracle")


def test_oracle_hybrid_search_tool_builds_search_params_and_fetches_rows(
    oracle_connection_mock,
):
    cursor = CursorStub()
    state = {"search_called": False}

    def execute_side_effect(cursor_obj, sql, *args, **kwargs):
        if "DBMS_HYBRID_VECTOR.SEARCH" in sql:
            state["search_called"] = True
            search_params = json.loads(kwargs["search_params"])
            assert search_params["hybrid_index_name"] == "docsidx"
            assert search_params["vector"]["search_text"] == "autonomous database"
            assert search_params["text"]["search_text"] == "autonomous database"
            cursor_obj._fetchall_result = [
                (
                    '[{"rowid":"AAABBB","score":0.97,"vector_score":0.95,"text_score":0.88}]',
                )
            ]
            return None

        assert state["search_called"]
        assert "WHERE rowid = :1" in sql
        cursor_obj._fetchone_result = (
            "Autonomous Database is managed.",
            {"source": "doc-1"},
            "oracle",
        )

    cursor.execute_side_effect = execute_side_effect
    oracle_connection_mock.cursor.return_value = cursor

    tool = OracleHybridSearchTool(
        client=oracle_connection_mock,
        hybrid_index_name="docsidx",
        table_name="docs_table",
        metadata_columns=["category"],
    )

    result = json.loads(tool._run("autonomous database"))

    assert result["results"][0]["content"] == "Autonomous Database is managed."
    assert result["results"][0]["metadata"]["source"] == "doc-1"
    assert result["results"][0]["metadata"]["category"] == "oracle"
    assert result["results"][0]["score"] == 0.97


def test_oracle_hybrid_search_tool_rejects_reserved_params():
    tool = OracleHybridSearchTool(
        client=object(),
        hybrid_index_name="docsidx",
        table_name="docs_table",
        params={"return": {"topN": 1}},
    )

    with pytest.raises(ValueError, match="Reserved hybrid search params"):
        tool._build_search_params("oracle")


def test_oracle_vector_search_tool_embeds_query_and_fetches_rows(oracle_connection_mock):
    cursor = CursorStub()

    def execute_side_effect(cursor_obj, sql, *args, **kwargs):
        assert "VECTOR_DISTANCE(embedding, :query_embedding, COSINE)" in sql
        query_embedding = kwargs["query_embedding"]
        assert list(query_embedding) == pytest.approx([0.1, 0.2, 0.3])
        cursor_obj.description = [("TEXT",), ("METADATA",), ("CATEGORY",), ("DISTANCE",)]
        cursor_obj._fetchall_result = [
            ("Oracle vector result", {"source": "doc-1"}, "docs", 0.0123)
        ]

    cursor.execute_side_effect = execute_side_effect
    oracle_connection_mock.cursor.return_value = cursor

    tool = OracleVectorSearchTool(
        client=oracle_connection_mock,
        table_name="docs_table",
        metadata_columns=["category"],
        embedder=lambda inputs: [[0.1, 0.2, 0.3] for _ in inputs],
    )

    result = json.loads(tool._run("oracle vector database"))

    assert result["results"][0]["content"] == "Oracle vector result"
    assert result["results"][0]["metadata"]["source"] == "doc-1"
    assert result["results"][0]["metadata"]["category"] == "docs"
    assert result["results"][0]["distance"] == pytest.approx(0.0123)
