"""Tests for TopKSearchTool.

The unit tests stub the topk_sdk query builder and client, so they run without
topk-sdk installed or network access. The live test at the bottom runs only when
TOPK_API_KEY is set.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
import uuid

import pytest

from crewai_tools import TopKSearchTool
from crewai_tools.tools.topk_search_tool.topk_search_tool import TopKToolSchema


ROWS = [
    {"_id": "e4021", "text": "Error code E-4021 means the payment gateway timed out.", "topk_score": 2.5},
    {"_id": "e4012", "text": "Error code E-4012 means the card was declined.", "topk_score": 1.0},
]


def make_tool(rows=None, **kwargs):
    sdk = MagicMock()
    client = MagicMock()
    client.collection.return_value.query.return_value = rows if rows is not None else ROWS
    tool = TopKSearchTool(collection_name="runbook", client=client, **kwargs)
    tool._topk = sdk
    return tool, sdk, client


def test_schema_tells_the_agent_when_to_use_each_mode():
    props = TopKToolSchema.model_json_schema()["properties"]
    assert props["search_mode"]["enum"] == ["hybrid", "keyword", "semantic"]
    assert "error codes" in props["search_mode"]["description"]
    assert "$gte" in props["filters"]["description"]


def test_keyword_search_uses_bm25_with_a_match_filter():
    tool, sdk, client = make_tool()
    out = json.loads(tool._run("E-4021", search_mode="keyword"))
    sdk.query.match.assert_called_once_with("E-4021", field="text")
    sdk.query.fn.bm25_score.assert_called_once()
    assert out[0] == {"id": "e4021", "score": 2.5, "text": ROWS[0]["text"]}
    client.collection.assert_called_once_with("runbook", partition=None)


def test_semantic_search_has_no_text_filter():
    tool, sdk, _ = make_tool()
    tool._run("why did checkout fail?", search_mode="semantic")
    sdk.query.fn.semantic_similarity.assert_called_once_with("text", "why did checkout fail?")
    sdk.query.match.assert_not_called()
    sdk.query.should.assert_not_called()


def test_hybrid_search_combines_scores_and_keeps_non_matches():
    rows = [{"_id": "a", "text": "x", "topk_dense": 1.0, "topk_keyword": 3.0}]
    tool, sdk, _ = make_tool(rows=rows, hybrid_alpha=0.25)
    out = json.loads(tool._run("checkout E-4021"))
    sdk.query.should.assert_called_once_with("checkout E-4021", field="text")
    sdk.query.match.assert_not_called()
    assert out[0]["score"] == pytest.approx(1.0 * 0.25 + 3.0 * 0.75)


def test_partition_and_limit():
    tool, sdk, client = make_tool(partition="user-42", limit=7)
    tool._run("x")
    client.collection.assert_called_once_with("runbook", partition="user-42")
    select = sdk.query.select.return_value
    select.filter.return_value.sort.return_value.limit.assert_called_once_with(7)


def test_return_fields_are_selected_and_returned():
    rows = [{"_id": "a", "text": "t", "source": "faq", "topk_score": 1.0}]
    tool, sdk, _ = make_tool(rows=rows, return_fields=["text", "source"])
    out = json.loads(tool._run("x", search_mode="semantic"))
    assert sdk.query.select.call_args.args == ("text", "source")
    assert out[0]["source"] == "faq"


def test_filters_translate_to_topk_expressions():
    tool, sdk, _ = make_tool()
    field = sdk.query.field
    tool._run(
        "x",
        search_mode="semantic",
        filters={"source": "runbook", "year": {"$gte": 2024}, "tag": {"$nin": ["a"]}},
    )
    field.return_value.eq.assert_called_with("runbook")
    field.return_value.gte.assert_called_with(2024)
    field.return_value.in_.assert_called_with(["a"])
    sdk.query.not_.assert_called_once()
    assert len(sdk.query.all.call_args.args[0]) == 3


def test_empty_partition_returns_no_results():
    class PartitionNotFoundError(Exception):
        pass

    tool, _, client = make_tool(partition="new-user")
    client.collection.return_value.query.side_effect = PartitionNotFoundError("new-user")
    assert tool._run("x") == "[]"
    client.collection.return_value.query.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        tool._run("x")


def test_unknown_filter_operator_is_rejected():
    tool, _, _ = make_tool()
    with pytest.raises(ValueError, match="Unknown filter operator"):
        tool._run("x", filters={"year": {"$regex": "20.*"}})


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    monkeypatch.delenv("TOPK_REGION", raising=False)
    tool = TopKSearchTool(collection_name="runbook")
    tool._topk = MagicMock()
    with pytest.raises(ValueError, match="TOPK_API_KEY"):
        tool._run("x")


def test_missing_sdk_has_a_helpful_error():
    tool = TopKSearchTool(collection_name="runbook")
    with patch("importlib.import_module", side_effect=ImportError("no topk_sdk")):
        with pytest.raises(ImportError, match="pip install topk-sdk"):
            tool._run("x")


@pytest.mark.skipif(not os.getenv("TOPK_API_KEY"), reason="needs TOPK_API_KEY and TOPK_REGION")
def test_live_topk():
    topk_sdk = pytest.importorskip("topk_sdk")
    from topk_sdk.schema import semantic_index, text

    client = topk_sdk.Client(api_key=os.environ["TOPK_API_KEY"], region=os.environ["TOPK_REGION"])
    name = f"crewai_test_{uuid.uuid4().hex[:10]}"
    client.collections().create(name, schema={"text": text().required().index(semantic_index())})
    try:
        docs = [
            ("e4021", "Error code E-4021 means the payment gateway timed out.", "runbook", 2024),
            ("e4012", "Error code E-4012 means the card was declined by the bank.", "runbook", 2024),
            ("churn", "Customers cancel subscriptions mostly because onboarding takes too long.", "notes", 2025),
        ]
        rows = [{"_id": i, "text": t, "source": s, "year": y} for i, t, s, y in docs]
        lsn = client.collection(name).upsert(rows)
        client.collection(name).count(lsn=lsn)  # wait until the writes are readable

        tool = TopKSearchTool(collection_name=name, client=client, return_fields=["text", "source"])
        keyword = json.loads(tool.run(query="E-4021", search_mode="keyword"))
        assert [r["id"] for r in keyword] == ["e4021"]
        semantic = json.loads(tool.run(query="why do people stop paying?", search_mode="semantic"))
        assert semantic[0]["id"] == "churn"
        hybrid = json.loads(tool.run(query="E-4021 checkout", filters={"source": "runbook"}))
        assert hybrid[0]["id"] == "e4021"
        assert {r["source"] for r in hybrid} == {"runbook"}
        assert json.loads(tool.run(query="E-9999", search_mode="keyword")) == []

        lsn = client.collection(name, partition="user-alice").upsert([{"_id": "n1", "text": "Alice prefers dark mode."}])
        client.collection(name, partition="user-alice").count(lsn=lsn)
        alice = TopKSearchTool(collection_name=name, client=client, partition="user-alice")
        assert [r["id"] for r in json.loads(alice.run(query="display preference"))] == ["n1"]
        bob = TopKSearchTool(collection_name=name, client=client, partition="user-bob")
        assert json.loads(bob.run(query="display preference")) == []
    finally:
        client.collections().delete(name)
