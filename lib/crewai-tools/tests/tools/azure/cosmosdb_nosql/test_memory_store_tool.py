"""Tests for AzureCosmosDBMemoryTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


CONTAINER_PROPS = {"partition_key": {"paths": ["/agent_id"], "kind": "Hash"}}


@pytest.fixture
def patched_helpers():
    with (
        patch(
            "crewai_tools.azure.cosmosdb_nosql.memory_store.memory_store_tool.build_cosmos_client"
        ),
        patch(
            "crewai_tools.azure.cosmosdb_nosql.memory_store.memory_store_tool.create_database_if_not_exists"
        ) as make_db,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.memory_store.memory_store_tool.create_container_if_not_exists"
        ) as make_container,
    ):
        container = MagicMock()
        make_container.return_value = container
        make_db.return_value.get_container_client.return_value = container
        yield {"container": container}


def _build_tool(**config_overrides):
    from crewai_tools.azure.cosmosdb_nosql.memory_store.memory_store_tool import (
        AzureCosmosDBMemoryConfig,
        AzureCosmosDBMemoryTool,
    )

    cfg = dict(
        cosmos_host="https://example.documents.azure.com:443/",
        key="dummy-key",
        cosmos_container_properties=CONTAINER_PROPS,
    )
    cfg.update(config_overrides)
    return AzureCosmosDBMemoryTool(config=AzureCosmosDBMemoryConfig(**cfg))


def test_unknown_operation_returns_valid_operations(patched_helpers):
    tool = _build_tool()
    payload = json.loads(tool._run(operation="bogus"))
    assert payload["error"].startswith("Unknown operation")
    assert set(payload["valid_operations"]) == {
        "store",
        "read",
        "retrieve",
        "update",
        "delete",
        "clear",
    }


def test_store_requires_memory_item(patched_helpers):
    tool = _build_tool()
    payload = json.loads(tool._run(operation="store"))
    assert payload == {"error": "memory_item is required for store operation"}


def test_store_persists_item_with_ttl(patched_helpers):
    tool = _build_tool()
    patched_helpers["container"].create_item.return_value = {
        "id": "m-1",
        "agent_id": "a-1",
    }
    payload = json.loads(
        tool._run(
            operation="store",
            memory_item={"id": "m-1", "agent_id": "a-1", "content": {"text": "hi"}},
            ttl=60,
        )
    )
    body = patched_helpers["container"].create_item.call_args.kwargs["body"]
    assert body["ttl"] == 60
    assert payload["id"] == "m-1"


def test_read_requires_partition_key_and_id(patched_helpers):
    tool = _build_tool()
    assert json.loads(tool._run(operation="read")) == {
        "error": "partition_key_value is required for read operation"
    }
    assert json.loads(tool._run(operation="read", partition_key_value="a-1")) == {
        "error": "memory_id is required for read operation"
    }


def test_retrieve_builds_partition_filter_with_query_filter(patched_helpers):
    tool = _build_tool()
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": "m-1", "agent_id": "a-1"}]
    )

    payload = json.loads(
        tool._run(
            operation="retrieve",
            partition_key_value="a-1",
            query_filter={"category": "facts"},
            max_results=3,
        )
    )

    sql = patched_helpers["container"].query_items.call_args.kwargs["query"]
    assert "TOP 3" in sql
    assert "c.agent_id = 'a-1'" in sql
    assert "c.content.category = 'facts'" in sql
    assert payload == [{"id": "m-1", "agent_id": "a-1"}]


def test_retrieve_hierarchical_partition_value_count_mismatch(patched_helpers):
    tool = _build_tool(
        cosmos_container_properties={
            "partition_key": {
                "paths": ["/tenant_id", "/agent_id"],
                "kind": "MultiHash",
            }
        }
    )
    payload = json.loads(
        tool._run(operation="retrieve", partition_key_value=["a-1"])
    )
    assert "partition key levels" in payload["error"]


def test_clear_uses_batch_delete_tuple_format(patched_helpers):
    tool = _build_tool()
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": "m-1"}, {"id": "m-2"}]
    )

    json.loads(tool._run(operation="clear", partition_key_value="a-1"))

    batch_kwargs = patched_helpers["container"].execute_item_batch.call_args.kwargs
    assert batch_kwargs["partition_key"] == "a-1"
    assert batch_kwargs["batch_operations"] == [
        ("delete", ("m-1",)),
        ("delete", ("m-2",)),
    ]


def test_retrieve_query_filter_escapes_single_quotes(patched_helpers):
    """Single quotes in filter values must be SQL-escaped."""
    tool = _build_tool()
    patched_helpers["container"].query_items.return_value = iter([])

    tool._run(
        operation="retrieve",
        partition_key_value="a-1",
        query_filter={"author": "O'Brien"},
    )
    sql = patched_helpers["container"].query_items.call_args.kwargs["query"]
    assert "c.content.author = 'O''Brien'" in sql


def test_retrieve_query_filter_rejects_bad_key(patched_helpers):
    tool = _build_tool()
    payload = json.loads(
        tool._run(
            operation="retrieve",
            partition_key_value="a-1",
            query_filter={"bad key; DROP TABLE c": "x"},
        )
    )
    assert "error" in payload


def test_partition_key_value_with_quote_is_escaped(patched_helpers):
    tool = _build_tool()
    patched_helpers["container"].query_items.return_value = iter([])
    tool._run(operation="retrieve", partition_key_value="agent's-1")
    sql = patched_helpers["container"].query_items.call_args.kwargs["query"]
    assert "c.agent_id = 'agent''s-1'" in sql


# ---------------------------------------------------------------------------
# Review-fix regressions
# ---------------------------------------------------------------------------


def test_args_schema_declares_all_operands():
    """args_schema must expose every field _run needs, or the agent path drops them."""
    from crewai_tools.azure.cosmosdb_nosql.memory_store.memory_store_tool import (
        AzureCosmosDBMemoryToolSchema,
    )

    fields = set(AzureCosmosDBMemoryToolSchema.model_fields)
    assert {
        "operation",
        "memory_item",
        "partition_key_value",
        "memory_id",
        "query_filter",
        "max_results",
        "ttl",
    } <= fields


def test_store_stamps_created_at(patched_helpers):
    """Stored items must get a created_at so retrieve's ORDER BY includes them."""
    tool = _build_tool()
    patched_helpers["container"].create_item.return_value = {"id": "m-1"}
    tool._run(
        operation="store",
        memory_item={"id": "m-1", "agent_id": "a-1", "content": {"text": "hi"}},
    )
    body = patched_helpers["container"].create_item.call_args.kwargs["body"]
    assert "created_at" in body


def test_update_merges_into_existing(patched_helpers):
    """update must preserve fields the caller did not resend."""
    tool = _build_tool()
    patched_helpers["container"].read_item.return_value = {
        "id": "m-1",
        "agent_id": "a-1",
        "created_at": "2024-01-01T00:00:00+00:00",
        "content": {"text": "old"},
        "keep": "me",
    }
    patched_helpers["container"].replace_item.return_value = {"id": "m-1"}

    tool._run(
        operation="update",
        partition_key_value="a-1",
        memory_id="m-1",
        memory_item={"content": {"text": "new"}},
    )

    body = patched_helpers["container"].replace_item.call_args.kwargs["body"]
    assert body["content"] == {"text": "new"}      # updated
    assert body["created_at"] == "2024-01-01T00:00:00+00:00"  # preserved
    assert body["keep"] == "me"                    # preserved
    assert body["id"] == "m-1"


def test_clear_reports_partial_count_on_failure(patched_helpers):
    """A mid-loop batch failure must still report how many were deleted."""
    tool = _build_tool()
    # 150 ids -> two batches (100 + 50); make the first batch delete fail.
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": f"m-{i}"} for i in range(150)]
    )
    patched_helpers["container"].execute_item_batch.side_effect = RuntimeError("boom")

    payload = json.loads(tool._run(operation="clear", partition_key_value="a-1"))
    assert "error" in payload
    assert payload["deleted_count"] == 0


def test_clear_streams_and_counts(patched_helpers):
    tool = _build_tool()
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": f"m-{i}"} for i in range(150)]
    )
    payload = json.loads(tool._run(operation="clear", partition_key_value="a-1"))
    assert payload["success"] is True
    assert payload["deleted_count"] == 150
    assert patched_helpers["container"].execute_item_batch.call_count == 2
