"""Tests for AzureCosmosDBSemanticCacheTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    "fullTextIndexes": [{"path": "/prompt"}],
}
EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "dimensions": 4,
            "distanceFunction": "cosine",
        }
    ]
}
FULL_TEXT_POLICY = {"fullTextPaths": [{"path": "/prompt", "language": "en-US"}]}
CONTAINER_PROPS = {"partition_key": {"paths": ["/agent_id"], "kind": "Hash"}}


def _embed_response(vec):
    item = MagicMock()
    item.embedding = vec
    response = MagicMock()
    response.data = [item]
    return response


@pytest.fixture
def patched_helpers():
    with (
        patch(
            "crewai_tools.azure.cosmosdb_nosql.semantic_cache.semantic_cache_tool.build_cosmos_client"
        ),
        patch(
            "crewai_tools.azure.cosmosdb_nosql.semantic_cache.semantic_cache_tool.create_database_if_not_exists"
        ) as make_db,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.semantic_cache.semantic_cache_tool.create_container_if_not_exists"
        ) as make_container,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.semantic_cache.semantic_cache_tool.build_openai_client"
        ) as openai_client,
    ):
        container = MagicMock()
        make_container.return_value = container
        make_db.return_value.get_container_client.return_value = container
        yield {"openai": openai_client.return_value, "container": container}


def _build_tool(**overrides):
    from crewai_tools.azure.cosmosdb_nosql.semantic_cache.semantic_cache_tool import (
        AzureCosmosDBSemanticCacheConfig,
        AzureCosmosDBSemanticCacheTool,
    )

    cfg_kwargs = dict(
        cosmos_host="https://example.documents.azure.com:443/",
        key="dummy-key",
        openai_api_key="dummy-openai-key",
        cosmos_container_properties=CONTAINER_PROPS,
        indexing_policy=INDEXING_POLICY,
        vector_embedding_policy=EMBEDDING_POLICY,
        full_text_policy=FULL_TEXT_POLICY,
        embedding_dimensions=4,
        dimensions=4,
    )
    cfg_kwargs.update(overrides)
    config = AzureCosmosDBSemanticCacheConfig(**cfg_kwargs)
    return AzureCosmosDBSemanticCacheTool(config=config)


def test_init_validates_vector_indexes(patched_helpers):
    with pytest.raises(ValueError, match="vectorIndexes"):
        _build_tool(indexing_policy={"vectorIndexes": []})


def test_hybrid_requires_full_text_policy(patched_helpers):
    with pytest.raises(ValueError, match="fullTextIndexes|fullTextPaths"):
        _build_tool(
            indexing_policy={
                "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}]
            }
        )


def test_unknown_operation_returns_error_payload(patched_helpers):
    tool = _build_tool()
    payload = json.loads(tool._run(operation="bogus"))
    assert payload["error"].startswith("Unknown operation")
    assert "search" in payload["valid_operations"]


def test_search_requires_prompt(patched_helpers):
    tool = _build_tool()
    payload = json.loads(tool._run(operation="search"))
    assert payload == {"error": "prompt is required for search operation"}


def test_search_returns_cached_response_above_threshold(patched_helpers):
    tool = _build_tool(similarity_threshold=0.5)
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    # Cosmos VectorDistance for cosine returns a similarity-like score
    # (higher = better). 0.95 is comfortably above threshold 0.5.
    patched_helpers["container"].query_items.return_value = iter(
        [
            {
                "id": "cache-1",
                "prompt": "what is crewAI",
                "response": "a multi-agent framework",
                "similarity_score": 0.95,
            }
        ]
    )

    payload = json.loads(tool._run(operation="search", prompt="what is crewAI?"))

    assert payload["cache_hit"] is True
    assert payload["cached_response"] == "a multi-agent framework"


def test_update_persists_document_with_partition_key_field(patched_helpers):
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].upsert_item.return_value = {"id": "cache-x"}

    json.loads(tool._run(operation="update", prompt="hi", response="hello"))

    upsert_call = patched_helpers["container"].upsert_item.call_args
    body = upsert_call.kwargs["body"]
    assert body["prompt"] == "hi"
    assert body["response"] == "hello"
    # partition key field should be filled in by setdefault
    assert "agent_id" in body


def test_search_filters_by_llm_namespace(patched_helpers):
    """Different llm_string values must produce different cache namespaces."""
    tool_a = _build_tool(llm_string="gpt-4o")
    tool_b = _build_tool(llm_string="claude-opus-4")
    assert tool_a._llm_namespace != tool_b._llm_namespace


def test_search_quote_escapes_fts_terms(patched_helpers):
    """Single quotes inside the prompt must not break out of the FTS literal."""
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter([])

    tool._run(operation="search", prompt="O'Brien")

    call = patched_helpers["container"].query_items.call_args
    sql = call.kwargs.get("query") or call.args[0]
    # Apostrophe must be doubled; raw single quote must NOT appear unescaped
    # adjacent to the token.
    assert "'O''Brien'" in sql


def test_search_uses_explicit_projection_not_star(patched_helpers):
    """Cosmos rejects 'SELECT *, <expr>'; the cache must project explicit fields.

    Covers both the hybrid (RRF) and vector-only query paths.
    """
    for enable_hybrid in (True, False):
        tool = _build_tool(enable_hybrid_search=enable_hybrid)
        patched_helpers["openai"].embeddings.create.return_value = _embed_response(
            [0.1, 0.2, 0.3, 0.4]
        )
        patched_helpers["container"].query_items.return_value = iter([])

        tool._run(operation="search", prompt="hello world")

        call = patched_helpers["container"].query_items.call_args
        sql = call.kwargs.get("query") or call.args[0]
        assert "*" not in sql, f"SELECT * leaked into SQL (hybrid={enable_hybrid})"
        assert "c.response" in sql
        assert "similarity_score" in sql


def test_clear_cache_deletes_items_not_database(patched_helpers):
    """_clear_cache must delete matching rows, not the whole database."""
    tool = _build_tool(llm_string="gpt-4o")
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": "a", "_pk": "agent1"}, {"id": "b", "_pk": "agent1"}]
    )

    payload = json.loads(tool._run(operation="clear"))

    assert payload["success"] is True
    assert payload["deleted_count"] == 2
    assert patched_helpers["container"].delete_item.call_count == 2


def test_update_uses_deterministic_id(patched_helpers):
    """Re-caching the same prompt+namespace must reuse the same document id."""
    tool = _build_tool(llm_string="gpt-4o")
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].upsert_item.return_value = {"id": "x"}

    tool._run(operation="update", prompt="hello", response="a")
    id_1 = patched_helpers["container"].upsert_item.call_args.kwargs["body"]["id"]
    tool._run(operation="update", prompt="hello", response="b")
    id_2 = patched_helpers["container"].upsert_item.call_args.kwargs["body"]["id"]

    assert id_1 == id_2  # same prompt -> same id -> upsert replaces

    tool._run(operation="update", prompt="different", response="c")
    id_3 = patched_helpers["container"].upsert_item.call_args.kwargs["body"]["id"]
    assert id_3 != id_1  # different prompt -> different id


def test_nested_partition_path_rejected_at_init(patched_helpers):
    """Nested partition paths would be written as a flat 'metadata/agent_id'
    field; the cache must reject them at construction time."""
    with pytest.raises(ValueError, match="Nested partition key"):
        _build_tool(
            cosmos_container_properties={
                "partition_key": {"paths": ["/metadata/agent_id"], "kind": "Hash"}
            }
        )


def test_clear_reports_failed_deletions(patched_helpers):
    """_clear_cache must surface failed deletions instead of always success."""
    tool = _build_tool(llm_string="gpt-4o")
    patched_helpers["container"].query_items.return_value = iter(
        [{"id": "a", "_pk": "p"}, {"id": "b", "_pk": None}]
    )
    # Second delete (missing _pk) raises; first succeeds.
    patched_helpers["container"].delete_item.side_effect = [None, RuntimeError("boom")]

    payload = json.loads(tool._run(operation="clear"))
    assert payload["deleted_count"] == 1
    assert payload["failed_count"] == 1
    assert payload["success"] is False
