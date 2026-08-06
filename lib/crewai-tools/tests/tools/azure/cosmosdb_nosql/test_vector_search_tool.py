"""Tests for AzureCosmosDBNoSqlSearchTool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# Reasonable defaults reused across tests --------------------------------------

INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
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
CONTAINER_PROPS = {"partition_key": {"paths": ["/pk"], "kind": "Hash"}}


def _embed_response(vec):
    item = MagicMock()
    item.embedding = vec
    response = MagicMock()
    response.data = [item]
    return response


@pytest.fixture
def patched_helpers():
    """Patch the shared client/embedding helpers and yield references."""
    with (
        patch(
            "crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool.build_cosmos_client"
        ) as cosmos,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool.create_database_if_not_exists"
        ) as make_db,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool.create_container_if_not_exists"
        ) as make_container,
        patch(
            "crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool.build_openai_client"
        ) as openai_client,
    ):
        container = MagicMock()
        make_container.return_value = container
        make_db.return_value.get_container_client.return_value = container
        yield {
            "cosmos": cosmos,
            "openai": openai_client.return_value,
            "container": container,
        }


def _build_tool(**overrides):
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchTool,
    )

    defaults = dict(
        cosmos_host="https://example.documents.azure.com:443/",
        key="dummy-key",
        openai_api_key="dummy-openai-key",
        indexing_policy=INDEXING_POLICY,
        vector_embedding_policy=EMBEDDING_POLICY,
        cosmos_container_properties=CONTAINER_PROPS,
        dimensions=4,
    )
    defaults.update(overrides)
    return AzureCosmosDBNoSqlSearchTool(**defaults)


def test_init_creates_database_and_container(patched_helpers):
    tool = _build_tool()
    assert tool._container is patched_helpers["container"]


def test_invalid_search_type_rejected(patched_helpers):
    with pytest.raises(ValueError, match="Invalid search_type"):
        _build_tool(search_type="not-a-real-type")


def test_missing_partition_key_rejected(patched_helpers):
    with pytest.raises(ValueError, match="partition_key"):
        _build_tool(cosmos_container_properties={})


def test_full_text_requires_indexes_when_enabled(patched_helpers):
    with pytest.raises(ValueError, match="fullTextIndexes"):
        _build_tool(full_text_search_enabled=True)


def test_run_vector_search_returns_filtered_results(patched_helpers):
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter(
        [
            {"id": "doc-1", "description": "high", "SimilarityScore": 0.9},
            {"id": "doc-2", "description": "low", "SimilarityScore": 0.0},
        ]
    )
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchConfig,
    )

    tool.query_config = AzureCosmosDBNoSqlSearchConfig(threshold=0.1, max_results=5)

    output = json.loads(tool._run("hello"))

    assert [item["id"] for item in output] == ["doc-1"]
    call_kwargs = patched_helpers["container"].query_items.call_args.kwargs
    assert "VectorDistance" in call_kwargs["query"]


def test_add_texts_without_metadata_partition_key_raises(patched_helpers):
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    with pytest.raises(ValueError, match="Partition key 'pk'"):
        tool.add_texts(["hello"])


def test_add_texts_wraps_batch_body_in_tuple(patched_helpers):
    """execute_item_batch requires (op, (doc,)); a bare dict body is rejected by
    the SDK's _format_batch_operations. Use 'upsert' so retries are idempotent."""
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )

    ids = tool.add_texts(["hello"], metadatas=[{"pk": "tenant-1"}])

    assert len(ids) == 1
    ops = patched_helpers["container"].execute_item_batch.call_args.kwargs[
        "batch_operations"
    ]
    assert ops, "expected at least one batch operation"
    for op in ops:
        assert op[0] == "upsert"
        assert isinstance(op[1], tuple) and len(op[1]) == 1
        assert isinstance(op[1][0], dict)


def test_add_texts_writes_partition_value_at_configured_field(patched_helpers):
    """The partition value must land on the container's partition path, not 'pk'."""
    tool = _build_tool(
        cosmos_container_properties={
            "partition_key": {"paths": ["/agent_id"], "kind": "Hash"}
        }
    )
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )

    tool.add_texts(["hello"], metadatas=[{"agent_id": "a-1"}])

    call = patched_helpers["container"].execute_item_batch.call_args
    doc = call.kwargs["batch_operations"][0][1][0]
    assert doc["agent_id"] == "a-1"
    assert "pk" not in doc
    assert call.kwargs["partition_key"] == "a-1"


def test_add_texts_rejects_hierarchical_partition_key(patched_helpers):
    tool = _build_tool(
        cosmos_container_properties={
            "partition_key": {"paths": ["/tenant", "/agent_id"], "kind": "MultiHash"}
        }
    )
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    with pytest.raises(ValueError, match="hierarchical"):
        tool.add_texts(["hello"], metadatas=[{"tenant": "t", "agent_id": "a"}])


def test_add_texts_rejects_length_mismatch(patched_helpers):
    tool = _build_tool()
    with pytest.raises(ValueError, match="metadatas length"):
        tool.add_texts(["a", "b"], metadatas=[{"pk": "x"}])


def test_full_text_search_uses_query_as_predicate(patched_helpers):
    """full_text_search must turn the query into a FullTextContains predicate."""
    fts_indexing = {**INDEXING_POLICY, "fullTextIndexes": [{"path": "/text"}]}
    tool = _build_tool(
        full_text_search_enabled=True,
        search_type="full_text_search",
        indexing_policy=fts_indexing,
        full_text_policy={"fullTextPaths": [{"path": "/text", "language": "en-US"}]},
    )
    patched_helpers["container"].query_items.return_value = iter([])

    tool._run("quick brown fox")

    sql = patched_helpers["container"].query_items.call_args.kwargs["query"]
    assert "FullTextContainsAll(c.text, 'quick', 'brown', 'fox')" in sql
    patched_helpers["openai"].embeddings.create.assert_not_called()


def test_max_results_none_omits_top_clause(patched_helpers):
    """max_results=None must not emit 'TOP None' or a null @top parameter."""
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchConfig,
    )

    tool = _build_tool()
    tool.query_config = AzureCosmosDBNoSqlSearchConfig(max_results=None)
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter([])

    tool._run("hi")

    call = patched_helpers["container"].query_items.call_args
    sql = call.kwargs["query"]
    assert "TOP None" not in sql
    assert "TOP " not in sql
    names = [p["name"] for p in call.kwargs["parameters"]]
    assert "@top" not in names


def test_plain_vector_euclidean_keeps_results_without_threshold(patched_helpers):
    """Plain 'vector' + euclidean must not drop every result when no threshold set."""
    euclidean_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "dimensions": 4,
                "distanceFunction": "euclidean",
            }
        ]
    }
    tool = _build_tool(vector_embedding_policy=euclidean_policy)
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter(
        [
            {"id": "near", "description": "x", "SimilarityScore": 0.05},
            {"id": "far", "description": "y", "SimilarityScore": 5.00},
        ]
    )
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchConfig,
    )

    tool.query_config = AzureCosmosDBNoSqlSearchConfig(max_results=10)
    output = json.loads(tool._run("hi"))
    assert [item["id"] for item in output] == ["near", "far"]


def test_run_swallows_exceptions_into_error_payload(patched_helpers):
    tool = _build_tool()
    patched_helpers["openai"].embeddings.create.side_effect = RuntimeError("boom")
    output = json.loads(tool._run("hi"))
    assert output == {"error": "boom"}


def test_init_validates_text_key_against_injection(patched_helpers):
    """Identifier fields must reject characters that could break out of SQL."""
    with pytest.raises(ValueError):
        _build_tool(text_key="text; DROP TABLE c")


def test_full_text_terms_are_quote_escaped(patched_helpers):
    """Single quotes in FTS search terms must be doubled in the SQL emitted."""
    fts_indexing = {
        **INDEXING_POLICY,
        "fullTextIndexes": [{"path": "/text"}],
    }
    tool = _build_tool(
        full_text_search_enabled=True,
        search_type="full_text_ranking",
        indexing_policy=fts_indexing,
        full_text_policy={"fullTextPaths": [{"path": "/text", "language": "en-US"}]},
    )
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchConfig,
    )

    tool.query_config = AzureCosmosDBNoSqlSearchConfig(
        full_text_rank_filter=[{"search_field": "text", "search_text": "O'Brien"}],
        max_results=5,
        threshold=0.0,
    )
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter([])
    tool._run("anything")
    sql = patched_helpers["container"].query_items.call_args.kwargs["query"]
    assert "'O''Brien'" in sql


def test_full_text_search_does_not_require_embedder(patched_helpers):
    """Pure full-text search must not call the embedder / require an OpenAI key."""
    fts_indexing = {
        **INDEXING_POLICY,
        "fullTextIndexes": [{"path": "/text"}],
    }
    tool = _build_tool(
        full_text_search_enabled=True,
        search_type="full_text_search",
        indexing_policy=fts_indexing,
        full_text_policy={"fullTextPaths": [{"path": "/text", "language": "en-US"}]},
    )
    patched_helpers["container"].query_items.return_value = iter([])

    tool._run("some free text query")

    patched_helpers["openai"].embeddings.create.assert_not_called()


def test_distance_aware_threshold_for_euclidean(patched_helpers):
    """Euclidean: lower distance is better; threshold filters by distance."""
    euclidean_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "dimensions": 4,
                "distanceFunction": "euclidean",
            }
        ]
    }
    tool = _build_tool(
        vector_embedding_policy=euclidean_policy,
        search_type="vector_score_threshold",
    )
    patched_helpers["openai"].embeddings.create.return_value = _embed_response(
        [0.1, 0.2, 0.3, 0.4]
    )
    patched_helpers["container"].query_items.return_value = iter(
        [
            {"id": "near", "description": "x", "SimilarityScore": 0.05},
            {"id": "far",  "description": "y", "SimilarityScore": 5.00},
        ]
    )
    from crewai_tools.azure.cosmosdb_nosql.vector_search.vector_search_tool import (
        AzureCosmosDBNoSqlSearchConfig,
    )

    # For euclidean, threshold means MAX allowed distance.
    tool.query_config = AzureCosmosDBNoSqlSearchConfig(threshold=1.0, max_results=10)
    output = json.loads(tool._run("hi"))
    assert [item["id"] for item in output] == ["near"]


def test_hybrid_search_type_requires_full_text_policy(patched_helpers):
    """search_type='hybrid' must require a full-text policy even when the
    full_text_search_enabled flag is left at its default (False)."""
    with pytest.raises(ValueError, match="fullTextIndexes|fullTextPaths"):
        _build_tool(search_type="hybrid")  # no fullTextIndexes/paths provided
