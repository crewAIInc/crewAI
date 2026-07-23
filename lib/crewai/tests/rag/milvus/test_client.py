"""Tests for MilvusClient implementation."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pymilvus = pytest.importorskip("pymilvus")

from crewai.rag.milvus.client import MilvusClient
from crewai.rag.milvus.constants import DEFAULT_CONTENT_MAX_LENGTH
from crewai.rag.milvus.constants import DEFAULT_ID_MAX_LENGTH
from crewai.rag.milvus.utils import _build_filter_expression
from crewai.rag.types import BaseRecord


DataType = pymilvus.DataType
SyncMilvusClient = pymilvus.MilvusClient


def embed_text(text: str) -> list[float]:
    """Return deterministic embeddings for Milvus Lite tests."""
    text = text.lower()
    if "alpha" in text:
        return [1.0, 0.0, 0.0]
    if "beta" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


@pytest.fixture
def raw_client(tmp_path: Path) -> Iterator[Any]:
    """Create a real Milvus Lite client."""
    client = SyncMilvusClient(uri=str(tmp_path / "milvus.db"))
    yield client
    for collection_name in client.list_collections():
        client.drop_collection(collection_name=collection_name)
    if hasattr(client, "close"):
        client.close()


@pytest.fixture
def client(raw_client: Any) -> MilvusClient:
    """Create a CrewAI Milvus client with deterministic embeddings."""
    return MilvusClient(
        client=raw_client,
        embedding_function=embed_text,
        dimension=3,
        default_score_threshold=0.0,
    )


def unique_collection_name(prefix: str = "test_collection") -> str:
    """Return a Milvus-compatible unique collection name."""
    return f"{prefix}_{uuid4().hex}"


def field_by_name(description: dict[str, Any], name: str) -> dict[str, Any]:
    """Find a field description by field name."""
    return next(field for field in description["fields"] if field["name"] == name)


def test_get_or_create_collection_creates_explicit_schema(client: MilvusClient) -> None:
    """Test that Milvus collections use the expected explicit schema."""
    collection_name = unique_collection_name()

    description = client.get_or_create_collection(collection_name=collection_name)

    assert description.get("auto_id") is False
    id_field = field_by_name(description, "id")
    vector_field = field_by_name(description, "vector")
    content_field = field_by_name(description, "content")
    metadata_field = field_by_name(description, "metadata")

    assert id_field["is_primary"] is True
    assert id_field["type"] == DataType.VARCHAR
    assert int(id_field["params"]["max_length"]) == DEFAULT_ID_MAX_LENGTH
    assert vector_field["type"] == DataType.FLOAT_VECTOR
    assert int(vector_field["params"]["dim"]) == 3
    assert content_field["type"] == DataType.VARCHAR
    assert int(content_field["params"]["max_length"]) == DEFAULT_CONTENT_MAX_LENGTH
    assert metadata_field["type"] == DataType.JSON


def test_add_documents_search_orders_scores_and_filters(client: MilvusClient) -> None:
    """Test real Milvus Lite document upsert, search, ordering, and filtering."""
    collection_name = unique_collection_name()
    client.get_or_create_collection(collection_name=collection_name)

    documents: list[BaseRecord] = [
        {
            "doc_id": "alpha-doc",
            "content": "Alpha reference",
            "metadata": {"category": "alpha", "priority": 1, "active": True},
        },
        {
            "doc_id": "beta-doc",
            "content": "Beta reference",
            "metadata": {"category": "beta", "priority": 2, "active": False},
        },
    ]

    client.add_documents(collection_name=collection_name, documents=documents)

    results = client.search(
        collection_name=collection_name,
        query="alpha query",
        limit=2,
    )

    assert [result["id"] for result in results] == ["alpha-doc", "beta-doc"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["metadata"] == {
        "category": "alpha",
        "priority": 1,
        "active": True,
    }

    filtered = client.search(
        collection_name=collection_name,
        query="alpha query",
        limit=2,
        metadata_filter={"category": "alpha"},
        score_threshold=0.9,
    )

    assert [result["id"] for result in filtered] == ["alpha-doc"]


@pytest.mark.asyncio
async def test_async_methods_use_real_milvus_lite(client: MilvusClient) -> None:
    """Test async Milvus operations against a real Milvus Lite database."""
    collection_name = unique_collection_name()
    await client.aget_or_create_collection(collection_name=collection_name)

    await client.aadd_documents(
        collection_name=collection_name,
        documents=[
            {
                "doc_id": "alpha-async",
                "content": "Alpha async document",
                "metadata": {"category": "alpha"},
            }
        ],
    )

    results = await client.asearch(
        collection_name=collection_name,
        query="alpha query",
        limit=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == "alpha-async"
    assert results[0]["score"] == pytest.approx(1.0)


def test_get_or_create_collection_rejects_schema_mismatch(raw_client: Any) -> None:
    """Test that an existing collection with the wrong vector dimension is rejected."""
    collection_name = unique_collection_name()
    schema = raw_client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=DEFAULT_ID_MAX_LENGTH,
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=2)
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=DEFAULT_CONTENT_MAX_LENGTH,
    )
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    index_params = raw_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    raw_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    client = MilvusClient(
        client=raw_client,
        embedding_function=embed_text,
        dimension=3,
    )

    with pytest.raises(ValueError, match="vector dimension does not match"):
        client.get_or_create_collection(collection_name=collection_name)


def test_metadata_filter_expression_escapes_values() -> None:
    """Test that scalar metadata values are escaped for Milvus expressions."""
    expression = _build_filter_expression(
        {
            "category": 'alpha "quoted"',
            "priority": 1,
            "ratio": 1.5,
            "active": True,
        }
    )

    assert expression == (
        'metadata["category"] == "alpha \\"quoted\\"" and '
        'metadata["priority"] == 1 and '
        'metadata["ratio"] == 1.5 and '
        'metadata["active"] == true'
    )


@pytest.mark.parametrize(
    ("metadata_filter", "match"),
    [
        ({"bad-key": "value"}, "filter keys"),
        ({"content": "value"}, "reserved collection field"),
        ({"category": ["alpha"]}, "support only"),
    ],
)
def test_metadata_filter_rejects_unsafe_inputs(
    metadata_filter: dict[str, Any],
    match: str,
) -> None:
    """Test safe metadata filter validation."""
    with pytest.raises(ValueError, match=match):
        _build_filter_expression(metadata_filter)
