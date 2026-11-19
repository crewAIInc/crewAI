from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

from chromadb import PersistentClient
from chromadb.api import AsyncClientAPI
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from crewai.rag.chromadb.client import ChromaDBClient
import pytest


class FixedEmbedding(EmbeddingFunction[Documents]):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[1.0, 0.0] for _ in input]

    @staticmethod
    def name() -> str:
        return "fixed_metadata_filter_embedding"

    def get_config(self) -> dict[str, Any]:
        return {}


def test_search_applies_flat_metadata_filter(tmp_path: Path) -> None:
    database = PersistentClient(path=str(tmp_path))
    embedder = FixedEmbedding()
    collection = database.get_or_create_collection(
        name="handbook", embedding_function=embedder
    )
    collection.add(
        ids=["match", "other_source", "other_kind"],
        documents=["travel", "wiki", "hr"],
        embeddings=[[1.0, 0.0]] * 3,
        metadatas=[
            {"source": "handbook", "kind": "travel"},
            {"source": "wiki", "kind": "travel"},
            {"source": "handbook", "kind": "hr"},
        ],
    )
    client = ChromaDBClient(database, embedder, default_score_threshold=0)
    criteria = {"source": "handbook", "kind": "travel"}

    results = client.search(
        collection_name="handbook", query="travel", metadata_filter=criteria
    )
    assert [result["id"] for result in results] == ["match"]
    assert criteria == {"source": "handbook", "kind": "travel"}

    results = client.search(
        collection_name="handbook", query="travel", metadata_filter={}
    )
    assert {result["id"] for result in results} == {
        "match",
        "other_source",
        "other_kind",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("criteria", "native_where", "expected"),
    [
        (
            {"source": "handbook", "kind": "travel"},
            None,
            {"$and": [{"source": "handbook"}, {"kind": "travel"}]},
        ),
        ({}, None, None),
        ({"source": "handbook"}, None, {"source": "handbook"}),
        (
            {"$or": [{"source": "handbook"}, {"source": "wiki"}]},
            None,
            {"$or": [{"source": "handbook"}, {"source": "wiki"}]},
        ),
        ({"source": "handbook"}, {"source": "wiki"}, {"source": "wiki"}),
    ],
)
async def test_asearch_metadata_filter(
    criteria: dict[str, Any],
    native_where: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    database = Mock(spec=AsyncClientAPI)
    collection = Mock(metadata={})
    collection.query = AsyncMock(return_value={"ids": [[]]})
    database.get_or_create_collection = AsyncMock(return_value=collection)
    client = ChromaDBClient(database, FixedEmbedding())

    await client.asearch(
        collection_name="handbook",
        query="travel",
        metadata_filter=criteria,
        **({"where": native_where} if native_where is not None else {}),
    )

    assert collection.query.call_args.kwargs["where"] == expected
