import json

from crewai_tools import QdrantVectorSearchTool
from crewai_tools.tools.qdrant_vector_search_tool.qdrant_search_tool import QdrantConfig
import pytest


qdrant = pytest.importorskip("qdrant_client")
models = qdrant.models


@pytest.mark.parametrize("filter_kind", ["none", "empty", "tenant"])
def test_search_filters_are_isolated_between_calls(filter_kind: str) -> None:
    base_filter = None
    if filter_kind != "none":
        base_filter = models.Filter(must=[])
    if filter_kind == "tenant":
        base_filter.must.append(
            models.FieldCondition(key="tenant", match=models.MatchValue(value="allowed"))
        )

    client = qdrant.QdrantClient(":memory:")
    try:
        client.create_collection(
            "documents",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        client.upsert(
            "documents",
            points=[
                models.PointStruct(
                    id=number,
                    vector=[1.0, 0.0],
                    payload={"text": text, "category": category, "tenant": tenant},
                )
                for number, text, category, tenant in [
                    (1, "Red document", "red", "allowed"),
                    (2, "Blue document", "blue", "allowed"),
                    (3, "Other tenant", "red", "other"),
                ]
            ],
        )
        tool = QdrantVectorSearchTool(
            qdrant_config=QdrantConfig(
                qdrant_url="http://unused.example",
                collection_name="documents",
                filter=base_filter,
            ),
            client=client,
            custom_embedding_fn=lambda query: [1.0, 0.0],
        )
        original_filter = base_filter.model_dump() if base_filter is not None else None

        def search(**kwargs: str) -> set[str]:
            return {
                row["context"]
                for row in json.loads(tool.run(query="documents", **kwargs))
            }

        expected_red = {"Red document"}
        if filter_kind != "tenant":
            expected_red.add("Other tenant")
        assert search(filter_by="category", filter_value="red") == expected_red
        assert search(filter_by="category", filter_value="blue") == {"Blue document"}
        assert search() == expected_red | {"Blue document"}
        if base_filter is not None:
            assert base_filter.model_dump() == original_filter
    finally:
        client.close()
