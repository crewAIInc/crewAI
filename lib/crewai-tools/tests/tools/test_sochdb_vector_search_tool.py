import json

from crewai_tools import SochDBConfig, SochDBVectorSearchTool


class FakeDocument:
    def __init__(self, doc_id: str, content: str, metadata: dict[str, str]) -> None:
        self.id = doc_id
        self.content = content
        self.metadata = metadata


class FakeSochDBClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_collection(
        self,
        collection_name: str,
        query: list[float],
        k: int = 10,
        namespace: str = "default",
        filter: dict[str, str] | None = None,
    ) -> list[FakeDocument]:
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "k": k,
                "namespace": namespace,
                "filter": filter,
            }
        )
        return [
            FakeDocument(
                doc_id="doc-1",
                content="SochDB supports hosted gRPC access.",
                metadata={"topic": "deployment"},
            )
        ]


def test_sochdb_vector_search_tool_successful_query() -> None:
    fake_client = FakeSochDBClient()
    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
            namespace="crew",
            limit=2,
        ),
        client=fake_client,
        custom_embedding_fn=lambda text: [0.1, 0.2, 0.3],
    )

    results = json.loads(
        tool._run(
            query="Where is the hosted endpoint?",
            metadata_filter_json='{"topic":"deployment"}',
        )
    )

    assert len(results) == 1
    assert results[0]["id"] == "doc-1"
    assert fake_client.calls[0]["collection_name"] == "knowledge"
    assert fake_client.calls[0]["namespace"] == "crew"
    assert fake_client.calls[0]["k"] == 2
    assert fake_client.calls[0]["filter"] == {"topic": "deployment"}


def test_sochdb_vector_search_tool_requires_object_filter() -> None:
    fake_client = FakeSochDBClient()
    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
        ),
        client=fake_client,
        custom_embedding_fn=lambda text: [0.1, 0.2, 0.3],
    )

    try:
        tool._run(query="test", metadata_filter_json='["bad"]')
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object metadata_filter_json")
