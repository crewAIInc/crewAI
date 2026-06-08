import json
import os
from unittest.mock import patch

import crewai_tools.tools as tools
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


def test_sochdb_vector_search_tool_custom_embedding_does_not_require_openai_key() -> None:
    fake_client = FakeSochDBClient()
    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
        ),
        client=fake_client,
        custom_embedding_fn=lambda text: [0.4, 0.5, 0.6],
    )

    with patch.dict(os.environ, {}, clear=True):
        results = json.loads(tool._run(query="custom embedding query"))

    assert len(results) == 1
    assert fake_client.calls[0]["query"] == [0.4, 0.5, 0.6]
    assert tool.env_vars[0].required is False


def test_sochdb_vector_search_tool_requires_openai_key_without_custom_embedding() -> None:
    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
        ),
        client=FakeSochDBClient(),
    )

    with patch.dict(os.environ, {}, clear=True):
        try:
            tool._embed_query("default embedding query")
        except ValueError as exc:
            assert "OPENAI_API_KEY" in str(exc)
            assert "custom_embedding_fn" in str(exc)
        else:
            raise AssertionError("Expected ValueError when OPENAI_API_KEY is missing")


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


def test_sochdb_vector_search_tool_is_exported_from_tools_package() -> None:
    assert tools.SochDBConfig is SochDBConfig
    assert tools.SochDBVectorSearchTool is SochDBVectorSearchTool
    assert "SochDBConfig" in tools.__all__
    assert "SochDBVectorSearchTool" in tools.__all__


def test_sochdb_vector_search_tool_builds_default_client() -> None:
    class FakeSochDBPackage:
        class SochDBClient:
            def __init__(self, grpc_address: str) -> None:
                self.grpc_address = grpc_address

    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
        ),
        sochdb_package=FakeSochDBPackage,
        custom_embedding_fn=lambda text: [0.1],
    )

    assert isinstance(tool.client, FakeSochDBPackage.SochDBClient)
    assert tool.client.grpc_address == "studio.agentslab.host:50053"
