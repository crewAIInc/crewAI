import json
import os
from unittest.mock import patch

from crewai_tools import SochDBConfig, SochDBVectorSearchTool
import crewai_tools.tools as tools
import pytest


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
    openai_key_var = next(
        (env_var for env_var in tool.env_vars if env_var.name == "OPENAI_API_KEY"),
        None,
    )
    assert openai_key_var is not None, "OPENAI_API_KEY should be in env_vars"
    assert openai_key_var.required is False


def test_sochdb_vector_search_tool_requires_openai_key_without_custom_embedding() -> None:
    tool = SochDBVectorSearchTool(
        sochdb_config=SochDBConfig(
            grpc_address="studio.agentslab.host:50053",
            collection_name="knowledge",
        ),
        client=FakeSochDBClient(),
    )

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            tool._embed_query("default embedding query")

    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert "custom_embedding_fn" in str(excinfo.value)


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

    with pytest.raises(ValueError, match="JSON object"):
        tool._run(query="test", metadata_filter_json='["bad"]')


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
