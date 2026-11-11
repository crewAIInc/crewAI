"""Opt-in tests against local DocumentDB or live Azure with Entra authentication."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from threading import Thread
from uuid import uuid4

from crewai_tools import (
    AzureDocumentDBVectorSearchConfig,
    AzureDocumentDBVectorSearchTool,
)
from pymongo.errors import BulkWriteError
import pytest


@pytest.fixture
def documentdb_tool(monkeypatch):
    """Use real OpenAI HTTP/SDK calls with deterministic, offline embeddings."""
    if not os.environ.get("AZURE_DOCUMENTDB_E2E_CONNECTION_STRING"):
        pytest.skip("Set AZURE_DOCUMENTDB_E2E_CONNECTION_STRING for local E2E tests.")
    vectors = {
        "apple": [1.0, 0.0, 0.0],
        "pear": [0.8, 0.2, 0.0],
        "truck": [0.0, 0.0, 1.0],
    }
    requests = []

    class EmbeddingsHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            """Validate an embedding request and return deterministic vectors."""
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            assert self.path == "/v1/embeddings"
            assert request["dimensions"] == 3
            assert request["model"] == "text-embedding-3-small"
            payload = json.dumps(
                {
                    "object": "list",
                    "model": request["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": vectors[text],
                        }
                        for index, text in enumerate(request["input"])
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep the local embedding server's request logs out of test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingsHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-e2e-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    tool = AzureDocumentDBVectorSearchTool(
        connection_string=os.environ["AZURE_DOCUMENTDB_E2E_CONNECTION_STRING"],
        database_name="crewai_e2e",
        collection_name=f"vectors_{uuid4().hex}",
        dimensions=3,
        query_config=AzureDocumentDBVectorSearchConfig(limit=3),
    )
    collection_created = False
    try:
        tool._client.admin.command("ping")
        tool._coll.database.create_collection(tool.collection_name)
        collection_created = True
        yield tool, requests
    finally:
        try:
            if collection_created:
                tool._coll.drop()
        finally:
            tool.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_real_documentdb_ingestion_index_and_ranked_search(documentdb_tool):
    """Verify ingestion, IVF indexing, ranking, and optional result fields."""
    tool, requests = documentdb_tool
    assert tool.add_texts(
        (text for text in ["apple", "pear", "truck"]),
        metadatas=[{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "vehicle"}],
        ids=["apple-id", "pear-id", "truck-id"],
        batch_size=2,
    ) == ["apple-id", "pear-id", "truck-id"]
    tool.create_vector_search_index()
    tool.create_vector_search_index()  # Repeating identical index creation is safe.
    assert tool._coll.count_documents({}) == 3
    assert "crewai_vector_index" in tool._coll.index_information()
    results = json.loads(tool.run(query="apple"))
    assert [doc["_id"] for doc in results] == ["apple-id", "pear-id", "truck-id"]
    assert results[0]["text"] == "apple"
    assert results[0]["metadata"] == {"kind": "fruit"}
    assert all("embedding" not in doc for doc in results)
    assert all(math.isfinite(doc["score"]) for doc in results)
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    assert [request["input"] for request in requests] == [
        ["apple", "pear"],
        ["truck"],
        ["apple"],
    ]

    tool.query_config = AzureDocumentDBVectorSearchConfig(
        limit=1, include_embeddings=True, include_score=False
    )
    results = json.loads(tool.run(query="truck"))
    assert len(results) == 1
    assert results[0]["_id"] == "truck-id"
    assert results[0]["embedding"] == [0.0, 0.0, 1.0]
    assert "score" not in results[0]


def test_real_documentdb_prefilter_and_duplicate_id_error(documentdb_tool):
    """Verify native metadata filtering and duplicate-ID write failures."""
    tool, _ = documentdb_tool
    tool.add_texts(
        ["apple", "pear", "truck"],
        metadatas=[{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "vehicle"}],
        ids=["a", "b", "c"],
    )
    tool.create_vector_search_index()
    tool._coll.create_index("metadata.kind")
    tool.query_config = AzureDocumentDBVectorSearchConfig(
        limit=2, pre_filter={"metadata.kind": "vehicle"}
    )
    results = json.loads(tool.run(query="apple"))
    assert len(results) == 1
    assert results[0]["_id"] == "c"
    tool.query_config.pre_filter = {"metadata.kind": "missing"}
    assert json.loads(tool.run(query="apple")) == []
    with pytest.raises(BulkWriteError):
        tool.add_texts(["apple"], ids=["a"])


@pytest.mark.skipif(
    not os.environ.get("AZURE_DOCUMENTDB_LIVE_CONNECTION_STRING"),
    reason="Set AZURE_DOCUMENTDB_LIVE_CONNECTION_STRING to opt into live Azure tests.",
)
def test_live_azure_documentdb_and_openai_with_entra(monkeypatch):
    """Use an explicitly selected identity without falling back to another login."""
    from azure.identity import AzureCliCredential, EnvironmentCredential

    endpoint = os.environ["AZURE_DOCUMENTDB_LIVE_OPENAI_ENDPOINT"]
    deployment = os.environ["AZURE_DOCUMENTDB_LIVE_EMBEDDING_DEPLOYMENT"]
    api_version = os.environ["AZURE_DOCUMENTDB_LIVE_OPENAI_API_VERSION"]
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", endpoint)
    monkeypatch.setenv("OPENAI_API_VERSION", api_version)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    text = "CrewAI coordinates collaborative AI agents to complete research tasks."
    identity = os.environ.get("AZURE_DOCUMENTDB_LIVE_IDENTITY", "environment")
    if identity == "azure_cli":
        credential = AzureCliCredential(
            tenant_id=os.environ["AZURE_DOCUMENTDB_LIVE_TENANT_ID"]
        )
    elif identity == "environment":
        credential = EnvironmentCredential()
    else:
        raise ValueError(
            "AZURE_DOCUMENTDB_LIVE_IDENTITY must be azure_cli or environment."
        )
    with credential:
        tool = AzureDocumentDBVectorSearchTool(
            connection_string=os.environ["AZURE_DOCUMENTDB_LIVE_CONNECTION_STRING"],
            database_name=os.environ.get(
                "AZURE_DOCUMENTDB_LIVE_DATABASE", "crewai_e2e"
            ),
            collection_name=f"live_vectors_{uuid4().hex}",
            embedding_model=deployment,
            dimensions=1536,
            azure_credential=credential,
            query_config=AzureDocumentDBVectorSearchConfig(limit=2),
        )
        collection_created = False
        try:
            assert tool._client.admin.command("ping")["ok"] == 1
            tool._coll.database.create_collection(tool.collection_name)
            collection_created = True
            assert tool.add_texts(
                [text, "A truck transports heavy cargo along the highway."],
                metadatas=[{"category": "agents"}, {"category": "transport"}],
                ids=["agents", "truck"],
            ) == ["agents", "truck"]
            tool.create_vector_search_index()
            tool._coll.create_index("metadata.category")
            results = json.loads(tool.run(query=text))
            assert len(results) == 2
            assert results[0]["_id"] == "agents"
            assert results[0]["score"] > 0.9
            assert results[0]["score"] > results[1]["score"]
            assert all("embedding" not in doc for doc in results)
            assert tool._coll.count_documents({}) == 2
            tool.query_config = AzureDocumentDBVectorSearchConfig(
                limit=1,
                pre_filter={"metadata.category": "transport"},
                include_embeddings=True,
                include_score=False,
            )
            results = json.loads(tool.run(query=text))
            assert len(results) == 1
            assert results[0]["_id"] == "truck"
            assert len(results[0]["embedding"]) == 1536
            assert all(math.isfinite(value) for value in results[0]["embedding"])
            assert "score" not in results[0]
            tool.query_config.pre_filter = {"metadata.category": "missing"}
            assert json.loads(tool.run(query=text)) == []
        finally:
            try:
                if collection_created:
                    tool._coll.drop()
                    assert (
                        tool.collection_name
                        not in tool._coll.database.list_collection_names()
                    )
            finally:
                tool.close()
