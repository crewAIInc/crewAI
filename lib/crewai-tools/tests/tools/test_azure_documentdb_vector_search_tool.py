import builtins
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bson import ObjectId
from crewai_tools import (
    AzureDocumentDBToolSchema,
    AzureDocumentDBVectorSearchConfig,
    AzureDocumentDBVectorSearchTool,
)
from crewai_tools.generate_tool_specs import ToolSpecExtractor
from crewai_tools.tools.azure_documentdb_vector_search_tool import vector_search
from pydantic import ValidationError
from pymongo.errors import OperationFailure
import pytest


@pytest.fixture
def tool(monkeypatch):
    """Provide an isolated tool with mocked database and embedding clients."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with (
        patch("pymongo.MongoClient"),
        patch.object(vector_search, "OpenAI") as openai,
    ):
        openai.return_value.embeddings.create.side_effect = lambda **kwargs: (
            SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[1.0, 0.0, 0.0])
                    for index, _ in enumerate(kwargs["input"])
                ]
            )
        )
        instance = AzureDocumentDBVectorSearchTool(
            connection_string="mongodb://localhost",
            database_name="test",
            collection_name="documents",
            dimensions=3,
        )
        yield instance
        instance.close()


def test_public_schema_and_default_configuration():
    """Verify query validation and the default search result limit."""
    assert AzureDocumentDBToolSchema(query="hello").query == "hello"
    assert AzureDocumentDBVectorSearchConfig().limit == 4
    with pytest.raises(ValidationError):
        AzureDocumentDBVectorSearchConfig(limit=0)
    with pytest.raises(ValidationError):
        AzureDocumentDBToolSchema(query="")


def test_tool_is_discoverable_without_instantiation():
    """Verify public exports and tool metadata without opening connections."""
    from crewai_tools import tools

    assert tools.AzureDocumentDBVectorSearchTool is AzureDocumentDBVectorSearchTool
    extractor = ToolSpecExtractor()
    extractor.extract_tool_info(AzureDocumentDBVectorSearchTool)
    (spec,) = extractor.tools_spec
    assert spec["name"] == "AzureDocumentDBVectorSearchTool"
    assert spec["package_dependencies"] == ["pymongo", "azure-identity"]
    assert spec["run_params_schema"]["required"] == ["query"]
    assert set(spec["init_params_schema"]["required"]) == {
        "database_name",
        "collection_name",
    }


def test_connection_from_environment_and_no_secret_in_serialization(monkeypatch):
    """Verify environment configuration and connection-string redaction."""
    uri = "mongodb://username:private-password@localhost"
    monkeypatch.setenv("AZURE_DOCUMENTDB_CONNECTION_STRING", uri)
    with patch("pymongo.MongoClient") as mongo:
        instance = AzureDocumentDBVectorSearchTool(
            database_name="db", collection_name="docs"
        )
        assert instance.connection_string == uri
        assert "private-password" not in repr(instance)
        assert "connection_string" not in instance.model_dump()
        assert mongo.call_args.kwargs["serverSelectionTimeoutMS"] == 10000
        instance.close()


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    """Verify a missing database driver produces installation guidance."""
    original_import = builtins.__import__

    def without_pymongo(name, *args, **kwargs):
        """Simulate an unavailable PyMongo package without blocking other imports."""
        if name == "pymongo":
            raise ImportError("missing pymongo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pymongo)
    with pytest.raises(ImportError, match="crewai-tools\\[azure-documentdb\\]"):
        AzureDocumentDBVectorSearchTool(
            connection_string="mongodb://localhost",
            database_name="db",
            collection_name="docs",
        )


def test_native_ivf_index(tool):
    """Verify the index helper sends DocumentDB's native IVF definition."""
    tool.create_vector_search_index(num_lists=10, similarity="IP")
    tool._coll.database.command.assert_called_once_with(
        {
            "createIndexes": "documents",
            "indexes": [
                {
                    "name": "crewai_vector_index",
                    "key": {"embedding": "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": "vector-ivf",
                        "numLists": 10,
                        "dimensions": 3,
                        "similarity": "IP",
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("kwargs", [{"num_lists": 0}, {"similarity": "cosine"}])
def test_invalid_index_options_do_not_reach_database(tool, kwargs):
    """Verify invalid index settings fail before database access."""
    with pytest.raises(ValueError):
        tool.create_vector_search_index(**kwargs)
    tool._coll.database.command.assert_not_called()


def test_ingest_generator_batches_preserves_ids_and_metadata(tool):
    """Verify batched generator ingestion preserves IDs and nested metadata."""
    metadata = [{"text": "cannot overwrite"}, {"category": "two"}, {}]
    ids = tool.add_texts(
        (text for text in ["one", "two", "three"]),
        metadatas=metadata,
        ids=["a", "b", "c"],
        batch_size=2,
    )
    assert ids == ["a", "b", "c"]
    calls = tool._coll.insert_many.call_args_list
    assert [len(call.args[0]) for call in calls] == [2, 1]
    assert calls[0].args[0][0] == {
        "_id": "a",
        "text": "one",
        "metadata": {"text": "cannot overwrite"},
        "embedding": [1.0, 0.0, 0.0],
    }
    assert calls[1].args[0][0]["_id"] == "c"
    assert metadata[0] == {"text": "cannot overwrite"}
    assert tool._openai_client.embeddings.create.call_args.kwargs == {
        "input": ["three"],
        "model": "text-embedding-3-small",
        "dimensions": 3,
    }


def test_generated_ids_and_empty_input(tool):
    """Verify generated IDs are unique and empty input performs no writes."""
    ids = tool.add_texts(["one", "two"])
    assert len(ids) == len(set(ids)) == 2
    assert all(isinstance(value, str) for value in ids)
    tool._coll.insert_many.reset_mock()
    assert tool.add_texts(iter([])) == []
    tool._coll.insert_many.assert_not_called()


@pytest.mark.parametrize(
    "texts,kwargs",
    [
        (["one", "two"], {"metadatas": [{}]}),
        (["one", "two"], {"ids": ["a"]}),
        (["one", "two"], {"ids": ["a", "a"]}),
        (["one"], {"ids": [""]}),
        (["one"], {"batch_size": 0}),
        ([" "], {}),
        ([None], {}),
    ],
)
def test_invalid_ingestion_has_no_side_effects(tool, texts, kwargs):
    """Verify invalid documents and batch options fail before external calls."""
    with pytest.raises(ValueError):
        tool.add_texts(texts, **kwargs)
    tool._coll.insert_many.assert_not_called()
    assert tool._openai_client is None


def test_query_uses_documentdb_not_atlas_and_returns_extended_json(tool):
    """Verify native search stages and JSON serialization of BSON values."""
    object_id = ObjectId()
    tool._coll.aggregate.return_value = [
        {
            "_id": object_id,
            "text": "one",
            "metadata": {"at": datetime(2026, 1, 1, tzinfo=timezone.utc)},
            "score": 1.0,
        }
    ]
    results = json.loads(tool.run(query="one"))
    assert results[0]["_id"] == {"$oid": str(object_id)}
    assert "$date" in results[0]["metadata"]["at"]
    assert results[0]["score"] == 1.0
    tool._coll.aggregate.assert_called_once_with(
        [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": [1.0, 0.0, 0.0],
                        "path": "embedding",
                        "k": 4,
                    },
                    "returnStoredSource": True,
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$project": {"embedding": 0}},
        ]
    )


def test_query_options(tool):
    """Verify filters, limits, and field options shape the search pipeline."""
    tool.query_config = AzureDocumentDBVectorSearchConfig(
        limit=2,
        pre_filter={"metadata.category": "one"},
        include_embeddings=True,
        include_score=False,
    )
    tool._coll.aggregate.return_value = []
    assert json.loads(tool._run("one")) == []
    tool._coll.aggregate.assert_called_once_with(
        [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": [1.0, 0.0, 0.0],
                        "path": "embedding",
                        "k": 2,
                        "filter": {"metadata.category": "one"},
                    },
                    "returnStoredSource": True,
                }
            }
        ]
    )


def test_blank_query_is_rejected(tool):
    """Verify whitespace-only queries never reach the database."""
    with pytest.raises(ValueError, match="blank"):
        tool._run("  ")
    tool._coll.aggregate.assert_not_called()


def test_database_errors_propagate(tool):
    """Verify search, ingestion, and indexing errors reach the caller."""
    tool._coll.aggregate.side_effect = OperationFailure("index missing")
    with pytest.raises(OperationFailure, match="index missing"):
        tool._run("one")
    tool._coll.insert_many.side_effect = OperationFailure("write denied")
    with pytest.raises(OperationFailure, match="write denied"):
        tool.add_texts(["one"])
    tool._coll.database.command.side_effect = OperationFailure("index denied")
    with pytest.raises(OperationFailure, match="index denied"):
        tool.create_vector_search_index()


def test_missing_embedding_credentials(tool, monkeypatch):
    """Verify missing embedding credentials fail before document insertion."""
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        tool.add_texts(["one"])
    tool._coll.insert_many.assert_not_called()


def test_azure_openai_uses_deployment_and_dimensions(tool, monkeypatch):
    """Verify Azure embeddings use the configured deployment and dimensions."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    tool.embedding_model = "my-embedding-deployment"
    with patch.object(vector_search, "AzureOpenAI") as azure:
        azure.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])]
        )
        tool.add_texts(["one"])
        azure.assert_called_once_with()
        azure.return_value.embeddings.create.assert_called_once_with(
            input=["one"], model="my-embedding-deployment", dimensions=3
        )


@pytest.mark.parametrize(
    "vectors", [[], [[1.0]], [[float("nan"), 0, 0]], [[float("inf"), 0, 0]]]
)
def test_invalid_embedding_response_prevents_write(tool, vectors):
    """Verify missing, malformed, or non-finite vectors prevent insertion."""
    with patch.object(vector_search, "OpenAI") as openai:
        openai.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=vector)
                for i, vector in enumerate(vectors)
            ]
        )
        with pytest.raises(ValueError, match="finite vector"):
            tool.add_texts(["one"])
    tool._coll.insert_many.assert_not_called()


def test_embeddings_are_ordered_by_response_index(tool):
    """Verify out-of-order embedding responses stay paired with their texts."""
    with patch.object(vector_search, "OpenAI") as openai:
        openai.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
            ]
        )
        tool.add_texts(["one", "two"])
    docs = tool._coll.insert_many.call_args.args[0]
    assert docs[0]["embedding"] == [1.0, 0.0, 0.0]
    assert docs[1]["embedding"] == [0.0, 1.0, 0.0]


def test_close_releases_both_clients(tool):
    """Verify closing the tool releases database and embedding connections."""
    tool.add_texts(["one"])
    tool.close()
    tool._client.close.assert_called_once()
    tool._openai_client.close.assert_called_once()


def test_oidc_connection_uses_injected_credential_and_refreshable_callback():
    """Verify OIDC token renewal and caller ownership of injected credentials."""
    from azure.core.credentials import AccessToken
    from pymongo.auth_oidc import OIDCCallbackContext

    credential = MagicMock()
    credential.get_token.side_effect = [
        AccessToken("first-token", 200),
        AccessToken("refreshed-token", 300),
    ]
    with patch("pymongo.MongoClient") as mongo:
        instance = AzureDocumentDBVectorSearchTool(
            connection_string="mongodb://localhost/?authMechanism=MONGODB-OIDC",
            database_name="db",
            collection_name="docs",
            azure_credential=credential,
        )
        callback = mongo.call_args.kwargs["authMechanismProperties"]["OIDC_CALLBACK"]
        with patch(
            "crewai_tools.tools.azure_documentdb_vector_search_tool.auth.time.time",
            return_value=100,
        ):
            first = callback.fetch(
                OIDCCallbackContext(timeout_seconds=30, username="", version=1)
            )
            refreshed = callback.fetch(
                OIDCCallbackContext(timeout_seconds=30, username="", version=1)
            )
        assert first.access_token == "first-token"
        assert first.expires_in_seconds == 100
        assert refreshed.access_token == "refreshed-token"
        assert refreshed.expires_in_seconds == 200
        assert credential.get_token.call_count == 2
        credential.get_token.assert_called_with(
            "https://ossrdbms-aad.database.windows.net/.default"
        )
        instance.close()
        credential.close.assert_not_called()


def test_default_identity_is_closed_by_tool():
    """Verify the tool closes credentials it creates for OIDC authentication."""
    with (
        patch("pymongo.MongoClient"),
        patch("azure.identity.DefaultAzureCredential") as default_credential,
    ):
        instance = AzureDocumentDBVectorSearchTool(
            connection_string="mongodb://localhost/?AUTHMECHANISM=MONGODB-OIDC",
            database_name="db",
            collection_name="docs",
        )
        default_credential.assert_called_once_with()
        instance.close()
        default_credential.return_value.close.assert_called_once()


def test_entra_openai_uses_renewable_token_provider(tool, monkeypatch):
    """Verify Azure OpenAI receives a renewable bearer-token provider."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    credential = MagicMock()
    tool._azure_credential = credential
    with (
        patch("azure.identity.get_bearer_token_provider") as provider,
        patch.object(vector_search, "AzureOpenAI") as azure,
    ):
        azure.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])]
        )
        tool.add_texts(["one"])
        provider.assert_called_once_with(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        azure.assert_called_once_with(azure_ad_token_provider=provider.return_value)
        tool.close()
        credential.close.assert_not_called()


def test_entra_embedding_requires_azure_endpoint(tool):
    """Verify keyless Azure embeddings require an Azure OpenAI endpoint."""
    tool.use_azure_identity = True
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        tool.add_texts(["one"])
    tool._coll.insert_many.assert_not_called()


def test_entra_embedding_without_oidc_database_uses_default_credential(
    tool, monkeypatch
):
    """Verify keyless embeddings work independently of database authentication."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    tool.use_azure_identity = True
    with (
        patch("azure.identity.DefaultAzureCredential") as credential,
        patch("azure.identity.get_bearer_token_provider"),
        patch.object(vector_search, "AzureOpenAI") as azure,
    ):
        azure.return_value.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0])]
        )
        tool.add_texts(["one"])
        credential.assert_called_once_with()
        tool.close()
        credential.return_value.close.assert_called_once()


def test_oidc_token_acquisition_failure_is_not_hidden():
    """Verify OIDC credential failures propagate without authentication fallback."""
    from azure.core.exceptions import ClientAuthenticationError
    from crewai_tools.tools.azure_documentdb_vector_search_tool.auth import (
        DocumentDBOIDCCallback,
    )
    from pymongo.auth_oidc import OIDCCallbackContext

    credential = MagicMock()
    credential.get_token.side_effect = ClientAuthenticationError("access denied")
    with pytest.raises(ClientAuthenticationError, match="access denied"):
        DocumentDBOIDCCallback(credential).fetch(
            OIDCCallbackContext(timeout_seconds=30, username="", version=1)
        )
