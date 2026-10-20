"""Tests for `WeaviateVectorSearchTool`.

Every test here runs fully mocked: the suite is network-blocked (`--block-network`) and CI
has no Weaviate to talk to.
"""

import importlib.metadata
import sys
from unittest.mock import MagicMock, patch

import pytest


pytest.importorskip("weaviate", reason="needs the weaviate-client optional dependency")

from crewai_tools.tools.weaviate_tool import vector_search  # noqa: E402
from crewai_tools.tools.weaviate_tool.vector_search import (  # noqa: E402
    WeaviateVectorSearchTool,
    _integration_version,
    _register_integration,
)


INTEGRATION_HEADER = "X-Weaviate-Client-Integration"


@pytest.fixture
def weaviate_tool(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    return WeaviateVectorSearchTool(
        collection_name="Docs",
        weaviate_cluster_url="https://example.weaviate.cloud",
        weaviate_api_key="fake-weaviate-key",
    )


@pytest.fixture
def connected_client():
    """A stand-in Weaviate client that returns an empty result set."""
    client = MagicMock()
    client.collections.get.return_value.query.hybrid.return_value.objects = []
    return client


def _configured_header(client: MagicMock) -> dict[str, str]:
    """The header dict the tool handed to `client.integrations.configure`."""
    return client.integrations.configure.call_args.args[0]._to_header()


def _run_against(tool: WeaviateVectorSearchTool, client: MagicMock) -> MagicMock:
    """Runs `tool` against `client`, returning the patched `weaviate` module."""
    with patch.object(vector_search, "weaviate") as mock_weaviate:
        mock_weaviate.connect_to_weaviate_cloud.return_value = client
        tool._run("what is weaviate")
    return mock_weaviate


# --- the version reported alongside the integration name --------------------------------


def test_integration_version_comes_from_the_package_metadata():
    with patch.object(importlib.metadata, "version", return_value="1.2.3"):
        assert _integration_version() == "1.2.3"


def test_integration_version_is_stripped():
    with patch.object(importlib.metadata, "version", return_value="  1.2.3\n"):
        assert _integration_version() == "1.2.3"


def test_integration_version_falls_back_to_unknown():
    with patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        assert _integration_version() == "unknown"


def test_integration_version_falls_back_when_the_metadata_is_empty():
    with patch.object(importlib.metadata, "version", return_value="   "):
        assert _integration_version() == "unknown"


def test_the_version_is_read_from_the_crewai_tools_distribution():
    """`crewai` is a pinned dependency on the same version, so asserting on the resulting
    string cannot tell the two distributions apart. Assert on which one was asked."""
    with patch.object(
        importlib.metadata, "version", return_value="1.2.3"
    ) as mock_version:
        _integration_version()

    mock_version.assert_called_once_with("crewai-tools")


def test_the_reported_version_resolves_against_real_metadata():
    assert _integration_version() == importlib.metadata.version("crewai-tools")
    assert _integration_version() != "unknown"


# --- header registration ----------------------------------------------------------------


def test_register_integration_sets_the_header():
    client = MagicMock()

    with patch.object(importlib.metadata, "version", return_value="1.2.3"):
        _register_integration(client)

    client.integrations.configure.assert_called_once()
    assert _configured_header(client) == {INTEGRATION_HEADER: "crewai-tools/1.2.3"}


def test_register_integration_reports_an_unknown_version():
    client = MagicMock()

    with patch.object(
        importlib.metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError,
    ):
        _register_integration(client)

    assert _configured_header(client) == {INTEGRATION_HEADER: "crewai-tools/unknown"}


def test_the_header_matches_what_weaviate_parses():
    """Weaviate truncates the value to 128 bytes, then splits on the first `/`."""
    client = MagicMock()

    _register_integration(client)

    value = _configured_header(client)[INTEGRATION_HEADER]
    assert value.startswith("crewai-tools/")
    assert value.count("/") == 1
    assert len(value.encode()) <= 128


def test_register_integration_skips_a_client_without_the_integrations_api():
    """`_IntegrationConfig` is private API; an older weaviate-client may not have it."""
    client = MagicMock()

    with patch.dict(sys.modules, {"weaviate.connect.integrations": None}):
        _register_integration(client)

    client.integrations.configure.assert_not_called()


def test_register_integration_swallows_a_failing_configure():
    client = MagicMock()
    client.integrations.configure.side_effect = RuntimeError("boom")

    _register_integration(client)


# --- how the tool uses it ---------------------------------------------------------------


def test_run_sets_the_header_on_the_connected_client(weaviate_tool, connected_client):
    _run_against(weaviate_tool, connected_client)

    assert _configured_header(connected_client) == {
        INTEGRATION_HEADER: f"crewai-tools/{importlib.metadata.version('crewai-tools')}"
    }


def test_run_registers_the_header_before_querying(weaviate_tool, connected_client):
    """Registering after the query would leave the search itself unattributed."""
    _run_against(weaviate_tool, connected_client)

    calls = [call[0] for call in connected_client.mock_calls]
    assert calls.index("integrations.configure") < calls.index("collections.get")


def test_run_succeeds_when_registration_fails(weaviate_tool, connected_client):
    """Telemetry is best-effort: a broken integrations API must not break searching."""
    connected_client.integrations.configure.side_effect = RuntimeError("boom")

    _run_against(weaviate_tool, connected_client)

    connected_client.collections.get.return_value.query.hybrid.assert_called_once()


def test_the_integration_header_is_not_a_connection_header(
    weaviate_tool, connected_client
):
    """It goes through `integrations.configure` so that it reaches gRPC as well as REST."""
    mock_weaviate = _run_against(weaviate_tool, connected_client)

    headers = mock_weaviate.connect_to_weaviate_cloud.call_args.kwargs["headers"]
    assert INTEGRATION_HEADER not in (headers or {})


# --- connection -------------------------------------------------------------------------


def _run_tool(tool: WeaviateVectorSearchTool, client: MagicMock) -> MagicMock:
    """Runs `tool` against `client` whichever way it connects; returns the `weaviate` mock."""
    with patch.object(vector_search, "weaviate") as mock_weaviate:
        mock_weaviate.connect_to_weaviate_cloud.return_value = client
        mock_weaviate.connect_to_custom.return_value = client
        tool._run("what is weaviate")
    return mock_weaviate


def test_a_cluster_url_connects_to_weaviate_cloud(weaviate_tool, connected_client):
    mock_weaviate = _run_tool(weaviate_tool, connected_client)

    mock_weaviate.connect_to_weaviate_cloud.assert_called_once()
    assert (
        mock_weaviate.connect_to_weaviate_cloud.call_args.kwargs["cluster_url"]
        == "https://example.weaviate.cloud"
    )
    mock_weaviate.connect_to_custom.assert_not_called()


def test_a_cluster_url_without_an_api_key_is_rejected(connected_client):
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", weaviate_cluster_url="https://example.weaviate.cloud"
    )

    with pytest.raises(ValueError, match="weaviate_api_key"):
        _run_tool(tool, connected_client)


def test_without_a_cluster_url_it_connects_to_a_self_hosted_weaviate(
    connected_client,
):
    tool = WeaviateVectorSearchTool(
        collection_name="Docs",
        weaviate_host="weaviate.internal",
        weaviate_http_port=443,
        weaviate_grpc_port=50052,
        weaviate_secure=True,
    )

    mock_weaviate = _run_tool(tool, connected_client)

    kwargs = mock_weaviate.connect_to_custom.call_args.kwargs
    assert kwargs["http_host"] == kwargs["grpc_host"] == "weaviate.internal"
    assert (kwargs["http_port"], kwargs["grpc_port"]) == (443, 50052)
    assert kwargs["http_secure"] is kwargs["grpc_secure"] is True
    assert kwargs["auth_credentials"] is None
    mock_weaviate.connect_to_weaviate_cloud.assert_not_called()


def test_the_default_connection_is_a_local_weaviate(connected_client):
    tool = WeaviateVectorSearchTool(collection_name="Docs")

    kwargs = _run_tool(tool, connected_client).connect_to_custom.call_args.kwargs

    assert (kwargs["http_host"], kwargs["http_port"], kwargs["grpc_port"]) == (
        "localhost",
        8080,
        50051,
    )
    assert kwargs["http_secure"] is False


def test_the_client_is_closed_when_the_search_fails(weaviate_tool, connected_client):
    connected_client.collections.get.return_value.query.hybrid.side_effect = (
        RuntimeError("boom")
    )

    with pytest.raises(RuntimeError):
        _run_tool(weaviate_tool, connected_client)

    connected_client.close.assert_called_once()


# --- the OpenAI key ---------------------------------------------------------------------


def test_the_openai_key_is_optional(monkeypatch):
    """A server-side or self-provided vectorizer needs no OpenAI key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    tool = WeaviateVectorSearchTool(collection_name="Docs")

    assert "X-OpenAI-Api-Key" not in (tool.headers or {})


def test_the_openai_key_is_added_without_dropping_user_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    tool = WeaviateVectorSearchTool(
        collection_name="Docs", headers={"X-Cohere-Api-Key": "fake-cohere-key"}
    )

    assert tool.headers == {
        "X-OpenAI-Api-Key": "fake-openai-key",
        "X-Cohere-Api-Key": "fake-cohere-key",
    }


# --- multi-tenancy ----------------------------------------------------------------------


def test_a_tenant_scopes_the_search_to_that_tenant(connected_client):
    tool = WeaviateVectorSearchTool(collection_name="Docs", tenant="acme")

    _run_tool(tool, connected_client)

    collection = connected_client.collections.get.return_value
    collection.with_tenant.assert_called_once_with("acme")
    collection.with_tenant.return_value.query.hybrid.assert_called_once()
    collection.query.hybrid.assert_not_called()


def test_without_a_tenant_the_collection_is_searched_directly(
    weaviate_tool, connected_client
):
    _run_tool(weaviate_tool, connected_client)

    collection = connected_client.collections.get.return_value
    collection.with_tenant.assert_not_called()
    collection.query.hybrid.assert_called_once()


def test_the_tenant_is_not_an_agent_argument():
    """The agent picks the query; which tenant it may read is fixed by whoever built the tool."""
    assert set(WeaviateVectorSearchTool.model_fields["args_schema"].default.model_fields) == {
        "query"
    }


# --- search controls --------------------------------------------------------------------


def test_hybrid_search_passes_every_search_option(connected_client):
    filters = MagicMock(name="filters")
    tool = WeaviateVectorSearchTool(
        collection_name="Docs",
        limit=5,
        alpha=0.3,
        target_vector="title_vector",
        return_properties=["title"],
        filters=filters,
    )

    _run_tool(tool, connected_client)

    connected_client.collections.get.return_value.query.hybrid.assert_called_once_with(
        query="what is weaviate",
        alpha=0.3,
        target_vector="title_vector",
        limit=5,
        filters=filters,
        return_properties=["title"],
    )


def test_near_text_search(connected_client):
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", search_type="near_text", target_vector=["a", "b"]
    )

    _run_tool(tool, connected_client)

    query = connected_client.collections.get.return_value.query
    query.near_text.assert_called_once_with(
        query="what is weaviate",
        target_vector=["a", "b"],
        limit=3,
        filters=None,
        return_properties=None,
    )
    query.hybrid.assert_not_called()


def test_bm25_search(connected_client):
    tool = WeaviateVectorSearchTool(collection_name="Docs", search_type="bm25")

    _run_tool(tool, connected_client)

    query = connected_client.collections.get.return_value.query
    query.bm25.assert_called_once_with(
        query="what is weaviate", limit=3, filters=None, return_properties=None
    )
    query.hybrid.assert_not_called()


def test_an_unknown_search_type_is_rejected():
    with pytest.raises(ValueError):
        WeaviateVectorSearchTool(collection_name="Docs", search_type="vector")


def test_results_are_returned_as_json(weaviate_tool, connected_client):
    hit = MagicMock(properties={"title": "Weaviate"})
    connected_client.collections.get.return_value.query.hybrid.return_value.objects = [
        hit
    ]

    with patch.object(vector_search, "weaviate") as mock_weaviate:
        mock_weaviate.connect_to_weaviate_cloud.return_value = connected_client
        result = weaviate_tool._run("what is weaviate")

    assert '"title": "Weaviate"' in result


# --- collection creation ----------------------------------------------------------------


def _created_with(client: MagicMock) -> dict:
    client.collections.create.assert_called_once()
    return client.collections.create.call_args.kwargs


def test_a_missing_collection_is_not_created_by_default(weaviate_tool, connected_client):
    """A typo in `collection_name` should fail the search, not create an empty collection."""
    connected_client.collections.exists.return_value = False

    _run_tool(weaviate_tool, connected_client)

    connected_client.collections.create.assert_not_called()


def test_an_existing_collection_is_not_recreated(connected_client):
    connected_client.collections.exists.return_value = True
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", create_collection_if_missing=True
    )

    _run_tool(tool, connected_client)

    connected_client.collections.create.assert_not_called()


def test_a_missing_collection_is_created_when_asked(connected_client):
    from weaviate.classes.config import Configure

    connected_client.collections.exists.return_value = False
    vectorizer = Configure.Vectors.text2vec_openai(model="text-embedding-3-large")
    tool = WeaviateVectorSearchTool(
        collection_name="Docs",
        create_collection_if_missing=True,
        vectorizer=vectorizer,
    )

    _run_tool(tool, connected_client)

    kwargs = _created_with(connected_client)
    assert kwargs["name"] == "Docs"
    assert kwargs["vector_config"] is vectorizer
    assert kwargs["generative_config"] is tool.generative_model
    assert kwargs["multi_tenancy_config"] is None
    assert "vectorizer_config" not in kwargs


def test_the_quantizer_reaches_the_created_collection(connected_client):
    from weaviate.classes.config import Configure

    connected_client.collections.exists.return_value = False
    tool = WeaviateVectorSearchTool(
        collection_name="Docs",
        create_collection_if_missing=True,
        vectorizer=Configure.Vectors.text2vec_openai(
            quantizer=Configure.VectorIndex.Quantizer.bq()
        ),
    )

    _run_tool(tool, connected_client)

    config = _created_with(connected_client)["vector_config"]._to_dict()
    assert "bq" in config["vectorIndexConfig"]
    assert config["vectorIndexConfig"]["bq"]["enabled"] is True


def test_a_legacy_vectorizer_still_creates_the_collection(connected_client):
    from weaviate.classes.config import Configure

    connected_client.collections.exists.return_value = False
    vectorizer = Configure.Vectorizer.text2vec_openai()
    tool = WeaviateVectorSearchTool(
        collection_name="Docs",
        create_collection_if_missing=True,
        vectorizer=vectorizer,
    )

    _run_tool(tool, connected_client)

    kwargs = _created_with(connected_client)
    assert kwargs["vectorizer_config"] is vectorizer
    assert "vector_config" not in kwargs


def test_a_tenant_creates_a_multi_tenant_collection(connected_client):
    connected_client.collections.exists.return_value = False
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", create_collection_if_missing=True, tenant="acme"
    )

    _run_tool(tool, connected_client)

    mt = _created_with(connected_client)["multi_tenancy_config"]
    assert mt.enabled is True
    assert mt.autoTenantCreation is True


def test_a_created_multi_tenant_collection_gets_the_tenant(connected_client):
    """Searching a tenant that does not exist fails, even with auto tenant creation."""
    connected_client.collections.exists.return_value = False
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", create_collection_if_missing=True, tenant="acme"
    )

    _run_tool(tool, connected_client)

    tenants = connected_client.collections.get.return_value.tenants
    tenants.create.assert_called_once_with("acme")


def test_a_just_created_collection_is_not_searched(connected_client):
    """It is empty, and Weaviate rejects searching a collection with no properties."""
    connected_client.collections.exists.return_value = False
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", create_collection_if_missing=True
    )

    with patch.object(vector_search, "weaviate") as mock_weaviate:
        mock_weaviate.connect_to_custom.return_value = connected_client
        result = tool._run("what is weaviate")

    assert result == ""
    connected_client.collections.get.return_value.query.hybrid.assert_not_called()
    connected_client.close.assert_called_once()


def test_tenants_are_not_created_on_an_existing_collection(connected_client):
    """A missing tenant on an existing collection is the caller's mistake to see."""
    connected_client.collections.exists.return_value = True
    tool = WeaviateVectorSearchTool(
        collection_name="Docs", create_collection_if_missing=True, tenant="acme"
    )

    _run_tool(tool, connected_client)

    connected_client.collections.get.return_value.tenants.create.assert_not_called()
