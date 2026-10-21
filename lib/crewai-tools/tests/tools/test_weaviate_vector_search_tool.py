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
