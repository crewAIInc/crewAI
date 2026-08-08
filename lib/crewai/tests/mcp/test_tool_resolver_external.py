"""Tests for MCPToolResolver external (streamable-HTTP) resolution paths."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.tool_resolver import MCPToolResolver


SERVER_URL = "https://mcp.example.com/mcp"


@pytest.fixture
def agent():
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )


@pytest.fixture
def resolver(agent):
    return MCPToolResolver(agent=agent, logger=agent._logger)


class _FakeSession:
    """Minimal stand-in for ``mcp.ClientSession``."""

    def __init__(self, tools):
        self._tools = tools

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)


class _FakeTransport:
    async def __aenter__(self):
        return (None, None, None)

    async def __aexit__(self, *exc):
        return False


def _discover(resolver, tools):
    with (
        patch("mcp.ClientSession", lambda *a, **k: _FakeSession(tools)),
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            lambda *a, **k: _FakeTransport(),
        ),
    ):
        return asyncio.run(resolver._discover_mcp_tools(SERVER_URL))


class TestDiscoverMcpToolsPreservesOriginalName:
    def test_schema_keys_are_sanitized(self, resolver):
        schemas = _discover(
            resolver,
            [SimpleNamespace(name="resolve-library-id", description="", inputSchema=None)],
        )

        assert set(schemas) == {"resolve_library_id"}

    def test_schema_retains_unsanitized_server_side_name(self, resolver):
        schemas = _discover(
            resolver,
            [SimpleNamespace(name="resolve-library-id", description="", inputSchema=None)],
        )

        assert schemas["resolve_library_id"]["original_name"] == "resolve-library-id"

    def test_names_needing_no_sanitization_are_unchanged(self, resolver):
        schemas = _discover(
            resolver, [SimpleNamespace(name="search", description="", inputSchema=None)]
        )

        assert schemas["search"]["original_name"] == "search"

    def test_description_is_preserved(self, resolver):
        schemas = _discover(
            resolver,
            [SimpleNamespace(name="query-docs", description="Search docs", inputSchema=None)],
        )

        assert schemas["query_docs"]["description"] == "Search docs"


class TestResolveExternalOriginalName:
    """The name sent to ``session.call_tool`` must be the server's own name."""

    def _resolve_one(self, resolver, schema):
        with patch.object(
            MCPToolResolver, "_get_mcp_tool_schemas", return_value={"resolve_library_id": schema}
        ):
            return resolver._resolve_external(SERVER_URL)

    def test_hyphenated_name_is_used_for_server_calls(self, resolver):
        tools = self._resolve_one(
            resolver,
            {
                "description": "Resolve a library id",
                "args_schema": None,
                "original_name": "resolve-library-id",
            },
        )

        assert len(tools) == 1
        assert tools[0].original_tool_name == "resolve-library-id"

    def test_agent_facing_name_stays_sanitized(self, resolver):
        """Sanitization still protects the name exposed for LLM function calling."""
        tools = self._resolve_one(
            resolver,
            {
                "description": "Resolve a library id",
                "args_schema": None,
                "original_name": "resolve-library-id",
            },
        )

        assert "-" not in tools[0].name

    def test_falls_back_to_tool_name_when_original_absent(self, resolver):
        """Schemas cached by an older version carry no ``original_name``."""
        tools = self._resolve_one(
            resolver, {"description": "Resolve a library id", "args_schema": None}
        )

        assert tools[0].original_tool_name == "resolve_library_id"

    def test_specific_tool_selector_still_matches_sanitized_name(self, resolver):
        """``url#tool`` filtering compares against sanitized keys and must keep working."""
        schema = {
            "description": "Resolve a library id",
            "args_schema": None,
            "original_name": "resolve-library-id",
        }
        with patch.object(
            MCPToolResolver,
            "_get_mcp_tool_schemas",
            return_value={"resolve_library_id": schema, "query_docs": dict(schema)},
        ):
            tools = resolver._resolve_external(f"{SERVER_URL}#resolve-library-id")

        assert len(tools) == 1
        assert tools[0].original_tool_name == "resolve-library-id"


class TestMCPToolWrapperOriginalName:
    def test_defaults_to_tool_name_when_not_supplied(self):
        from crewai.tools.mcp_tool_wrapper import MCPToolWrapper

        wrapper = MCPToolWrapper(
            mcp_server_params={"url": SERVER_URL},
            tool_name="search",
            tool_schema={"description": "d", "args_schema": None},
            server_name="example",
        )

        assert wrapper.original_tool_name == "search"

    def test_explicit_original_name_takes_precedence(self):
        from crewai.tools.mcp_tool_wrapper import MCPToolWrapper

        wrapper = MCPToolWrapper(
            mcp_server_params={"url": SERVER_URL},
            tool_name="resolve_library_id",
            tool_schema={"description": "d", "args_schema": None},
            server_name="example",
            original_tool_name="resolve-library-id",
        )

        assert wrapper.original_tool_name == "resolve-library-id"
        assert wrapper.name == "example_resolve_library_id"


class TestResolveExternalLogging:
    def test_warns_when_specific_tool_not_found(self, resolver):
        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        with patch.object(MCPToolResolver, "_get_mcp_tool_schemas", return_value={}):
            tools = resolver._resolve_external(f"{SERVER_URL}#missing-tool")

        assert tools == []
        assert any(call.args[0] == "warning" for call in mock_log.call_args_list)
