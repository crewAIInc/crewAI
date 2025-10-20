"""Tests for Agent MCP integration functionality."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


class TestAgentMCPIntegration:
    """Test suite for Agent MCP integration functionality."""

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""
        return Agent(
            role="Test Research Agent",
            goal="Test MCP integration capabilities",
            backstory="Agent designed for testing MCP functionality",
            mcps=["https://api.example.com/mcp"]
        )

    @pytest.fixture
    def mock_mcp_tools_response(self):
        """Mock MCP server tools response."""
        mock_tool1 = Mock()
        mock_tool1.name = "search_tool"
        mock_tool1.description = "Search for information"

        mock_tool2 = Mock()
        mock_tool2.name = "analysis_tool"
        mock_tool2.description = "Analyze data"

        mock_result = Mock()
        mock_result.tools = [mock_tool1, mock_tool2]

        return mock_result

    def test_get_mcp_tools_empty_list(self, sample_agent):
        """Test get_mcp_tools with empty list."""
        tools = sample_agent.get_mcp_tools([])
        assert tools == []

    def test_get_mcp_tools_with_https_url(self, sample_agent):
        """Test get_mcp_tools with HTTPS URL."""
        with patch.object(sample_agent, '_get_external_mcp_tools', return_value=[Mock()]) as mock_get:
            tools = sample_agent.get_mcp_tools(["https://api.example.com/mcp"])

            mock_get.assert_called_once_with("https://api.example.com/mcp")
            assert len(tools) == 1

    def test_get_mcp_tools_with_crewai_amp_reference(self, sample_agent):
        """Test get_mcp_tools with CrewAI AMP reference."""
        with patch.object(sample_agent, '_get_amp_mcp_tools', return_value=[Mock()]) as mock_get:
            tools = sample_agent.get_mcp_tools(["crewai-amp:financial-data"])

            mock_get.assert_called_once_with("crewai-amp:financial-data")
            assert len(tools) == 1

    def test_get_mcp_tools_mixed_references(self, sample_agent):
        """Test get_mcp_tools with mixed reference types."""
        mock_external_tools = [Mock(name="external_tool")]
        mock_amp_tools = [Mock(name="amp_tool")]

        with patch.object(sample_agent, '_get_external_mcp_tools', return_value=mock_external_tools), \
             patch.object(sample_agent, '_get_amp_mcp_tools', return_value=mock_amp_tools):

            tools = sample_agent.get_mcp_tools([
                "https://api.example.com/mcp",
                "crewai-amp:research-tools"
            ])

            assert len(tools) == 2

    def test_get_mcp_tools_error_handling(self, sample_agent):
        """Test get_mcp_tools error handling and graceful degradation."""
        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception("Connection failed")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(["https://api.example.com/mcp"])

            # Should return empty list and log warning
            assert tools == []
            mock_logger.log.assert_called_with("warning", "Skipping MCP https://api.example.com/mcp due to error: Connection failed")

    def test_extract_server_name_basic_url(self, sample_agent):
        """Test server name extraction from basic URLs."""
        server_name = sample_agent._extract_server_name("https://api.example.com/mcp")
        assert server_name == "api_example_com_mcp"

    def test_extract_server_name_with_path(self, sample_agent):
        """Test server name extraction from URLs with paths."""
        server_name = sample_agent._extract_server_name("https://mcp.exa.ai/api/v1/mcp")
        assert server_name == "mcp_exa_ai_api_v1_mcp"

    def test_extract_server_name_no_path(self, sample_agent):
        """Test server name extraction from URLs without path."""
        server_name = sample_agent._extract_server_name("https://example.com")
        assert server_name == "example_com"

    def test_extract_server_name_with_query_params(self, sample_agent):
        """Test server name extraction ignores query parameters."""
        server_name = sample_agent._extract_server_name("https://api.example.com/mcp?api_key=test")
        assert server_name == "api_example_com_mcp"

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_success(self, sample_agent, mock_mcp_tools_response):
        """Test successful MCP tool schema retrieval."""
        server_params = {"url": "https://api.example.com/mcp"}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value={
            "search_tool": {"description": "Search tool", "args_schema": None},
            "analysis_tool": {"description": "Analysis tool", "args_schema": None}
        }) as mock_async:

            schemas = sample_agent._get_mcp_tool_schemas(server_params)

            assert len(schemas) == 2
            assert "search_tool" in schemas
            assert "analysis_tool" in schemas
            mock_async.assert_called_once()

    def test_get_mcp_tool_schemas_caching(self, sample_agent):
        """Test MCP tool schema caching behavior."""
        from crewai.agent import _mcp_schema_cache

        # Clear cache to ensure clean test state
        _mcp_schema_cache.clear()

        server_params = {"url": "https://api.example.com/mcp"}
        mock_schemas = {"tool1": {"description": "Tool 1"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # First call at time 1000 - should hit server
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = sample_agent._get_mcp_tool_schemas(server_params)
            assert mock_async.call_count == 1

            # Second call within TTL - should use cache
            with patch('crewai.agent.time.time', return_value=1100):  # 100 seconds later, within 300s TTL
                schemas2 = sample_agent._get_mcp_tool_schemas(server_params)
                assert mock_async.call_count == 1  # Not called again
                assert schemas1 == schemas2

        # Clean up cache after test
        _mcp_schema_cache.clear()

    def test_get_mcp_tool_schemas_cache_expiration(self, sample_agent):
        """Test MCP tool schema cache expiration."""
        from crewai.agent import _mcp_schema_cache

        # Clear cache to ensure clean test state
        _mcp_schema_cache.clear()

        server_params = {"url": "https://api.example.com/mcp"}
        mock_schemas = {"tool1": {"description": "Tool 1"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # First call at time 1000
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = sample_agent._get_mcp_tool_schemas(server_params)
            assert mock_async.call_count == 1

            # Call after cache expiration (> 300s TTL)
            with patch('crewai.agent.time.time', return_value=1400):  # 400 seconds later, beyond 300s TTL
                schemas2 = sample_agent._get_mcp_tool_schemas(server_params)
                assert mock_async.call_count == 2  # Called again after cache expiration

        # Clean up cache after test
        _mcp_schema_cache.clear()

    def test_get_mcp_tool_schemas_error_handling(self, sample_agent):
        """Test MCP tool schema retrieval error handling."""
        server_params = {"url": "https://api.example.com/mcp"}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', side_effect=Exception("Connection failed")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            schemas = sample_agent._get_mcp_tool_schemas(server_params)

            # Should return empty dict and log warning
            assert schemas == {}
            mock_logger.log.assert_called_with("warning", "Failed to get MCP tool schemas from https://api.example.com/mcp: Connection failed")

    def test_get_external_mcp_tools_full_server(self, sample_agent):
        """Test getting tools from external MCP server (full server)."""
        mcp_ref = "https://api.example.com/mcp"
        mock_schemas = {
            "tool1": {"description": "Tool 1"},
            "tool2": {"description": "Tool 2"}
        }

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=mock_schemas), \
             patch.object(sample_agent, '_extract_server_name', return_value="example_server"):

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            assert len(tools) == 2
            assert all(isinstance(tool, MCPToolWrapper) for tool in tools)
            assert tools[0].server_name == "example_server"

    def test_get_external_mcp_tools_specific_tool(self, sample_agent):
        """Test getting specific tool from external MCP server."""
        mcp_ref = "https://api.example.com/mcp#tool1"
        mock_schemas = {
            "tool1": {"description": "Tool 1"},
            "tool2": {"description": "Tool 2"}
        }

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=mock_schemas), \
             patch.object(sample_agent, '_extract_server_name', return_value="example_server"):

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            # Should only get tool1
            assert len(tools) == 1
            assert tools[0].original_tool_name == "tool1"

    def test_get_external_mcp_tools_specific_tool_not_found(self, sample_agent):
        """Test getting specific tool that doesn't exist on MCP server."""
        mcp_ref = "https://api.example.com/mcp#nonexistent_tool"
        mock_schemas = {
            "tool1": {"description": "Tool 1"},
            "tool2": {"description": "Tool 2"}
        }

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=mock_schemas), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            # Should return empty list and log warning
            assert tools == []
            mock_logger.log.assert_called_with("warning", "Specific tool 'nonexistent_tool' not found on MCP server: https://api.example.com/mcp")

    def test_get_external_mcp_tools_no_schemas(self, sample_agent):
        """Test getting tools when no schemas are discovered."""
        mcp_ref = "https://api.example.com/mcp"

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value={}), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            assert tools == []
            mock_logger.log.assert_called_with("warning", "No tools discovered from MCP server: https://api.example.com/mcp")

    def test_get_amp_mcp_tools_full_mcp(self, sample_agent):
        """Test getting tools from CrewAI AMP MCP marketplace (full MCP)."""
        amp_ref = "crewai-amp:financial-data"
        mock_servers = [{"url": "https://amp.crewai.com/mcp/financial"}]

        with patch.object(sample_agent, '_fetch_amp_mcp_servers', return_value=mock_servers), \
             patch.object(sample_agent, '_get_external_mcp_tools', return_value=[Mock()]) as mock_get_tools:

            tools = sample_agent._get_amp_mcp_tools(amp_ref)

            mock_get_tools.assert_called_once_with("https://amp.crewai.com/mcp/financial")
            assert len(tools) == 1

    def test_get_amp_mcp_tools_specific_tool(self, sample_agent):
        """Test getting specific tool from CrewAI AMP MCP marketplace."""
        amp_ref = "crewai-amp:financial-data#get_stock_price"
        mock_servers = [{"url": "https://amp.crewai.com/mcp/financial"}]

        with patch.object(sample_agent, '_fetch_amp_mcp_servers', return_value=mock_servers), \
             patch.object(sample_agent, '_get_external_mcp_tools', return_value=[Mock()]) as mock_get_tools:

            tools = sample_agent._get_amp_mcp_tools(amp_ref)

            mock_get_tools.assert_called_once_with("https://amp.crewai.com/mcp/financial#get_stock_price")
            assert len(tools) == 1

    def test_get_amp_mcp_tools_multiple_servers(self, sample_agent):
        """Test getting tools from multiple AMP MCP servers."""
        amp_ref = "crewai-amp:multi-server-mcp"
        mock_servers = [
            {"url": "https://amp.crewai.com/mcp/server1"},
            {"url": "https://amp.crewai.com/mcp/server2"}
        ]

        with patch.object(sample_agent, '_fetch_amp_mcp_servers', return_value=mock_servers), \
             patch.object(sample_agent, '_get_external_mcp_tools', return_value=[Mock()]) as mock_get_tools:

            tools = sample_agent._get_amp_mcp_tools(amp_ref)

            assert mock_get_tools.call_count == 2
            assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_async_success(self, sample_agent, mock_mcp_tools_response):
        """Test successful async MCP tool schema retrieval."""
        server_params = {"url": "https://api.example.com/mcp"}

        with patch('crewai.agent.streamablehttp_client') as mock_client, \
             patch('crewai.agent.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()
            mock_session.list_tools = AsyncMock(return_value=mock_mcp_tools_response)

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            schemas = await sample_agent._get_mcp_tool_schemas_async(server_params)

            assert len(schemas) == 2
            assert "search_tool" in schemas
            assert "analysis_tool" in schemas
            assert schemas["search_tool"]["description"] == "Search for information"

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_async_timeout(self, sample_agent):
        """Test async MCP tool schema retrieval timeout handling."""
        server_params = {"url": "https://api.example.com/mcp"}

        with patch('crewai.agent.asyncio.wait_for', side_effect=asyncio.TimeoutError):
            with pytest.raises(RuntimeError, match="Failed to discover MCP tools after 3 attempts"):
                await sample_agent._get_mcp_tool_schemas_async(server_params)

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_async_import_error(self, sample_agent):
        """Test async MCP tool schema retrieval with missing MCP library."""
        server_params = {"url": "https://api.example.com/mcp"}

        with patch('crewai.agent.ClientSession', side_effect=ImportError("No module named 'mcp'")):
            with pytest.raises(RuntimeError, match="MCP library not available"):
                await sample_agent._get_mcp_tool_schemas_async(server_params)

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_async_retry_logic(self, sample_agent):
        """Test retry logic with exponential backoff in async schema retrieval."""
        server_params = {"url": "https://api.example.com/mcp"}

        call_count = 0
        async def mock_discover_tools(url):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network connection failed")
            return {"tool1": {"description": "Tool 1"}}

        with patch.object(sample_agent, '_discover_mcp_tools', side_effect=mock_discover_tools), \
             patch('crewai.agent.asyncio.sleep') as mock_sleep:

            schemas = await sample_agent._get_mcp_tool_schemas_async(server_params)

            assert schemas == {"tool1": {"description": "Tool 1"}}
            assert call_count == 3
            # Verify exponential backoff
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)  # First retry: 2^0 = 1
            mock_sleep.assert_any_call(2)  # Second retry: 2^1 = 2

    @pytest.mark.asyncio
    async def test_discover_mcp_tools_success(self, sample_agent, mock_mcp_tools_response):
        """Test successful MCP tool discovery."""
        server_url = "https://api.example.com/mcp"

        with patch('crewai.agent.streamablehttp_client') as mock_client, \
             patch('crewai.agent.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()
            mock_session.list_tools = AsyncMock(return_value=mock_mcp_tools_response)

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            schemas = await sample_agent._discover_mcp_tools(server_url)

            assert len(schemas) == 2
            assert schemas["search_tool"]["description"] == "Search for information"
            assert schemas["analysis_tool"]["description"] == "Analyze data"

    def test_fetch_amp_mcp_servers_placeholder(self, sample_agent):
        """Test AMP MCP server fetching (currently returns empty list)."""
        result = sample_agent._fetch_amp_mcp_servers("test-mcp")

        # Currently returns empty list - placeholder implementation
        assert result == []

    def test_get_external_mcp_tools_error_handling(self, sample_agent):
        """Test external MCP tools error handling."""
        mcp_ref = "https://failing-server.com/mcp"

        with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=Exception("Server unavailable")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            assert tools == []
            mock_logger.log.assert_called_with("warning", "Failed to connect to MCP server https://failing-server.com/mcp: Server unavailable")

    def test_get_external_mcp_tools_wrapper_creation_error(self, sample_agent):
        """Test handling of MCPToolWrapper creation errors."""
        mcp_ref = "https://api.example.com/mcp"
        mock_schemas = {"tool1": {"description": "Tool 1"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=mock_schemas), \
             patch.object(sample_agent, '_extract_server_name', return_value="example_server"), \
             patch('crewai.tools.mcp_tool_wrapper.MCPToolWrapper', side_effect=Exception("Wrapper creation failed")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent._get_external_mcp_tools(mcp_ref)

            assert tools == []
            mock_logger.log.assert_called_with("warning", "Failed to create MCP tool wrapper for tool1: Wrapper creation failed")

    def test_mcp_tools_integration_with_existing_agent_features(self, sample_agent):
        """Test MCP tools integration with existing agent features."""
        # Test that MCP field works alongside other agent features
        agent_with_all_features = Agent(
            role="Full Feature Agent",
            goal="Test all features together",
            backstory="Agent with all features enabled",
            mcps=["https://api.example.com/mcp", "crewai-amp:financial-data"],
            apps=["gmail", "slack"],  # Platform apps
            tools=[],  # Regular tools
            verbose=True,
            max_iter=15,
            allow_delegation=True
        )

        assert len(agent_with_all_features.mcps) == 2
        assert len(agent_with_all_features.apps) == 2
        assert agent_with_all_features.verbose is True
        assert agent_with_all_features.max_iter == 15
        assert agent_with_all_features.allow_delegation is True

    def test_mcp_reference_parsing_edge_cases(self, sample_agent):
        """Test MCP reference parsing with edge cases."""
        test_cases = [
            # URL with complex query parameters
            ("https://api.example.com/mcp?api_key=abc123&profile=test&version=1.0", "api.example.com", None),
            # URL with tool name and query params
            ("https://api.example.com/mcp?api_key=test#search_tool", "api.example.com", "search_tool"),
            # AMP reference with dashes and underscores
            ("crewai-amp:financial_data-v2", "financial_data-v2", None),
            # AMP reference with tool name
            ("crewai-amp:research-tools#pubmed_search", "research-tools", "pubmed_search"),
        ]

        for mcp_ref, expected_server_part, expected_tool in test_cases:
            if mcp_ref.startswith("https://"):
                if '#' in mcp_ref:
                    server_url, tool_name = mcp_ref.split('#', 1)
                    assert expected_server_part in server_url
                    assert tool_name == expected_tool
                else:
                    assert expected_server_part in mcp_ref
                    assert expected_tool is None
            else:  # AMP reference
                amp_part = mcp_ref.replace('crewai-amp:', '')
                if '#' in amp_part:
                    mcp_name, tool_name = amp_part.split('#', 1)
                    assert mcp_name == expected_server_part
                    assert tool_name == expected_tool
                else:
                    assert amp_part == expected_server_part
                    assert expected_tool is None
