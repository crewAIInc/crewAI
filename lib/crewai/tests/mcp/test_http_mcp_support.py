"""Tests for HTTP MCP server support (issue #3876)."""

import pytest
from unittest.mock import MagicMock, patch

from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP


class TestHTTPMCPValidation:
    """Test validation of HTTP MCP URLs."""

    def test_validator_accepts_http_urls(self):
        """Test that validator accepts http:// URLs."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["http://localhost:7365/mcp"],
        )
        assert agent.mcps == ["http://localhost:7365/mcp"]

    def test_validator_accepts_https_urls(self):
        """Test that validator still accepts https:// URLs."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://api.example.com/mcp"],
        )
        assert agent.mcps == ["https://api.example.com/mcp"]

    def test_validator_accepts_crewai_amp_urls(self):
        """Test that validator still accepts crewai-amp: URLs."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["crewai-amp:mcp-name"],
        )
        assert agent.mcps == ["crewai-amp:mcp-name"]

    def test_validator_accepts_http_with_fragment(self):
        """Test that validator accepts http:// URLs with #tool fragment."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["http://localhost:7365/mcp#diff_general_info"],
        )
        assert agent.mcps == ["http://localhost:7365/mcp#diff_general_info"]

    def test_validator_rejects_unsupported_schemes(self):
        """Test that validator rejects unsupported URL schemes with updated error message."""
        with pytest.raises(ValueError) as exc_info:
            Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                mcps=["ftp://example.com/mcp"],
            )
        
        error_message = str(exc_info.value)
        assert "Invalid MCP reference: ftp://example.com/mcp" in error_message
        assert "http://" in error_message
        assert "https://" in error_message
        assert "crewai-amp:" in error_message


class TestHTTPMCPRouting:
    """Test routing of HTTP MCP URLs."""

    def test_get_mcp_tools_from_string_routes_http_urls(self):
        """Test that _get_mcp_tools_from_string routes http:// URLs correctly."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )
        
        sentinel_tools = [MagicMock()]
        
        with patch.object(agent, '_get_external_mcp_tools', return_value=sentinel_tools) as mock_external:
            result = agent._get_mcp_tools_from_string("http://localhost:7365/mcp")
            
            assert result == sentinel_tools
            mock_external.assert_called_once_with("http://localhost:7365/mcp")

    def test_get_mcp_tools_from_string_routes_https_urls(self):
        """Test that _get_mcp_tools_from_string still routes https:// URLs correctly."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )
        
        sentinel_tools = [MagicMock()]
        
        with patch.object(agent, '_get_external_mcp_tools', return_value=sentinel_tools) as mock_external:
            result = agent._get_mcp_tools_from_string("https://api.example.com/mcp")
            
            assert result == sentinel_tools
            mock_external.assert_called_once_with("https://api.example.com/mcp")


class TestMCPServerHTTPConfig:
    """Test MCPServerHTTP configuration with HTTP URLs."""

    def test_mcp_server_http_accepts_http_url(self):
        """Test that MCPServerHTTP accepts http:// URLs (prevent regression)."""
        config = MCPServerHTTP(url="http://localhost:8000/mcp")
        assert config.url == "http://localhost:8000/mcp"

    def test_mcp_server_http_accepts_https_url(self):
        """Test that MCPServerHTTP still accepts https:// URLs."""
        config = MCPServerHTTP(url="https://api.example.com/mcp")
        assert config.url == "https://api.example.com/mcp"

    def test_agent_with_http_mcp_server_config(self):
        """Test that Agent accepts MCPServerHTTP with http:// URL."""
        http_config = MCPServerHTTP(url="http://localhost:8000/mcp")
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )
        
        assert agent.mcps == [http_config]


class TestHTTPMCPFragmentFiltering:
    """Test fragment filtering for HTTP MCP URLs."""

    def test_http_url_with_fragment_filters_correctly(self):
        """Test that http:// URL with #tool fragment filters correctly."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )
        
        mock_schemas = {
            "tool1": {"description": "Tool 1"},
            "tool2": {"description": "Tool 2"},
            "specific_tool": {"description": "Specific Tool"},
        }
        
        with patch.object(agent, '_get_mcp_tool_schemas', return_value=mock_schemas):
            with patch('crewai.tools.mcp_tool_wrapper.MCPToolWrapper') as mock_wrapper_class:
                mock_tool = MagicMock()
                mock_wrapper_class.return_value = mock_tool
                
                result = agent._get_external_mcp_tools("http://localhost:7365/mcp#specific_tool")
                
                mock_wrapper_class.assert_called_once()
                call_args = mock_wrapper_class.call_args
                assert call_args.kwargs['tool_name'] == 'specific_tool'
                
                assert len(result) == 1
                assert result[0] == mock_tool


class TestHTTPMCPWarningLog:
    """Test warning log for HTTP MCP URLs."""

    def test_warning_log_emitted_for_http_url(self):
        """Test that warning log is emitted when http:// is used."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            verbose=True,
        )
        
        log_calls = []
        logger_class = type(agent._logger)
        original_log = logger_class.log
        
        def mock_log(self, level, message):
            log_calls.append((level, message))
            return original_log(self, level, message)
        
        with patch.object(logger_class, 'log', new=mock_log):
            with patch.object(agent, '_get_mcp_tool_schemas', return_value={}):
                agent._get_external_mcp_tools("http://localhost:7365/mcp")
                
                warning_messages = [msg for level, msg in log_calls if level == "warning"]
                assert any("http://" in msg for msg in warning_messages)
                assert any("local development" in msg for msg in warning_messages)
                assert any("https://" in msg for msg in warning_messages)

    def test_no_warning_log_for_https_url(self):
        """Test that no warning log is emitted for https:// URLs."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            verbose=True,
        )
        
        log_calls = []
        logger_class = type(agent._logger)
        original_log = logger_class.log
        
        def mock_log(self, level, message):
            log_calls.append((level, message))
            return original_log(self, level, message)
        
        with patch.object(logger_class, 'log', new=mock_log):
            with patch.object(agent, '_get_mcp_tool_schemas', return_value={}):
                agent._get_external_mcp_tools("https://api.example.com/mcp")
                
                warning_messages = [msg for level, msg in log_calls if level == "warning"]
                assert not any("http://" in msg and "local development" in msg for msg in warning_messages)
