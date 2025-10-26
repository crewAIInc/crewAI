"""Tests for Agent MCP progress and headers configuration."""

from unittest.mock import Mock, patch

import pytest

from crewai.agent import Agent


class TestAgentMCPProgressConfiguration:
    """Test suite for Agent MCP progress configuration."""

    def test_agent_initialization_with_mcp_progress_enabled(self):
        """Test that Agent can be initialized with mcp_progress_enabled."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://example.com/mcp"],
            mcp_progress_enabled=True,
        )
        
        assert agent.mcp_progress_enabled is True

    def test_agent_initialization_with_mcp_progress_disabled(self):
        """Test that Agent defaults to mcp_progress_enabled=False."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://example.com/mcp"],
        )
        
        assert agent.mcp_progress_enabled is False

    def test_agent_initialization_with_mcp_server_headers(self):
        """Test that Agent can be initialized with mcp_server_headers."""
        headers = {"Authorization": "Bearer token123", "X-Client-ID": "test-client"}
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://example.com/mcp"],
            mcp_server_headers=headers,
        )
        
        assert agent.mcp_server_headers == headers

    def test_agent_initialization_without_mcp_server_headers(self):
        """Test that Agent defaults to None for mcp_server_headers."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://example.com/mcp"],
        )
        
        assert agent.mcp_server_headers is None

    def test_agent_with_both_progress_and_headers(self):
        """Test that Agent can be initialized with both progress and headers."""
        headers = {"Authorization": "Bearer token123"}
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=["https://example.com/mcp"],
            mcp_progress_enabled=True,
            mcp_server_headers=headers,
        )
        
        assert agent.mcp_progress_enabled is True
        assert agent.mcp_server_headers == headers


class TestAgentMCPToolCreation:
    """Test suite for Agent MCP tool creation with progress and headers."""

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_passes_headers(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools passes headers to server_params."""
        headers = {"Authorization": "Bearer token123"}
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_server_headers=headers,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp")
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        server_params = call_args[1]["mcp_server_params"]
        assert "headers" in server_params
        assert server_params["headers"] == headers

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_no_headers_when_not_configured(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools doesn't pass headers when not configured."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp")
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        server_params = call_args[1]["mcp_server_params"]
        assert "headers" not in server_params

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_passes_progress_callback_when_enabled(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools passes progress callback when enabled."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=True,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp")
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        assert "progress_callback" in call_args[1]
        assert call_args[1]["progress_callback"] is not None

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_no_progress_callback_when_disabled(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools doesn't pass progress callback when disabled."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=False,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp")
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        assert call_args[1]["progress_callback"] is None

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_passes_agent_context(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools passes agent context to wrapper."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=True,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp")
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        assert "agent" in call_args[1]
        assert call_args[1]["agent"] == agent

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_passes_task_context(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that _get_external_mcp_tools passes task context to wrapper."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=True,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        mock_task = Mock()
        mock_task.id = "test-task-id"
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp", task=mock_task)
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        assert "task" in call_args[1]
        assert call_args[1]["task"] == mock_task

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_get_external_mcp_tools_with_all_features(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test _get_external_mcp_tools with progress, headers, and context."""
        headers = {"Authorization": "Bearer token123"}
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=True,
            mcp_server_headers=headers,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        mock_task = Mock()
        mock_task.id = "test-task-id"
        
        tools = agent._get_external_mcp_tools("https://example.com/mcp", task=mock_task)
        
        assert mock_wrapper_class.called
        call_args = mock_wrapper_class.call_args
        
        server_params = call_args[1]["mcp_server_params"]
        assert server_params["headers"] == headers
        
        assert call_args[1]["progress_callback"] is not None
        
        assert call_args[1]["agent"] == agent
        assert call_args[1]["task"] == mock_task


class TestAgentMCPProgressCallback:
    """Test suite for Agent MCP progress callback behavior."""

    @patch("crewai.agent.Agent._get_mcp_tool_schemas")
    @patch("crewai.tools.mcp_tool_wrapper.MCPToolWrapper")
    def test_progress_callback_logs_progress(
        self, mock_wrapper_class, mock_get_schemas
    ):
        """Test that progress callback logs progress information."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcp_progress_enabled=True,
        )
        
        mock_get_schemas.return_value = {
            "test_tool": {"description": "Test tool"}
        }
        
        mock_wrapper_instance = Mock()
        mock_wrapper_class.return_value = mock_wrapper_instance
        
        with patch.object(agent._logger, "log") as mock_log:
            tools = agent._get_external_mcp_tools("https://example.com/mcp")
            
            call_args = mock_wrapper_class.call_args
            progress_callback = call_args[1]["progress_callback"]
            
            progress_callback(50.0, 100.0, "Processing...")
            
            mock_log.assert_called_once()
            log_call = mock_log.call_args
            assert log_call[0][0] == "debug"
            assert "test_tool" in log_call[0][1]
            assert "50.0" in log_call[0][1]
            assert "100.0" in log_call[0][1]
            assert "Processing..." in log_call[0][1]
