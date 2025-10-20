"""Tests for MCPToolWrapper class."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


class TestMCPToolWrapper:
    """Test suite for MCPToolWrapper class."""

    def test_tool_wrapper_creation(self):
        """Test MCPToolWrapper creation with valid parameters."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool", "args_schema": None},
            server_name="test_server"
        )

        assert wrapper.name == "test_server_test_tool"
        assert wrapper.original_tool_name == "test_tool"
        assert wrapper.server_name == "test_server"
        assert wrapper.mcp_server_params == {"url": "https://test.com/mcp"}
        assert "Test tool" in wrapper.description

    def test_tool_wrapper_creation_with_custom_description(self):
        """Test MCPToolWrapper creation with custom description."""
        custom_description = "Custom test tool for analysis"
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://api.example.com/mcp"},
            tool_name="analysis_tool",
            tool_schema={"description": custom_description, "args_schema": None},
            server_name="example_server"
        )

        assert wrapper.name == "example_server_analysis_tool"
        assert custom_description in wrapper.description

    def test_tool_wrapper_creation_without_args_schema(self):
        """Test MCPToolWrapper creation when args_schema is None."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},  # No args_schema
            server_name="test_server"
        )

        assert wrapper.name == "test_server_test_tool"

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful MCP tool execution."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock successful MCP response
        mock_result = Mock()
        mock_result.content = [Mock(text="Test result from MCP server")]

        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
             patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            result = await wrapper._run_async(query="test query")

            assert result == "Test result from MCP server"
            mock_session.initialize.assert_called_once()
            mock_session.call_tool.assert_called_once_with("test_tool", {"query": "test query"})

    @pytest.mark.asyncio
    async def test_tool_execution_timeout(self):
        """Test MCP tool execution timeout handling."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="slow_tool",
            tool_schema={"description": "Slow tool"},
            server_name="test_server"
        )

        # Mock timeout scenario
        with patch('crewai.tools.mcp_tool_wrapper.asyncio.wait_for', side_effect=asyncio.TimeoutError):
            result = await wrapper._run_async(query="test")

            assert "timed out" in result.lower()
            assert "30 seconds" in result

    @pytest.mark.asyncio
    async def test_tool_execution_connection_error(self):
        """Test MCP tool execution with connection error."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock connection error
        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client',
                   side_effect=Exception("Connection refused")):
            result = await wrapper._run_async(query="test")

            assert "failed after 3 attempts" in result.lower()
            assert "connection" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_authentication_error(self):
        """Test MCP tool execution with authentication error."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock authentication error
        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client',
                   side_effect=Exception("Authentication failed")):
            result = await wrapper._run_async(query="test")

            assert "authentication failed" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_json_parsing_error(self):
        """Test MCP tool execution with JSON parsing error."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock JSON parsing error
        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client',
                   side_effect=Exception("JSON parsing error")):
            result = await wrapper._run_async(query="test")

            assert "failed after 3 attempts" in result.lower()
            assert "parsing error" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_retry_logic(self):
        """Test MCP tool execution retry logic with exponential backoff."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        call_count = 0
        async def mock_execute_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network connection failed")
            # Success on third attempt
            mock_result = Mock()
            mock_result.content = [Mock(text="Success after retry")]
            return "Success after retry"

        with patch.object(wrapper, '_execute_tool', side_effect=mock_execute_tool):
            with patch('crewai.tools.mcp_tool_wrapper.asyncio.sleep') as mock_sleep:
                result = await wrapper._run_async(query="test")

                assert result == "Success after retry"
                assert call_count == 3
                # Verify exponential backoff sleep calls
                assert mock_sleep.call_count == 2  # 2 retries
                mock_sleep.assert_any_call(1)  # 2^0 = 1
                mock_sleep.assert_any_call(2)  # 2^1 = 2

    @pytest.mark.asyncio
    async def test_tool_execution_mcp_library_missing(self):
        """Test MCP tool execution when MCP library is missing."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock ImportError for MCP library
        with patch('crewai.tools.mcp_tool_wrapper.ClientSession', side_effect=ImportError("No module named 'mcp'")):
            result = await wrapper._run_async(query="test")

            assert "mcp library not available" in result.lower()
            assert "pip install mcp" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_various_content_formats(self):
        """Test MCP tool execution with various response content formats."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        test_cases = [
            # Content as list with text attribute
            ([Mock(text="List text content")], "List text content"),
            # Content as list without text attribute
            ([Mock(spec=[])], "Mock object"),
            # Content as string
            ("String content", "String content"),
            # No content
            (None, "Mock object"),
        ]

        for content, expected_substring in test_cases:
            mock_result = Mock()
            mock_result.content = content

            with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
                 patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()
                mock_session.call_tool = AsyncMock(return_value=mock_result)

                mock_client.return_value.__aenter__.return_value = (None, None, None)

                result = await wrapper._run_async(query="test")

                if expected_substring != "Mock object":
                    assert expected_substring in result
                else:
                    # For mock objects, just verify it's a string
                    assert isinstance(result, str)

    def test_sync_run_method(self):
        """Test the synchronous _run method wrapper."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock successful async execution
        async def mock_async_run(**kwargs):
            return "Async result"

        with patch.object(wrapper, '_run_async', side_effect=mock_async_run):
            result = wrapper._run(query="test")

            assert result == "Async result"

    def test_sync_run_method_timeout_error(self):
        """Test the synchronous _run method handling asyncio.TimeoutError."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        with patch('asyncio.run', side_effect=asyncio.TimeoutError()):
            result = wrapper._run(query="test")

            assert "test_tool" in result
            assert "timed out after 30 seconds" in result

    def test_sync_run_method_general_error(self):
        """Test the synchronous _run method handling general exceptions."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        with patch('asyncio.run', side_effect=Exception("General error")):
            result = wrapper._run(query="test")

            assert "error executing mcp tool test_tool" in result.lower()
            assert "general error" in result.lower()
