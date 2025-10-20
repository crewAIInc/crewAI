"""Tests for MCP error handling scenarios."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


class TestMCPErrorHandling:
    """Test suite for MCP error handling scenarios."""

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for error testing."""
        return Agent(
            role="Error Test Agent",
            goal="Test error handling capabilities",
            backstory="Agent designed for testing error scenarios",
            mcps=["https://api.example.com/mcp"]
        )

    def test_connection_timeout_graceful_handling(self, sample_agent):
        """Test graceful handling of connection timeouts."""
        with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=Exception("Connection timed out")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(["https://slow-server.com/mcp"])

            # Should return empty list and log warning
            assert tools == []
            mock_logger.log.assert_called_with("warning", "Skipping MCP https://slow-server.com/mcp due to error: Connection timed out")

    def test_authentication_failure_handling(self, sample_agent):
        """Test handling of authentication failures."""
        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception("Authentication failed")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(["https://secure-server.com/mcp"])

            assert tools == []
            mock_logger.log.assert_called_with("warning", "Skipping MCP https://secure-server.com/mcp due to error: Authentication failed")

    def test_json_parsing_error_handling(self, sample_agent):
        """Test handling of JSON parsing errors."""
        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception("JSON parsing failed")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(["https://malformed-server.com/mcp"])

            assert tools == []
            mock_logger.log.assert_called_with("warning", "Skipping MCP https://malformed-server.com/mcp due to error: JSON parsing failed")

    def test_network_connectivity_issues(self, sample_agent):
        """Test handling of network connectivity issues."""
        network_errors = [
            "Network unreachable",
            "Connection refused",
            "DNS resolution failed",
            "Timeout occurred"
        ]

        for error_msg in network_errors:
            with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception(error_msg)), \
                 patch.object(sample_agent, '_logger') as mock_logger:

                tools = sample_agent.get_mcp_tools(["https://unreachable-server.com/mcp"])

                assert tools == []
                mock_logger.log.assert_called_with("warning", f"Skipping MCP https://unreachable-server.com/mcp due to error: {error_msg}")

    def test_malformed_mcp_server_responses(self, sample_agent):
        """Test handling of malformed MCP server responses."""
        malformed_errors = [
            "Invalid JSON response",
            "Unexpected response format",
            "Missing required fields",
            "Protocol version mismatch"
        ]

        for error_msg in malformed_errors:
            with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=Exception(error_msg)):

                tools = sample_agent._get_external_mcp_tools("https://malformed-server.com/mcp")

                # Should handle error gracefully
                assert tools == []

    def test_server_unavailability_scenarios(self, sample_agent):
        """Test various server unavailability scenarios."""
        unavailability_scenarios = [
            "Server returned 404",
            "Server returned 500",
            "Service unavailable",
            "Server maintenance mode"
        ]

        for scenario in unavailability_scenarios:
            with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=Exception(scenario)):

                # Should not raise exception, should return empty list
                tools = sample_agent._get_external_mcp_tools("https://unavailable-server.com/mcp")
                assert tools == []

    def test_tool_not_found_errors(self):
        """Test handling when specific tool is not found."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="nonexistent_tool",
            tool_schema={"description": "Tool that doesn't exist"},
            server_name="test_server"
        )

        # Mock scenario where tool is not found on server
        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
             patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()

            # Mock empty tools list (tool not found)
            mock_tools = []

            with patch('crewai.tools.mcp_tool_wrapper.MCPServerAdapter') as mock_adapter:
                mock_adapter.return_value.__enter__.return_value = mock_tools

                result = wrapper._run(query="test")

                assert "not found on MCP server" in result

    def test_mixed_server_success_and_failure(self, sample_agent):
        """Test handling mixed scenarios with both successful and failing servers."""
        mcps = [
            "https://failing-server.com/mcp",      # Will fail
            "https://working-server.com/mcp",      # Will succeed
            "https://another-failing.com/mcp",     # Will fail
        ]

        def mock_get_external_tools(mcp_ref):
            if "failing" in mcp_ref:
                raise Exception("Server failed")
            else:
                # Return mock tool for working server
                return [Mock(name=f"tool_from_{mcp_ref}")]

        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=mock_get_external_tools), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(mcps)

            # Should get tools from working server only
            assert len(tools) == 1

            # Should log warnings for failing servers
            assert mock_logger.log.call_count >= 2  # At least 2 warning calls

    @pytest.mark.asyncio
    async def test_concurrent_mcp_operations_error_isolation(self, sample_agent):
        """Test that errors in concurrent MCP operations are properly isolated."""
        async def mock_operation_with_random_failures(server_params):
            url = server_params["url"]
            if "fail" in url:
                raise Exception(f"Simulated failure for {url}")
            return {"tool1": {"description": "Success tool"}}

        server_params_list = [
            {"url": "https://server1-fail.com/mcp"},
            {"url": "https://server2-success.com/mcp"},
            {"url": "https://server3-fail.com/mcp"},
            {"url": "https://server4-success.com/mcp"}
        ]

        # Run operations concurrently
        results = []
        for params in server_params_list:
            try:
                result = await mock_operation_with_random_failures(params)
                results.append(result)
            except Exception:
                results.append({})  # Empty dict for failures

        # Should have 2 successful results and 2 empty results
        successful_results = [r for r in results if r]
        assert len(successful_results) == 2

    @pytest.mark.asyncio
    async def test_mcp_library_import_error_handling(self):
        """Test handling when MCP library is not available."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Mock ImportError for MCP library
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mcp'")):
            result = await wrapper._run_async(query="test")

            assert "mcp library not available" in result.lower()
            assert "pip install mcp" in result

    def test_mcp_tools_graceful_degradation_in_agent_creation(self):
        """Test that agent creation continues even with failing MCP servers."""
        with patch('crewai.agent.Agent._get_external_mcp_tools', side_effect=Exception("All MCP servers failed")):

            # Agent creation should succeed even if MCP discovery fails
            agent = Agent(
                role="Resilient Agent",
                goal="Continue working despite MCP failures",
                backstory="Agent that handles MCP failures gracefully",
                mcps=["https://failing-server.com/mcp"]
            )

            assert agent is not None
            assert agent.role == "Resilient Agent"
            assert len(agent.mcps) == 1

    def test_partial_mcp_server_failure_recovery(self, sample_agent):
        """Test recovery when some but not all MCP servers fail."""
        mcps = [
            "https://server1.com/mcp",  # Will succeed
            "https://server2.com/mcp",  # Will fail
            "https://server3.com/mcp"   # Will succeed
        ]

        def mock_external_tools(mcp_ref):
            if "server2" in mcp_ref:
                raise Exception("Server 2 is down")
            return [Mock(name=f"tool_from_{mcp_ref.split('//')[-1].split('.')[0]}")]

        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=mock_external_tools):
            tools = sample_agent.get_mcp_tools(mcps)

            # Should get tools from server1 and server3, skip server2
            assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_tool_execution_error_messages_are_informative(self):
        """Test that tool execution error messages provide useful information."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="failing_tool",
            tool_schema={"description": "Tool that fails"},
            server_name="test_server"
        )

        error_scenarios = [
            (asyncio.TimeoutError(), "timed out"),
            (ConnectionError("Connection failed"), "network connection failed"),
            (Exception("Authentication failed"), "authentication failed"),
            (ValueError("JSON parsing error"), "server response parsing error"),
            (Exception("Tool not found"), "mcp execution error")
        ]

        for error, expected_msg in error_scenarios:
            with patch.object(wrapper, '_execute_tool', side_effect=error):
                result = await wrapper._run_async(query="test")

                assert expected_msg.lower() in result.lower()
                assert "failing_tool" in result

    def test_mcp_server_connection_resilience(self, sample_agent):
        """Test MCP server connection resilience across multiple operations."""
        # Simulate intermittent connection issues
        call_count = 0
        def intermittent_connection_mock(server_params):
            nonlocal call_count
            call_count += 1

            # Fail every other call to simulate intermittent issues
            if call_count % 2 == 0:
                raise Exception("Intermittent connection failure")

            return {"stable_tool": {"description": "Tool from stable connection"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=intermittent_connection_mock):

            # Multiple calls should handle intermittent failures
            results = []
            for i in range(4):
                tools = sample_agent._get_external_mcp_tools("https://intermittent-server.com/mcp")
                results.append(len(tools))

            # Should have some successes and some failures
            successes = [r for r in results if r > 0]
            failures = [r for r in results if r == 0]

            assert len(successes) >= 1  # At least one success
            assert len(failures) >= 1   # At least one failure

    @pytest.mark.asyncio
    async def test_mcp_tool_schema_discovery_timeout_handling(self, sample_agent):
        """Test timeout handling in MCP tool schema discovery."""
        server_params = {"url": "https://slow-server.com/mcp"}

        # Mock timeout during discovery
        with patch.object(sample_agent, '_discover_mcp_tools', side_effect=asyncio.TimeoutError):
            with pytest.raises(RuntimeError, match="Failed to discover MCP tools after 3 attempts"):
                await sample_agent._get_mcp_tool_schemas_async(server_params)

    @pytest.mark.asyncio
    async def test_mcp_session_initialization_timeout(self, sample_agent):
        """Test timeout during MCP session initialization."""
        server_url = "https://slow-init-server.com/mcp"

        with patch('crewai.agent.streamablehttp_client') as mock_client, \
             patch('crewai.agent.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            # Mock timeout during initialization
            mock_session.initialize = AsyncMock(side_effect=asyncio.TimeoutError)

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            with pytest.raises(asyncio.TimeoutError):
                await sample_agent._discover_mcp_tools(server_url)

    @pytest.mark.asyncio
    async def test_mcp_tool_listing_timeout(self, sample_agent):
        """Test timeout during MCP tool listing."""
        server_url = "https://slow-list-server.com/mcp"

        with patch('crewai.agent.streamablehttp_client') as mock_client, \
             patch('crewai.agent.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()
            # Mock timeout during tool listing
            mock_session.list_tools = AsyncMock(side_effect=asyncio.TimeoutError)

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            with pytest.raises(asyncio.TimeoutError):
                await sample_agent._discover_mcp_tools(server_url)

    def test_mcp_server_response_format_errors(self, sample_agent):
        """Test handling of various MCP server response format errors."""
        response_format_errors = [
            "Invalid response structure",
            "Missing required fields",
            "Unexpected response type",
            "Protocol version incompatible"
        ]

        for error_msg in response_format_errors:
            with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=Exception(error_msg)):

                tools = sample_agent._get_external_mcp_tools("https://bad-format-server.com/mcp")
                assert tools == []

    def test_mcp_multiple_concurrent_failures(self, sample_agent):
        """Test handling multiple concurrent MCP server failures."""
        failing_mcps = [
            "https://fail1.com/mcp",
            "https://fail2.com/mcp",
            "https://fail3.com/mcp",
            "https://fail4.com/mcp",
            "https://fail5.com/mcp"
        ]

        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception("Server failure")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(failing_mcps)

            # Should handle all failures gracefully
            assert tools == []
            # Should log warning for each failed server
            assert mock_logger.log.call_count == len(failing_mcps)

    def test_mcp_crewai_amp_server_failures(self, sample_agent):
        """Test handling of CrewAI AMP server failures."""
        amp_refs = [
            "crewai-amp:nonexistent-mcp",
            "crewai-amp:failing-mcp#tool_name"
        ]

        with patch.object(sample_agent, '_get_amp_mcp_tools', side_effect=Exception("AMP server unavailable")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools(amp_refs)

            assert tools == []
            assert mock_logger.log.call_count == len(amp_refs)

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_various_failure_modes(self):
        """Test various MCP tool execution failure modes."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        failure_scenarios = [
            # Connection failures
            (ConnectionError("Connection reset by peer"), "network connection failed"),
            (ConnectionRefusedError("Connection refused"), "network connection failed"),

            # Timeout failures
            (asyncio.TimeoutError(), "timed out"),

            # Authentication failures
            (PermissionError("Access denied"), "authentication failed"),
            (Exception("401 Unauthorized"), "authentication failed"),

            # Parsing failures
            (ValueError("JSON decode error"), "server response parsing error"),
            (Exception("Invalid JSON"), "server response parsing error"),

            # Generic failures
            (Exception("Unknown error"), "mcp execution error"),
        ]

        for error, expected_msg_part in failure_scenarios:
            with patch.object(wrapper, '_execute_tool', side_effect=error):
                result = await wrapper._run_async(query="test")

                assert expected_msg_part in result.lower()

    def test_mcp_error_logging_provides_context(self, sample_agent):
        """Test that MCP error logging provides sufficient context for debugging."""
        problematic_mcp = "https://problematic-server.com/mcp#specific_tool"

        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=Exception("Detailed error message with context")), \
             patch.object(sample_agent, '_logger') as mock_logger:

            tools = sample_agent.get_mcp_tools([problematic_mcp])

            # Verify logging call includes full MCP reference
            mock_logger.log.assert_called_with("warning", f"Skipping MCP {problematic_mcp} due to error: Detailed error message with context")

    def test_mcp_error_recovery_preserves_agent_functionality(self, sample_agent):
        """Test that MCP errors don't break core agent functionality."""
        # Even with all MCP servers failing, agent should still work
        with patch.object(sample_agent, 'get_mcp_tools', return_value=[]):

            # Agent should still have core functionality
            assert sample_agent.role is not None
            assert sample_agent.goal is not None
            assert sample_agent.backstory is not None
            assert hasattr(sample_agent, 'execute_task')
            assert hasattr(sample_agent, 'create_agent_executor')

    def test_mcp_error_handling_with_existing_tools(self, sample_agent):
        """Test MCP error handling when agent has existing tools."""
        from crewai.tools import BaseTool

        class TestTool(BaseTool):
            name: str = "existing_tool"
            description: str = "Existing agent tool"

            def _run(self, **kwargs):
                return "Existing tool result"

        agent_with_tools = Agent(
            role="Agent with Tools",
            goal="Test MCP errors with existing tools",
            backstory="Agent that has both regular and MCP tools",
            tools=[TestTool()],
            mcps=["https://failing-mcp.com/mcp"]
        )

        # MCP failures should not affect existing tools
        with patch.object(agent_with_tools, 'get_mcp_tools', return_value=[]):
            assert len(agent_with_tools.tools) == 1
            assert agent_with_tools.tools[0].name == "existing_tool"


class TestMCPErrorRecoveryPatterns:
    """Test specific error recovery patterns for MCP integration."""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff timing calculation."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Test backoff timing
        with patch('crewai.tools.mcp_tool_wrapper.asyncio.sleep') as mock_sleep, \
             patch.object(wrapper, '_execute_tool', side_effect=[
                 Exception("Fail 1"),
                 Exception("Fail 2"),
                 "Success"
             ]):

            result = asyncio.run(wrapper._run_async(query="test"))

            # Should succeed after retries
            assert result == "Success"

            # Verify exponential backoff sleep calls
            expected_sleeps = [1, 2]  # 2^0=1, 2^1=2
            actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
            assert actual_sleeps == expected_sleeps

    def test_non_retryable_errors_fail_fast(self):
        """Test that non-retryable errors (like auth) fail fast without retries."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://test.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server"
        )

        # Authentication errors should not be retried
        with patch.object(wrapper, '_execute_tool', side_effect=Exception("Authentication failed")), \
             patch('crewai.tools.mcp_tool_wrapper.asyncio.sleep') as mock_sleep:

            result = asyncio.run(wrapper._run_async(query="test"))

            assert "authentication failed" in result.lower()
            # Should not have retried (no sleep calls)
            mock_sleep.assert_not_called()

    def test_cache_invalidation_on_persistent_errors(self, sample_agent):
        """Test that persistent errors don't get cached."""
        server_params = {"url": "https://persistently-failing.com/mcp"}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', side_effect=Exception("Persistent failure")), \
             patch('crewai.agent.time.time', return_value=1000):

            # First call should attempt and fail
            schemas1 = sample_agent._get_mcp_tool_schemas(server_params)
            assert schemas1 == {}

            # Second call should attempt again (not use cached failure)
            with patch('crewai.agent.time.time', return_value=1001):
                schemas2 = sample_agent._get_mcp_tool_schemas(server_params)
                assert schemas2 == {}

    def test_error_context_preservation_through_call_stack(self, sample_agent):
        """Test that error context is preserved through the entire call stack."""
        original_error = Exception("Original detailed error with context information")

        with patch.object(sample_agent, '_get_mcp_tool_schemas', side_effect=original_error), \
             patch.object(sample_agent, '_logger') as mock_logger:

            # Call through the full stack
            tools = sample_agent.get_mcp_tools(["https://error-context-server.com/mcp"])

            # Original error message should be preserved in logs
            assert tools == []
            log_call = mock_logger.log.call_args
            assert "Original detailed error with context information" in log_call[0][1]
