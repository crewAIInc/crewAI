"""Tests for MCP performance and timeout behavior."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent, MCP_CONNECTION_TIMEOUT, MCP_TOOL_EXECUTION_TIMEOUT, MCP_DISCOVERY_TIMEOUT
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


class TestMCPPerformanceAndTimeouts:
    """Test suite for MCP performance and timeout behavior."""

    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for performance testing."""
        return Agent(
            role="Performance Test Agent",
            goal="Test MCP performance characteristics",
            backstory="Agent designed for performance testing",
            mcps=["https://api.example.com/mcp"]
        )

    @pytest.fixture
    def performance_wrapper(self):
        """Create MCPToolWrapper for performance testing."""
        return MCPToolWrapper(
            mcp_server_params={"url": "https://performance-test.com/mcp"},
            tool_name="performance_tool",
            tool_schema={"description": "Tool for performance testing"},
            server_name="performance_server"
        )

    def test_connection_timeout_constant_value(self):
        """Test that connection timeout constant is set correctly."""
        assert MCP_CONNECTION_TIMEOUT == 10
        assert isinstance(MCP_CONNECTION_TIMEOUT, int)

    def test_tool_execution_timeout_constant_value(self):
        """Test that tool execution timeout constant is set correctly."""
        assert MCP_TOOL_EXECUTION_TIMEOUT == 30
        assert isinstance(MCP_TOOL_EXECUTION_TIMEOUT, int)

    def test_discovery_timeout_constant_value(self):
        """Test that discovery timeout constant is set correctly."""
        assert MCP_DISCOVERY_TIMEOUT == 15
        assert isinstance(MCP_DISCOVERY_TIMEOUT, int)

    @pytest.mark.asyncio
    async def test_connection_timeout_enforcement(self, performance_wrapper):
        """Test that connection timeout is properly enforced."""
        # Mock slow connection that exceeds timeout
        slow_init = AsyncMock()
        slow_init.side_effect = asyncio.sleep(MCP_CONNECTION_TIMEOUT + 5)  # Exceed timeout

        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
             patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = slow_init

            mock_client.return_value.__aenter__.return_value = (None, None, None)

            start_time = time.time()
            result = await performance_wrapper._run_async(query="test")
            elapsed_time = time.time() - start_time

            # Should timeout and not take much longer than timeout period
            assert elapsed_time < MCP_TOOL_EXECUTION_TIMEOUT + 5
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_timeout_enforcement(self, performance_wrapper):
        """Test that tool execution timeout is properly enforced."""
        # Mock slow tool execution
        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
             patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.initialize = AsyncMock()

            # Mock slow tool call
            async def slow_tool_call(*args, **kwargs):
                await asyncio.sleep(MCP_TOOL_EXECUTION_TIMEOUT + 5)  # Exceed timeout
                return Mock(content="Should not reach here")

            mock_session.call_tool = slow_tool_call
            mock_client.return_value.__aenter__.return_value = (None, None, None)

            start_time = time.time()
            result = await performance_wrapper._run_async(query="test")
            elapsed_time = time.time() - start_time

            # Should timeout within reasonable time
            assert elapsed_time < MCP_TOOL_EXECUTION_TIMEOUT + 10
            assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_discovery_timeout_enforcement(self, sample_agent):
        """Test that discovery timeout is properly enforced."""
        server_params = {"url": "https://slow-discovery.com/mcp"}

        # Mock slow discovery operation
        async def slow_discover(server_url):
            await asyncio.sleep(MCP_DISCOVERY_TIMEOUT + 5)  # Exceed timeout
            return {"tool": {"description": "Should not reach here"}}

        with patch.object(sample_agent, '_discover_mcp_tools', side_effect=slow_discover):

            start_time = time.time()

            with pytest.raises(RuntimeError, match="Failed to discover MCP tools after 3 attempts"):
                await sample_agent._get_mcp_tool_schemas_async(server_params)

            elapsed_time = time.time() - start_time

            # Should timeout within reasonable bounds (including retries)
            max_expected_time = (MCP_DISCOVERY_TIMEOUT + 5) * 3 + 10  # Retries + buffer
            assert elapsed_time < max_expected_time

    def test_cache_performance_improvement(self, sample_agent):
        """Test that caching provides significant performance improvement."""
        server_params = {"url": "https://cached-server.com/mcp"}
        mock_schemas = {"tool1": {"description": "Cached tool"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # First call - should hit server
            start_time = time.time()
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = sample_agent._get_mcp_tool_schemas(server_params)
            first_call_time = time.time() - start_time

            assert mock_async.call_count == 1
            assert schemas1 == mock_schemas

            # Second call - should use cache
            start_time = time.time()
            with patch('crewai.agent.time.time', return_value=1100):  # Within 300s TTL
                schemas2 = sample_agent._get_mcp_tool_schemas(server_params)
            second_call_time = time.time() - start_time

            # Async method should not be called again
            assert mock_async.call_count == 1
            assert schemas2 == mock_schemas

            # Second call should be much faster (cache hit)
            assert second_call_time < first_call_time / 10  # At least 10x faster

    def test_cache_ttl_expiration_behavior(self, sample_agent):
        """Test cache TTL expiration and refresh behavior."""
        server_params = {"url": "https://ttl-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "TTL test tool"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # Initial call at time 1000
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = sample_agent._get_mcp_tool_schemas(server_params)

            assert mock_async.call_count == 1

            # Call within TTL (300 seconds) - should use cache
            with patch('crewai.agent.time.time', return_value=1200):  # 200s later, within TTL
                schemas2 = sample_agent._get_mcp_tool_schemas(server_params)

            assert mock_async.call_count == 1  # No additional call

            # Call after TTL expiration - should refresh
            with patch('crewai.agent.time.time', return_value=1400):  # 400s later, beyond 300s TTL
                schemas3 = sample_agent._get_mcp_tool_schemas(server_params)

            assert mock_async.call_count == 2  # Additional call made

    def test_retry_logic_exponential_backoff_timing(self, performance_wrapper):
        """Test that retry logic uses proper exponential backoff timing."""
        failure_count = 0
        sleep_times = []

        async def mock_failing_execute(**kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception("Network connection failed")  # Retryable error
            return "Success after retries"

        async def track_sleep(seconds):
            sleep_times.append(seconds)

        with patch.object(performance_wrapper, '_execute_tool', side_effect=mock_failing_execute), \
             patch('crewai.tools.mcp_tool_wrapper.asyncio.sleep', side_effect=track_sleep):

            result = await performance_wrapper._run_async(query="test")

            assert result == "Success after retries"
            assert failure_count == 3

            # Verify exponential backoff: 2^0=1, 2^1=2
            assert sleep_times == [1, 2]

    @pytest.mark.asyncio
    async def test_concurrent_mcp_operations_performance(self, sample_agent):
        """Test performance of concurrent MCP operations."""
        server_urls = [
            "https://concurrent1.com/mcp",
            "https://concurrent2.com/mcp",
            "https://concurrent3.com/mcp",
            "https://concurrent4.com/mcp",
            "https://concurrent5.com/mcp"
        ]

        async def mock_discovery(server_url):
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return {f"tool_from_{server_url.split('//')[1].split('.')[0]}": {"description": "Concurrent tool"}}

        with patch.object(sample_agent, '_discover_mcp_tools', side_effect=mock_discovery):

            start_time = time.time()

            # Run concurrent operations
            tasks = []
            for url in server_urls:
                server_params = {"url": url}
                task = sample_agent._get_mcp_tool_schemas_async(server_params)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            elapsed_time = time.time() - start_time

            # Concurrent operations should complete faster than sequential
            # With 0.1s per operation, concurrent should be ~0.1s, sequential would be ~0.5s
            assert elapsed_time < 0.5
            assert len(results) == len(server_urls)

    def test_mcp_tool_creation_performance(self, sample_agent):
        """Test performance of MCP tool creation."""
        # Large number of tools to test scaling
        large_tool_schemas = {}
        for i in range(100):
            large_tool_schemas[f"tool_{i}"] = {"description": f"Tool {i}"}

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=large_tool_schemas), \
             patch.object(sample_agent, '_extract_server_name', return_value="test_server"):

            start_time = time.time()

            tools = sample_agent._get_external_mcp_tools("https://many-tools-server.com/mcp")

            creation_time = time.time() - start_time

            # Should create 100 tools quickly (less than 1 second)
            assert len(tools) == 100
            assert creation_time < 1.0

    def test_memory_usage_with_large_mcp_tool_sets(self, sample_agent):
        """Test memory usage with large MCP tool sets."""
        import sys

        # Create large tool schema set
        large_schemas = {}
        for i in range(1000):
            large_schemas[f"tool_{i}"] = {
                "description": f"Tool {i} with description " * 10,  # Larger descriptions
                "args_schema": None
            }

        with patch.object(sample_agent, '_get_mcp_tool_schemas', return_value=large_schemas):

            # Measure memory usage
            initial_size = sys.getsizeof(sample_agent)

            tools = sample_agent._get_external_mcp_tools("https://large-server.com/mcp")

            final_size = sys.getsizeof(sample_agent)

            # Memory usage should be reasonable
            assert len(tools) == 1000
            memory_increase = final_size - initial_size
            # Should not use excessive memory (less than 10MB increase)
            assert memory_increase < 10 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_mcp_async_operation_timing_accuracy(self, performance_wrapper):
        """Test that async MCP operations respect timing constraints accurately."""
        # Test various timeout scenarios
        timeout_tests = [
            (5, "Should complete within timeout"),
            (15, "Should complete within longer timeout"),
        ]

        for test_timeout, description in timeout_tests:
            mock_result = Mock()
            mock_result.content = [Mock(text=f"Result for {description}")]

            with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client, \
                 patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:

                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()

                # Mock tool call with controlled timing
                async def timed_call(*args, **kwargs):
                    await asyncio.sleep(test_timeout - 2)  # Complete just before timeout
                    return mock_result

                mock_session.call_tool = timed_call
                mock_client.return_value.__aenter__.return_value = (None, None, None)

                start_time = time.time()
                result = await performance_wrapper._run_async(query="test")
                elapsed_time = time.time() - start_time

                # Should complete successfully within expected timeframe
                assert description.lower() in result.lower()
                assert elapsed_time < test_timeout + 2  # Small buffer for test execution

    def test_cache_performance_under_concurrent_access(self, sample_agent):
        """Test cache performance under concurrent access."""
        server_params = {"url": "https://concurrent-cache-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Concurrent test tool"}}

        with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async, \
             patch('crewai.agent.time.time', return_value=1000):

            # First call populates cache
            sample_agent._get_mcp_tool_schemas(server_params)
            assert mock_async.call_count == 1

            # Multiple concurrent cache accesses
            with patch('crewai.agent.time.time', return_value=1100):  # Within TTL

                start_time = time.time()

                # Simulate concurrent access to cache
                for _ in range(10):
                    schemas = sample_agent._get_mcp_tool_schemas(server_params)
                    assert schemas == mock_schemas

                concurrent_time = time.time() - start_time

                # All cache hits should be very fast
                assert concurrent_time < 0.1
                assert mock_async.call_count == 1  # Should not call async method again

    def test_mcp_tool_discovery_batch_performance(self, sample_agent):
        """Test performance when discovering tools from multiple MCP servers."""
        mcps = [
            "https://server1.com/mcp",
            "https://server2.com/mcp",
            "https://server3.com/mcp",
            "https://server4.com/mcp",
            "https://server5.com/mcp"
        ]

        def mock_get_tools(mcp_ref):
            # Simulate processing time per server
            time.sleep(0.05)  # Small delay per server
            return [Mock(name=f"tool_from_{mcp_ref}")]

        with patch.object(sample_agent, '_get_external_mcp_tools', side_effect=mock_get_tools):

            start_time = time.time()

            all_tools = sample_agent.get_mcp_tools(mcps)

            batch_time = time.time() - start_time

            # Should process all servers efficiently
            assert len(all_tools) == len(mcps)
            # Should complete in reasonable time despite multiple servers
            assert batch_time < 2.0

    def test_mcp_agent_initialization_performance_impact(self):
        """Test that MCP field addition doesn't impact agent initialization performance."""
        start_time = time.time()

        # Create agents with MCP configuration
        agents = []
        for i in range(50):
            agent = Agent(
                role=f"Agent {i}",
                goal=f"Goal {i}",
                backstory=f"Backstory {i}",
                mcps=[f"https://server{i}.com/mcp"]
            )
            agents.append(agent)

        initialization_time = time.time() - start_time

        # Should initialize quickly (less than 5 seconds for 50 agents)
        assert len(agents) == 50
        assert initialization_time < 5.0

        # Each agent should have MCP configuration
        for agent in agents:
            assert hasattr(agent, 'mcps')
            assert len(agent.mcps) == 1

    @pytest.mark.asyncio
    async def test_mcp_retry_backoff_total_time_bounds(self, performance_wrapper):
        """Test that retry backoff total time stays within reasonable bounds."""
        # Mock 3 failures (max retries)
        failure_count = 0
        async def always_fail(**kwargs):
            nonlocal failure_count
            failure_count += 1
            raise Exception("Retryable network error")

        with patch.object(performance_wrapper, '_execute_tool', side_effect=always_fail), \
             patch('crewai.tools.mcp_tool_wrapper.asyncio.sleep'):  # Don't actually sleep in test

            start_time = time.time()
            result = await performance_wrapper._run_async(query="test")
            total_time = time.time() - start_time

            # Should fail after 3 attempts without excessive delay
            assert "failed after 3 attempts" in result
            assert failure_count == 3
            # Total time should be reasonable (not including actual sleep time due to patch)
            assert total_time < 1.0

    def test_mcp_cache_memory_efficiency(self, sample_agent):
        """Test that MCP cache doesn't consume excessive memory."""
        import sys

        # Get initial cache size
        from crewai.agent import _mcp_schema_cache
        initial_cache_size = sys.getsizeof(_mcp_schema_cache)

        # Add multiple cached entries
        test_servers = []
        for i in range(20):
            server_url = f"https://server{i}.com/mcp"
            test_servers.append(server_url)

            mock_schemas = {f"tool_{i}": {"description": f"Tool {i}"}}

            with patch.object(sample_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas), \
                 patch('crewai.agent.time.time', return_value=1000 + i):

                sample_agent._get_mcp_tool_schemas({"url": server_url})

        final_cache_size = sys.getsizeof(_mcp_schema_cache)
        cache_growth = final_cache_size - initial_cache_size

        # Cache should not grow excessively (less than 1MB for 20 entries)
        assert len(_mcp_schema_cache) == 20
        assert cache_growth < 1024 * 1024  # Less than 1MB

    @pytest.mark.asyncio
    async def test_mcp_operation_cancellation_handling(self, performance_wrapper):
        """Test handling of cancelled MCP operations."""
        # Mock operation that gets cancelled
        async def cancellable_operation(**kwargs):
            try:
                await asyncio.sleep(10)  # Long operation
                return "Should not complete"
            except asyncio.CancelledError:
                raise asyncio.CancelledError("Operation was cancelled")

        with patch.object(performance_wrapper, '_execute_tool', side_effect=cancellable_operation):

            # Start operation and cancel it
            task = asyncio.create_task(performance_wrapper._run_async(query="test"))
            await asyncio.sleep(0.1)  # Let it start
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                # Cancellation should be handled gracefully
                assert task.cancelled()

    def test_mcp_performance_monitoring_integration(self, sample_agent):
        """Test integration with performance monitoring systems."""
        with patch.object(sample_agent, '_logger') as mock_logger:

            # Successful operation should log info
            with patch.object(sample_agent, '_get_external_mcp_tools', return_value=[Mock()]):
                tools = sample_agent.get_mcp_tools(["https://monitored-server.com/mcp"])

                # Should log successful tool loading
                info_calls = [call for call in mock_logger.log.call_args_list if call[0][0] == "info"]
                assert len(info_calls) > 0
                assert "successfully loaded" in info_calls[0][0][1].lower()

    def test_mcp_resource_cleanup_after_operations(self, performance_wrapper):
        """Test that MCP operations clean up resources properly."""
        # This is more of a structural test since resource cleanup
        # is handled by context managers in the implementation

        with patch('crewai.tools.mcp_tool_wrapper.streamablehttp_client') as mock_client:
            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=(None, None, None))
            mock_context.__aexit__ = AsyncMock()
            mock_client.return_value = mock_context

            with patch('crewai.tools.mcp_tool_wrapper.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                mock_session.initialize = AsyncMock()
                mock_session.call_tool = AsyncMock(return_value=Mock(content="Test"))

                result = await performance_wrapper._run_async(query="test")

                # Verify context managers were properly exited
                mock_context.__aexit__.assert_called_once()

    def test_mcp_performance_baseline_establishment(self, sample_agent):
        """Establish performance baselines for MCP operations."""
        performance_metrics = {}

        # Test agent creation performance
        start = time.time()
        agent = Agent(
            role="Baseline Agent",
            goal="Establish performance baselines",
            backstory="Agent for performance baseline testing",
            mcps=["https://baseline-server.com/mcp"]
        )
        performance_metrics["agent_creation"] = time.time() - start

        # Test tool discovery performance (mocked)
        with patch.object(agent, '_get_mcp_tool_schemas', return_value={"tool1": {"description": "Baseline tool"}}):
            start = time.time()
            tools = agent._get_external_mcp_tools("https://baseline-server.com/mcp")
            performance_metrics["tool_discovery"] = time.time() - start

        # Establish reasonable performance expectations
        assert performance_metrics["agent_creation"] < 0.1  # < 100ms
        assert performance_metrics["tool_discovery"] < 0.1   # < 100ms
        assert len(tools) == 1

        # Log performance metrics for future reference
        print(f"\nMCP Performance Baselines:")
        for metric, value in performance_metrics.items():
            print(f"  {metric}: {value:.3f}s")
