"""Tests for MCP caching functionality."""

import time
import pytest
from unittest.mock import Mock, patch

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent, _mcp_schema_cache, _cache_ttl


class TestMCPCaching:
    """Test suite for MCP caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        _mcp_schema_cache.clear()

    def teardown_method(self):
        """Clear cache after each test."""
        _mcp_schema_cache.clear()

    @pytest.fixture
    def caching_agent(self):
        """Create agent for caching tests."""
        return Agent(
            role="Caching Test Agent",
            goal="Test MCP caching behavior",
            backstory="Agent designed for testing cache functionality",
            mcps=["https://cache-test.com/mcp"]
        )

    def test_cache_initially_empty(self):
        """Test that MCP schema cache starts empty."""
        assert len(_mcp_schema_cache) == 0

    def test_cache_ttl_constant(self):
        """Test that cache TTL is set to expected value."""
        assert _cache_ttl == 300  # 5 minutes

    def test_cache_population_on_first_access(self, caching_agent):
        """Test that cache gets populated on first schema access."""
        server_params = {"url": "https://cache-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Cached tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas), \
             patch('crewai.agent.time.time', return_value=1000):

            # Cache should be empty initially
            assert len(_mcp_schema_cache) == 0

            # First call should populate cache
            schemas = caching_agent._get_mcp_tool_schemas(server_params)

            assert schemas == mock_schemas
            assert len(_mcp_schema_cache) == 1
            assert "https://cache-test.com/mcp" in _mcp_schema_cache

    def test_cache_hit_returns_cached_data(self, caching_agent):
        """Test that cache hit returns previously cached data."""
        server_params = {"url": "https://cache-hit-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Cache hit tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # First call - populates cache
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = caching_agent._get_mcp_tool_schemas(server_params)

            # Second call - should use cache
            with patch('crewai.agent.time.time', return_value=1150):  # 150s later, within TTL
                schemas2 = caching_agent._get_mcp_tool_schemas(server_params)

            assert schemas1 == schemas2 == mock_schemas
            assert mock_async.call_count == 1  # Only called once

    def test_cache_miss_after_ttl_expiration(self, caching_agent):
        """Test that cache miss occurs after TTL expiration."""
        server_params = {"url": "https://cache-expiry-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Expiry test tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # First call at time 1000
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = caching_agent._get_mcp_tool_schemas(server_params)

            # Call after TTL expiration (300s + buffer)
            with patch('crewai.agent.time.time', return_value=1400):  # 400s later, beyond TTL
                schemas2 = caching_agent._get_mcp_tool_schemas(server_params)

            assert schemas1 == schemas2 == mock_schemas
            assert mock_async.call_count == 2  # Called twice due to expiration

    def test_cache_key_generation(self, caching_agent):
        """Test that cache keys are generated correctly."""
        different_urls = [
            "https://server1.com/mcp",
            "https://server2.com/mcp",
            "https://server1.com/mcp?api_key=different"
        ]

        mock_schemas = {"tool1": {"description": "Key test tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas), \
             patch('crewai.agent.time.time', return_value=1000):

            # Call with different URLs
            for url in different_urls:
                caching_agent._get_mcp_tool_schemas({"url": url})

            # Should create separate cache entries for each URL
            assert len(_mcp_schema_cache) == len(different_urls)

            for url in different_urls:
                assert url in _mcp_schema_cache

    def test_cache_handles_identical_concurrent_requests(self, caching_agent):
        """Test cache behavior with identical concurrent requests."""
        server_params = {"url": "https://concurrent-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Concurrent tool"}}

        call_count = 0
        async def counted_async_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Add small delay to simulate network call
            await asyncio.sleep(0.1)
            return mock_schemas

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', side_effect=counted_async_call), \
             patch('crewai.agent.time.time', return_value=1000):

            # First call populates cache
            schemas1 = caching_agent._get_mcp_tool_schemas(server_params)

            # Subsequent calls should use cache
            with patch('crewai.agent.time.time', return_value=1100):
                schemas2 = caching_agent._get_mcp_tool_schemas(server_params)
                schemas3 = caching_agent._get_mcp_tool_schemas(server_params)

            assert schemas1 == schemas2 == schemas3 == mock_schemas
            assert call_count == 1  # Only first call should hit the server

    def test_cache_isolation_between_different_servers(self, caching_agent):
        """Test that cache entries are isolated between different servers."""
        server1_params = {"url": "https://server1.com/mcp"}
        server2_params = {"url": "https://server2.com/mcp"}

        server1_schemas = {"tool1": {"description": "Server 1 tool"}}
        server2_schemas = {"tool2": {"description": "Server 2 tool"}}

        def mock_async_by_url(server_params):
            url = server_params["url"]
            if "server1" in url:
                return server1_schemas
            elif "server2" in url:
                return server2_schemas
            return {}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', side_effect=mock_async_by_url), \
             patch('crewai.agent.time.time', return_value=1000):

            # Call both servers
            schemas1 = caching_agent._get_mcp_tool_schemas(server1_params)
            schemas2 = caching_agent._get_mcp_tool_schemas(server2_params)

            assert schemas1 == server1_schemas
            assert schemas2 == server2_schemas
            assert len(_mcp_schema_cache) == 2

    def test_cache_handles_failed_operations_correctly(self, caching_agent):
        """Test that cache doesn't store failed operations."""
        server_params = {"url": "https://failing-cache-test.com/mcp"}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', side_effect=Exception("Server failed")), \
             patch('crewai.agent.time.time', return_value=1000):

            # Failed operation should not populate cache
            schemas = caching_agent._get_mcp_tool_schemas(server_params)

            assert schemas == {}  # Empty dict returned on failure
            assert len(_mcp_schema_cache) == 0  # Cache should remain empty

    def test_cache_debug_logging(self, caching_agent):
        """Test cache debug logging functionality."""
        server_params = {"url": "https://debug-log-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Debug log tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas), \
             patch.object(caching_agent, '_logger') as mock_logger:

            # First call - populates cache
            with patch('crewai.agent.time.time', return_value=1000):
                caching_agent._get_mcp_tool_schemas(server_params)

            # Second call - should log cache hit
            with patch('crewai.agent.time.time', return_value=1100):  # Within TTL
                caching_agent._get_mcp_tool_schemas(server_params)

            # Should log debug message about cache usage
            debug_calls = [call for call in mock_logger.log.call_args_list if call[0][0] == "debug"]
            assert len(debug_calls) > 0
            assert "cached mcp tool schemas" in debug_calls[0][0][1].lower()

    def test_cache_thread_safety_simulation(self, caching_agent):
        """Simulate thread safety scenarios for cache access."""
        server_params = {"url": "https://thread-safety-test.com/mcp"}
        mock_schemas = {"tool1": {"description": "Thread safety tool"}}

        # Simulate multiple "threads" accessing cache simultaneously
        # (Note: This is a simplified simulation in a single-threaded test)

        results = []

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async, \
             patch('crewai.agent.time.time', return_value=1000):

            # First call populates cache
            result1 = caching_agent._get_mcp_tool_schemas(server_params)
            results.append(result1)

            # Multiple rapid subsequent calls (simulating concurrent access)
            with patch('crewai.agent.time.time', return_value=1001):
                for _ in range(5):
                    result = caching_agent._get_mcp_tool_schemas(server_params)
                    results.append(result)

        # All results should be identical (from cache)
        assert all(result == mock_schemas for result in results)
        assert len(results) == 6
        # Async method should only be called once
        assert mock_async.call_count == 1

    def test_cache_size_management_with_many_servers(self, caching_agent):
        """Test cache behavior with many different servers."""
        mock_schemas = {"tool1": {"description": "Size management tool"}}

        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas), \
             patch('crewai.agent.time.time', return_value=1000):

            # Add many server entries to cache
            for i in range(50):
                server_url = f"https://server{i:03d}.com/mcp"
                caching_agent._get_mcp_tool_schemas({"url": server_url})

            # Cache should contain all entries
            assert len(_mcp_schema_cache) == 50

            # Verify each entry has correct structure
            for server_url, (cached_schemas, cache_time) in _mcp_schema_cache.items():
                assert cached_schemas == mock_schemas
                assert cache_time == 1000

    def test_cache_performance_comparison_with_without_cache(self, caching_agent):
        """Compare performance with and without caching."""
        server_params = {"url": "https://performance-comparison.com/mcp"}
        mock_schemas = {"tool1": {"description": "Performance comparison tool"}}

        # Test without cache (cold call)
        with patch.object(caching_agent, '_get_mcp_tool_schemas_async', return_value=mock_schemas) as mock_async:

            # Cold call
            start_time = time.time()
            with patch('crewai.agent.time.time', return_value=1000):
                schemas1 = caching_agent._get_mcp_tool_schemas(server_params)
            cold_call_time = time.time() - start_time

            # Warm call (from cache)
            start_time = time.time()
            with patch('crewai.agent.time.time', return_value=1100):  # Within TTL
                schemas2 = caching_agent._get_mcp_tool_schemas(server_params)
            warm_call_time = time.time() - start_time

            assert schemas1 == schemas2 == mock_schemas
            assert mock_async.call_count == 1
            # Warm call should be significantly faster
            assert warm_call_time < cold_call_time / 2
