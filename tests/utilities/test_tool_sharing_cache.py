"""Tests for tool sharing cache optimization."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

import pytest

from crewai.utilities.tool_sharing_cache import (
    ToolSharingCache,
    get_tool_sharing_cache,
    should_use_tool_sharing,
)


@pytest.fixture
def cache():
    """Create a fresh cache instance for testing."""
    return ToolSharingCache()


@pytest.fixture
def cache_small():
    """Create a cache with small max size for eviction testing."""
    return ToolSharingCache(max_size=3)


@pytest.fixture
def mock_tools():
    """Create mock tool objects for testing."""
    return [Mock(name=f"tool_{i}") for i in range(3)]


def test_cache_initialization():
    """Test cache initializes with correct parameters."""
    cache = ToolSharingCache(max_size=64)
    assert cache.size() == 0
    assert cache._max_size == 64


def test_cache_key_generation_consistency():
    """Test that cache keys are generated consistently."""
    cache = ToolSharingCache()
    tools = [Mock(name="tool1"), Mock(name="tool2")]

    # Same parameters should generate same key
    key1 = cache._generate_cache_key(
        "agent1", "task1", tools, True, False, True, "sequential"
    )
    key2 = cache._generate_cache_key(
        "agent1", "task1", tools, True, False, True, "sequential"
    )
    assert key1 == key2


def test_cache_key_generation_uniqueness():
    """Test that different parameters generate different keys."""
    cache = ToolSharingCache()
    tools = [Mock(name="tool1")]

    key1 = cache._generate_cache_key(
        "agent1", "task1", tools, True, False, True, "sequential"
    )
    key2 = cache._generate_cache_key(
        "agent1", "task1", tools, False, False, True, "sequential"
    )
    key3 = cache._generate_cache_key(
        "agent2", "task1", tools, True, False, True, "sequential"
    )

    assert key1 != key2
    assert key1 != key3
    assert key2 != key3


def test_cache_miss_returns_none(cache):
    """Test that cache miss returns None."""
    tools = [Mock(name="tool1")]
    result = cache.get_tools("agent1", "task1", tools)
    assert result is None


def test_cache_hit_returns_tools(cache):
    """Test that cache hit returns stored tools."""
    tools = [Mock(name="tool1")]
    prepared_tools = tools + [Mock(name="extra_tool")]

    cache.store_tools("agent1", "task1", tools, prepared_tools)
    result = cache.get_tools("agent1", "task1", tools)

    assert result is not None
    assert len(result) == 2


def test_cache_returns_copy_not_reference(cache):
    """Test that cache returns a copy to prevent external mutations."""
    tools = [Mock(name="tool1")]
    prepared_tools = [Mock(name="prepared1")]

    cache.store_tools("agent1", "task1", tools, prepared_tools)

    # Get from cache and modify
    result1 = cache.get_tools("agent1", "task1", tools)
    result1.append(Mock(name="external_modification"))

    # Get again from cache
    result2 = cache.get_tools("agent1", "task1", tools)

    assert len(result2) == 1
    assert len(result1) == 2


def test_cache_miss_different_agent(cache):
    """Test cache miss when agent ID differs."""
    tools = [Mock(name="tool1")]
    cache.store_tools("agent1", "task1", tools, tools)

    result = cache.get_tools("agent2", "task1", tools)
    assert result is None


def test_cache_miss_different_task(cache):
    """Test cache miss when task ID differs."""
    tools = [Mock(name="tool1")]
    cache.store_tools("agent1", "task1", tools, tools)

    result = cache.get_tools("agent1", "task2", tools)
    assert result is None


def test_cache_miss_different_tools(cache):
    """Test cache miss when tools differ."""
    tools1 = [Mock(name="tool1")]
    tools2 = [Mock(name="tool2")]

    cache.store_tools("agent1", "task1", tools1, tools1)
    result = cache.get_tools("agent1", "task1", tools2)
    assert result is None


def test_cache_miss_different_flags(cache):
    """Test cache miss when configuration flags differ."""
    tools = [Mock(name="tool1")]

    # Store with all flags False
    cache.store_tools(
        "agent1", "task1", tools, tools, False, False, False, "sequential"
    )

    # Test each flag independently
    assert (
        cache.get_tools("agent1", "task1", tools, True, False, False, "sequential")
        is None
    )
    assert (
        cache.get_tools("agent1", "task1", tools, False, True, False, "sequential")
        is None
    )
    assert (
        cache.get_tools("agent1", "task1", tools, False, False, True, "sequential")
        is None
    )
    assert (
        cache.get_tools("agent1", "task1", tools, False, False, False, "hierarchical")
        is None
    )


def test_lru_eviction_basic(cache_small):
    """Test that least recently used item is evicted first."""
    # Store tools for reuse
    tools_list = []
    for i in range(3):
        tools = [Mock(name=f"tool{i}")]
        tools_list.append(tools)
        cache_small.store_tools(f"agent{i}", f"task{i}", tools, tools)

    assert cache_small.size() == 3

    # Add 4th item - should evict first
    tools4 = [Mock(name="tool4")]
    cache_small.store_tools("agent4", "task4", tools4, tools4)

    assert cache_small.size() == 3
    assert cache_small.get_tools("agent0", "task0", tools_list[0]) is None
    assert cache_small.get_tools("agent1", "task1", tools_list[1]) is not None


def test_lru_access_updates_order(cache_small):
    """Test that accessing an item updates its position in LRU order."""
    tools = []
    for i in range(3):
        tool = [Mock(name=f"tool{i}")]
        tools.append(tool)
        cache_small.store_tools(f"agent{i}", f"task{i}", tool, tool)

    # Access first item (makes it most recently used)
    cache_small.get_tools("agent0", "task0", tools[0])

    # Add 4th item - should evict agent1 (now least recently used)
    tools4 = [Mock(name="tool4")]
    cache_small.store_tools("agent4", "task4", tools4, tools4)

    assert cache_small.get_tools("agent0", "task0", tools[0]) is not None
    assert cache_small.get_tools("agent1", "task1", tools[1]) is None


def test_lru_with_single_slot():
    """Test LRU behavior with cache size of 1."""
    cache = ToolSharingCache(max_size=1)

    tools1 = [Mock(name="tool1")]
    tools2 = [Mock(name="tool2")]

    cache.store_tools("agent1", "task1", tools1, tools1)
    assert cache.get_tools("agent1", "task1", tools1) is not None

    cache.store_tools("agent2", "task2", tools2, tools2)
    assert cache.get_tools("agent1", "task1", tools1) is None
    assert cache.get_tools("agent2", "task2", tools2) is not None


def test_cache_clear(cache):
    """Test cache clearing functionality."""
    tools = [Mock(name="tool1")]
    cache.store_tools("agent1", "task1", tools, tools)

    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0
    assert cache.get_tools("agent1", "task1", tools) is None


def test_empty_tools_list(cache):
    """Test caching with empty tools list."""
    empty_tools = []
    cache.store_tools("agent1", "task1", empty_tools, empty_tools)

    result = cache.get_tools("agent1", "task1", empty_tools)
    assert result is not None
    assert len(result) == 0


def test_very_large_tool_list(cache):
    """Test caching with large number of tools."""
    large_tools = [Mock(name=f"tool_{i}") for i in range(100)]
    cache.store_tools("agent1", "task1", large_tools, large_tools)

    result = cache.get_tools("agent1", "task1", large_tools)
    assert result is not None
    assert len(result) == 100


def test_tools_without_name_attribute(cache):
    """Test tools without 'name' attribute."""
    # Create mock objects without name attribute
    tool1 = Mock(spec=[])  # No attributes
    tool2 = Mock(spec=[])
    tools = [tool1, tool2]

    # Should not raise error
    cache.store_tools("agent1", "task1", tools, tools)
    result = cache.get_tools("agent1", "task1", tools)
    assert result is not None


def test_max_size_zero():
    """Test cache with max_size of 0."""
    cache = ToolSharingCache(max_size=0)
    tools = [Mock(name="tool1")]

    # Should handle gracefully
    cache.store_tools("agent1", "task1", tools, tools)
    cache.get_tools("agent1", "task1", tools)

    # With max_size=0, nothing should be cached
    assert cache.size() == 0


def test_concurrent_reads():
    """Test that concurrent reads are thread-safe."""
    cache = ToolSharingCache()
    tools = [Mock(name="tool1")]
    cache.store_tools("agent1", "task1", tools, tools)

    results = []
    errors = []

    def read_cache():
        try:
            result = cache.get_tools("agent1", "task1", tools)
            results.append(result is not None)
        except Exception as e:
            errors.append(e)

    # Spawn multiple threads for concurrent reads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_cache) for _ in range(100)]
        for future in as_completed(futures):
            future.result()

    assert len(errors) == 0
    assert all(results)


def test_concurrent_writes():
    """Test that concurrent writes are thread-safe."""
    cache = ToolSharingCache(max_size=100)
    errors = []

    def write_cache(agent_id):
        try:
            tools = [Mock(name=f"tool_{agent_id}")]
            cache.store_tools(f"agent{agent_id}", f"task{agent_id}", tools, tools)
        except Exception as e:
            errors.append(e)

    # Spawn multiple threads for concurrent writes
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_cache, i) for i in range(50)]
        for future in as_completed(futures):
            future.result()

    assert len(errors) == 0
    assert cache.size() <= 100


def test_concurrent_read_write():
    """Test concurrent reads and writes are thread-safe."""
    cache = ToolSharingCache(max_size=50)
    errors = []

    def read_write_cache(operation_id):
        try:
            tools = [Mock(name=f"tool_{operation_id % 10}")]
            agent = f"agent{operation_id % 10}"
            task = f"task{operation_id % 10}"

            if operation_id % 2 == 0:
                # Read operation
                cache.get_tools(agent, task, tools)
            else:
                # Write operation
                cache.store_tools(agent, task, tools, tools)
        except Exception as e:
            errors.append(e)

    # Spawn multiple threads for mixed operations
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(read_write_cache, i) for i in range(200)]
        for future in as_completed(futures):
            future.result()

    assert len(errors) == 0


def test_should_use_tool_sharing_empty():
    """Test tool sharing decision with empty list."""
    assert should_use_tool_sharing([]) is False


def test_should_use_tool_sharing_single():
    """Test tool sharing decision with single tool."""
    tools = [Mock(name="single_tool")]
    assert should_use_tool_sharing(tools) is False


def test_should_use_tool_sharing_multiple():
    """Test tool sharing decision with multiple tools."""
    tools = [Mock(name=f"tool_{i}") for i in range(3)]
    assert should_use_tool_sharing(tools) is True


def test_should_use_tool_sharing_threshold():
    """Test tool sharing decision at threshold boundary."""
    # Exactly 2 tools should use sharing
    tools = [Mock(name="tool1"), Mock(name="tool2")]
    assert should_use_tool_sharing(tools) is True


def test_global_cache_singleton():
    """Test that global cache returns the same instance."""
    cache1 = get_tool_sharing_cache()
    cache2 = get_tool_sharing_cache()
    assert cache1 is cache2


def test_global_cache_functionality():
    """Test that global cache works for basic operations."""
    cache = get_tool_sharing_cache()

    # Clear any existing state
    cache.clear()

    tools = [Mock(name="test_tool")]

    # Test miss
    result = cache.get_tools("test_agent", "test_task", tools)
    assert result is None

    # Store and test hit
    cache.store_tools("test_agent", "test_task", tools, tools)
    result = cache.get_tools("test_agent", "test_task", tools)
    assert result is not None


def test_agent_capability_flags(cache):
    """Test that agent capability flags affect cache keys."""
    tools = [Mock(name="tool1")]

    # Store with delegation enabled
    cache.store_tools("agent1", "task1", tools, tools, allow_delegation=True)

    # Should hit with same flags
    result = cache.get_tools("agent1", "task1", tools, allow_delegation=True)
    assert result is not None

    # Should miss with different flags
    result = cache.get_tools("agent1", "task1", tools, allow_delegation=False)
    assert result is None


def test_process_type_affects_caching(cache):
    """Test that process type affects cache behavior."""
    tools = [Mock(name="tool1")]

    # Store with sequential process
    cache.store_tools("agent1", "task1", tools, tools, process_type="sequential")

    # Should hit with same process type
    result = cache.get_tools("agent1", "task1", tools, process_type="sequential")
    assert result is not None

    # Should miss with different process type
    result = cache.get_tools("agent1", "task1", tools, process_type="hierarchical")
    assert result is None


def test_cache_key_collision_resistance(cache):
    """Test that cache keys have good collision resistance."""
    # Create many different configurations
    keys = set()
    for i in range(100):
        tools = [Mock(name=f"tool_{i}")]
        key = cache._generate_cache_key(
            f"agent{i}",
            f"task{i}",
            tools,
            i % 2 == 0,
            i % 3 == 0,
            i % 5 == 0,
            "sequential" if i % 2 == 0 else "hierarchical",
        )
        keys.add(key)

    # All keys should be unique
    assert len(keys) == 100


def test_lru_eviction_thread_safety():
    """Test that LRU eviction is thread-safe under concurrent access."""
    cache = ToolSharingCache(max_size=10)
    errors = []

    def access_and_add(thread_id):
        try:
            # Each thread adds items and accesses existing ones
            for i in range(5):
                tools = [Mock(name=f"tool_{thread_id}_{i}")]
                cache.store_tools(
                    f"agent{thread_id}_{i}", f"task{thread_id}_{i}", tools, tools
                )

                # Try to access some existing items
                if i > 0:
                    old_tools = [Mock(name=f"tool_{thread_id}_{i - 1}")]
                    cache.get_tools(
                        f"agent{thread_id}_{i - 1}",
                        f"task{thread_id}_{i - 1}",
                        old_tools,
                    )
        except Exception as e:
            errors.append(e)

    # Run concurrent threads that trigger eviction
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(access_and_add, i) for i in range(10)]
        for future in as_completed(futures):
            future.result()

    assert len(errors) == 0
    assert cache.size() <= 10


@pytest.mark.parametrize(
    "tool_count,expected",
    [
        (0, False),  # Empty list
        (1, False),  # Single tool
        (2, True),  # Threshold boundary
        (3, True),  # Above threshold
        (10, True),  # Many tools
        (100, True),  # Very many tools
    ],
)
def test_tool_count_threshold_parametrized(tool_count, expected):
    """Test tool sharing decision based on tool count with parametrized inputs."""
    tools = [Mock(name=f"tool_{i}") for i in range(tool_count)]
    assert should_use_tool_sharing(tools) == expected
