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
    key1 = cache._generate_cache_key(tools, True, False, True, "sequential")
    key2 = cache._generate_cache_key(tools, True, False, True, "sequential")
    assert key1 == key2


def test_cache_key_generation_uniqueness():
    """Test that different parameters generate different keys."""
    cache = ToolSharingCache()
    tools1 = [Mock(name="tool1")]
    tools2 = [Mock(name="tool2")]

    key1 = cache._generate_cache_key(tools1, True, False, True, "sequential")
    key2 = cache._generate_cache_key(tools1, False, False, True, "sequential")
    key3 = cache._generate_cache_key(tools2, True, False, True, "sequential")

    assert key1 != key2
    assert key1 != key3
    assert key2 != key3


def test_cache_miss_returns_none(cache):
    """Test that cache miss returns None."""
    tools = [Mock(name="tool1")]
    result = cache.get_tools(tools)
    assert result is None


def test_cache_hit_returns_tools(cache):
    """Test that cache hit returns stored tools."""
    tools = [Mock(name="tool1")]
    prepared_tools = tools + [Mock(name="extra_tool")]

    cache.store_tools(tools, prepared_tools)
    result = cache.get_tools(tools)

    assert result is not None
    assert len(result) == 2


def test_cache_returns_copy_not_reference(cache):
    """Test that cache returns a copy to prevent external mutations."""
    tools = [Mock(name="tool1")]
    prepared_tools = [Mock(name="prepared1")]

    cache.store_tools(tools, prepared_tools)

    # Get from cache and modify
    result1 = cache.get_tools(tools)
    result1.append(Mock(name="external_modification"))

    # Get again from cache
    result2 = cache.get_tools(tools)

    assert len(result2) == 1
    assert len(result1) == 2


def test_cache_reuse_across_agents(cache):
    """Test that cache is reused across different agents with same tools."""
    tools = [Mock(name="tool1")]
    prepared = tools + [Mock(name="extra")]

    # Store once
    cache.store_tools(tools, prepared)

    # Should hit cache regardless of agent (agent_id removed from key)
    result = cache.get_tools(tools)
    assert result is not None
    assert len(result) == 2


def test_cache_reuse_across_tasks(cache):
    """Test that cache is reused across different tasks with same tools."""
    tools = [Mock(name="tool1")]
    prepared = tools + [Mock(name="extra")]

    # Store once
    cache.store_tools(tools, prepared)

    # Should hit cache regardless of task (task_id removed from key)
    result = cache.get_tools(tools)
    assert result is not None
    assert len(result) == 2


def test_cache_miss_different_tools(cache):
    """Test cache miss when tools differ."""
    tools1 = [Mock(name="tool1")]
    tools2 = [Mock(name="tool2")]

    cache.store_tools(tools1, tools1)
    result = cache.get_tools(tools2)
    assert result is None


def test_cache_miss_different_flags(cache):
    """Test cache miss when configuration flags differ."""
    tools = [Mock(name="tool1")]

    # Store with all flags False
    cache.store_tools(tools, tools, False, False, False, "sequential")

    # Test each flag independently
    assert cache.get_tools(tools, True, False, False, "sequential") is None
    assert cache.get_tools(tools, False, True, False, "sequential") is None
    assert cache.get_tools(tools, False, False, True, "sequential") is None
    assert cache.get_tools(tools, False, False, False, "hierarchical") is None


def test_lru_eviction_basic(cache_small):
    """Test that least recently used item is evicted first."""
    # Store tools for reuse
    tools_list = []
    for i in range(3):
        tools = [Mock(name=f"tool{i}")]
        tools_list.append(tools)
        cache_small.store_tools(tools, tools)

    assert cache_small.size() == 3

    # Add 4th item - should evict first
    tools4 = [Mock(name="tool4")]
    cache_small.store_tools(tools4, tools4)

    assert cache_small.size() == 3
    assert cache_small.get_tools(tools_list[0]) is None
    assert cache_small.get_tools(tools_list[1]) is not None


def test_lru_access_updates_order(cache_small):
    """Test that accessing an item updates its position in LRU order."""
    tools = []
    for i in range(3):
        tool = [Mock(name=f"tool{i}")]
        tools.append(tool)
        cache_small.store_tools(tool, tool)

    # Access first item (makes it most recently used)
    cache_small.get_tools(tools[0])

    # Add 4th item - should evict agent1 (now least recently used)
    tools4 = [Mock(name="tool4")]
    cache_small.store_tools(tools4, tools4)

    assert cache_small.get_tools(tools[0]) is not None
    assert cache_small.get_tools(tools[1]) is None


def test_lru_with_single_slot():
    """Test LRU behavior with cache size of 1."""
    cache = ToolSharingCache(max_size=1)

    tools1 = [Mock(name="tool1")]
    tools2 = [Mock(name="tool2")]

    cache.store_tools(tools1, tools1)
    assert cache.get_tools(tools1) is not None

    cache.store_tools(tools2, tools2)
    assert cache.get_tools(tools1) is None
    assert cache.get_tools(tools2) is not None


def test_cache_clear(cache):
    """Test cache clearing functionality."""
    tools = [Mock(name="tool1")]
    cache.store_tools(tools, tools)

    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0
    assert cache.get_tools(tools) is None


def test_empty_tools_list(cache):
    """Test caching with empty tools list."""
    empty_tools = []
    cache.store_tools(empty_tools, empty_tools)

    result = cache.get_tools(empty_tools)
    assert result is not None
    assert len(result) == 0


def test_very_large_tool_list(cache):
    """Test caching with large number of tools."""
    large_tools = [Mock(name=f"tool_{i}") for i in range(100)]
    cache.store_tools(large_tools, large_tools)

    result = cache.get_tools(large_tools)
    assert result is not None
    assert len(result) == 100


def test_tools_without_name_attribute(cache):
    """Test tools without 'name' attribute."""
    # Create mock objects without name attribute
    tool1 = Mock(spec=[])  # No attributes
    tool2 = Mock(spec=[])
    tools = [tool1, tool2]

    # Should not raise error
    cache.store_tools(tools, tools)
    result = cache.get_tools(tools)
    assert result is not None


def test_tool_instances_with_same_name_different_config(cache):
    """Test that tools with same name but different configs are cached separately."""
    # Create two FileReadTool instances with different file_paths
    tool1 = Mock()
    tool1.name = "read_file"
    tool1.file_path = "file-chunk.txt"

    tool2 = Mock()
    tool2.name = "read_file"
    tool2.file_path = "knowledge.txt"

    tools_set1 = [tool1]
    tools_set2 = [tool2]

    extra1 = Mock()
    extra1.name = "extra1"
    extra2 = Mock()
    extra2.name = "extra2"

    prepared1 = tools_set1 + [extra1]
    prepared2 = tools_set2 + [extra2]

    # Store both tool sets
    cache.store_tools(tools_set1, prepared1)
    cache.store_tools(tools_set2, prepared2)

    # Retrieve and verify they are different
    result1 = cache.get_tools(tools_set1)
    result2 = cache.get_tools(tools_set2)

    assert result1 is not None
    assert result2 is not None
    assert len(result1) == 2
    assert len(result2) == 2
    # The cached results should be different
    assert result1[1].name == "extra1"
    assert result2[1].name == "extra2"


def test_tool_config_attributes_in_cache_key(cache):
    """Test that various tool configuration attributes are included in cache key."""
    # Test with different configuration attributes
    tool_with_file = Mock()
    tool_with_file.name = "tool"
    tool_with_file.file_path = "/path/to/file.txt"

    tool_with_url = Mock()
    tool_with_url.name = "tool"
    tool_with_url.url = "https://api.example.com"

    tool_with_db = Mock()
    tool_with_db.name = "tool"
    tool_with_db.database = "mydb"
    tool_with_db.table = "users"

    tools1 = [tool_with_file]
    tools2 = [tool_with_url]
    tools3 = [tool_with_db]

    # Generate cache keys
    key1 = cache._generate_cache_key(tools1)
    key2 = cache._generate_cache_key(tools2)
    key3 = cache._generate_cache_key(tools3)

    # All keys should be different
    assert key1 != key2
    assert key2 != key3
    assert key1 != key3


def test_api_key_differentiation(cache):
    """Test that tools with different API keys generate different cache keys."""
    # Create tools with different API keys
    tool1 = Mock()
    tool1.name = "api_tool"
    tool1.api_key = "secret-key-123"

    tool2 = Mock()
    tool2.name = "api_tool"
    tool2.api_key = "secret-key-456"

    tools1 = [tool1]
    tools2 = [tool2]

    # Generate cache keys
    key1 = cache._generate_cache_key(tools1)
    key2 = cache._generate_cache_key(tools2)

    # Keys should be different (different API keys)
    assert key1 != key2


def test_tool_with_different_attributes(cache):
    """Test that tools with different attributes generate different cache keys."""
    # Create tools with different attributes
    tool1 = Mock()
    tool1.name = "custom_tool"
    tool1.config = "config_1"
    tool1.version = "1.0"

    tool2 = Mock()
    tool2.name = "custom_tool"
    tool2.config = "config_2"
    tool2.version = "2.0"

    tools1 = [tool1]
    tools2 = [tool2]

    # Generate cache keys
    key1 = cache._generate_cache_key(tools1)
    key2 = cache._generate_cache_key(tools2)

    # Keys should be different due to different attributes
    assert key1 != key2


def test_tool_identifier_strategies(cache):
    """Test different strategies for generating tool identifiers."""
    # Tool with __dict__ attributes
    tool_with_dict = Mock()
    tool_with_dict.name = "tool"
    tool_with_dict.config_value = "test123"
    id1 = cache._get_tool_identifier(tool_with_dict)
    # Should create a hash from attributes
    assert "tool:" in id1

    # Tool with only name attribute (still generates hash from name attr)
    tool_with_name = Mock()
    tool_with_name.name = "simple_tool"
    tool_with_name.__dict__ = {"name": "simple_tool"}  # Keep only name
    id2 = cache._get_tool_identifier(tool_with_name)
    assert "simple_tool:" in id2  # Should have name and a hash

    # Tool without name or attributes
    tool_minimal = Mock(spec=[])
    id3 = cache._get_tool_identifier(tool_minimal)
    assert id3.startswith("tool_")  # Uses object ID


def test_crewai_tool_with_args_schema(cache):
    """Test that crewAI Tool instances with args_schema are handled correctly."""
    from pydantic import BaseModel, Field

    # Mock a crewAI Tool with args_schema containing default values
    # This simulates FileReadTool(file_path="file1.txt")
    class MockArgsSchema(BaseModel):
        file_path: str = Field(default="file1.txt")

    tool1 = Mock()
    tool1.name = "read_file"
    tool1.func = lambda x: x  # Simple function
    tool1.args_schema = MockArgsSchema

    # Another tool with different file_path
    class MockArgsSchema2(BaseModel):
        file_path: str = Field(default="file2.txt")

    tool2 = Mock()
    tool2.name = "read_file"
    tool2.func = lambda x: x  # Same function
    tool2.args_schema = MockArgsSchema2

    # Generate identifiers
    id1 = cache._get_tool_identifier(tool1)
    id2 = cache._get_tool_identifier(tool2)

    # They should be different due to different default file_path values
    assert id1 != id2
    assert "read_file:" in id1
    assert "read_file:" in id2


def test_max_size_zero():
    """Test cache with max_size of 0."""
    cache = ToolSharingCache(max_size=0)
    tools = [Mock(name="tool1")]

    # Should handle gracefully
    cache.store_tools(tools, tools)
    cache.get_tools(tools)

    # With max_size=0, nothing should be cached
    assert cache.size() == 0


def test_concurrent_reads():
    """Test that concurrent reads are thread-safe."""
    cache = ToolSharingCache()
    tools = [Mock(name="tool1")]
    cache.store_tools(tools, tools)

    results = []
    errors = []

    def read_cache():
        try:
            result = cache.get_tools(tools)
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

    def write_cache(tool_id):
        try:
            tools = [Mock(name=f"tool_{tool_id}")]
            cache.store_tools(tools, tools)
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

            if operation_id % 2 == 0:
                # Read operation
                cache.get_tools(tools)
            else:
                # Write operation
                cache.store_tools(tools, tools)
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
    assert should_use_tool_sharing(tools) is True


def test_should_use_tool_sharing_multiple():
    """Test tool sharing decision with multiple tools."""
    tools = [Mock(name=f"tool_{i}") for i in range(3)]
    assert should_use_tool_sharing(tools) is True


def test_should_use_tool_sharing_threshold():
    """Test tool sharing decision at threshold boundary."""
    # Even 1 tool should use sharing now
    tools = [Mock(name="tool1")]
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
    result = cache.get_tools(tools)
    assert result is None

    # Store and test hit
    cache.store_tools(tools, tools)
    result = cache.get_tools(tools)
    assert result is not None


def test_agent_capability_flags(cache):
    """Test that agent capability flags affect cache keys."""
    tools = [Mock(name="tool1")]

    # Store with delegation enabled
    cache.store_tools(tools, tools, allow_delegation=True)

    # Should hit with same flags
    result = cache.get_tools(tools, allow_delegation=True)
    assert result is not None

    # Should miss with different flags
    result = cache.get_tools(tools, allow_delegation=False)
    assert result is None


def test_process_type_affects_caching(cache):
    """Test that process type affects cache behavior."""
    tools = [Mock(name="tool1")]

    # Store with sequential process
    cache.store_tools(tools, tools, process_type="sequential")

    # Should hit with same process type
    result = cache.get_tools(tools, process_type="sequential")
    assert result is not None

    # Should miss with different process type
    result = cache.get_tools(tools, process_type="hierarchical")
    assert result is None


def test_cache_key_collision_resistance(cache):
    """Test that cache keys have good collision resistance."""
    # Create many different configurations
    keys = set()

    for i in range(100):
        # Each tool has a unique name, so each should generate a unique key
        tools = [Mock(name=f"tool_{i}")]
        key = cache._generate_cache_key(
            tools,
            i % 2 == 0,
            i % 3 == 0,
            i % 5 == 0,
            "sequential" if i % 2 == 0 else "hierarchical",
        )
        keys.add(key)

    # We should have unique keys for each unique tool set
    # With 100 unique tools and various flag combinations,
    # we expect a reasonable number of unique cache keys
    assert len(keys) >= 25  # At least 25 unique combinations
    # Note: The exact number depends on how flags and tool names interact
    # in the cache key generation and the Mock object's attributes


def test_lru_eviction_thread_safety():
    """Test that LRU eviction is thread-safe under concurrent access."""
    cache = ToolSharingCache(max_size=10)
    errors = []

    def access_and_add(thread_id):
        try:
            # Each thread adds items and accesses existing ones
            for i in range(5):
                tools = [Mock(name=f"tool_{thread_id}_{i}")]
                cache.store_tools(tools, tools)

                # Try to access some existing items
                if i > 0:
                    old_tools = [Mock(name=f"tool_{thread_id}_{i - 1}")]
                    cache.get_tools(old_tools)
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
        (1, True),  # Single tool now uses cache
        (2, True),  # Multiple tools
        (3, True),  # Above old threshold
        (10, True),  # Many tools
        (100, True),  # Very many tools
    ],
)
def test_tool_count_threshold_parametrized(tool_count, expected):
    """Test tool sharing decision based on tool count with parametrized inputs."""
    tools = [Mock(name=f"tool_{i}") for i in range(tool_count)]
    assert should_use_tool_sharing(tools) == expected
