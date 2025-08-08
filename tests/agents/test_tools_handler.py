"""Tests for ToolsHandler type safety and functionality."""

from unittest.mock import Mock

from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.tool_calling import ToolCalling, InstructorToolCalling
from crewai.agents.cache.cache_handler import CacheHandler


class TestToolsHandler:
    """Test suite for ToolsHandler."""

    def test_initialization(self):
        """Test that ToolsHandler initializes correctly."""
        handler = ToolsHandler()

        assert handler.last_used_tool is None
        assert handler.cache is None

    def test_initialization_with_cache(self):
        """Test initialization with cache handler."""
        cache = Mock(spec=CacheHandler)
        handler = ToolsHandler(cache=cache)

        assert handler.last_used_tool is None
        assert handler.cache == cache

    def test_on_tool_use_with_tool_calling(self):
        """Test on_tool_use with ToolCalling object."""
        handler = ToolsHandler()

        tool_call = ToolCalling(
            tool_name="test_tool", arguments={"arg1": "value1", "arg2": 42}
        )

        handler.on_tool_use(tool_call, "test output")

        assert handler.last_used_tool == tool_call
        assert handler.last_used_tool.tool_name == "test_tool"
        assert handler.last_used_tool.arguments == {"arg1": "value1", "arg2": 42}

    def test_on_tool_use_with_instructor_tool_calling(self):
        """Test on_tool_use with InstructorToolCalling object."""
        handler = ToolsHandler()

        tool_call = InstructorToolCalling(
            tool_name="instructor_tool", arguments={"key": "value"}
        )

        handler.on_tool_use(tool_call, "instructor output")

        assert handler.last_used_tool == tool_call
        assert handler.last_used_tool.tool_name == "instructor_tool"
        assert handler.last_used_tool.arguments == {"key": "value"}

    def test_on_tool_use_with_cache(self):
        """Test that tool usage is cached when cache is available."""
        cache = Mock(spec=CacheHandler)
        handler = ToolsHandler(cache=cache)

        tool_call = ToolCalling(tool_name="cached_tool", arguments={"param": "test"})

        handler.on_tool_use(tool_call, "cached output", should_cache=True)

        cache.add.assert_called_once_with(
            tool="cached_tool", input={"param": "test"}, output="cached output"
        )

    def test_on_tool_use_without_cache(self):
        """Test that tool usage works without cache."""
        handler = ToolsHandler()  # No cache

        tool_call = ToolCalling(tool_name="no_cache_tool", arguments={"param": "test"})

        # Should not raise any errors
        handler.on_tool_use(tool_call, "output", should_cache=True)

        assert handler.last_used_tool == tool_call

    def test_on_tool_use_with_cache_disabled(self):
        """Test that caching can be disabled."""
        cache = Mock(spec=CacheHandler)
        handler = ToolsHandler(cache=cache)

        tool_call = ToolCalling(tool_name="no_cache_tool", arguments={"param": "test"})

        handler.on_tool_use(tool_call, "output", should_cache=False)

        # Cache should not be called
        cache.add.assert_not_called()

    def test_cache_tools_exclusion(self):
        """Test that CacheTools itself is not cached."""
        cache = Mock(spec=CacheHandler)
        handler = ToolsHandler(cache=cache)

        tool_call = ToolCalling(
            tool_name="Hit Cache",  # CacheTools name
            arguments={"query": "test"},
        )

        handler.on_tool_use(tool_call, "cache tool output", should_cache=True)

        # Cache should not be called for CacheTools
        cache.add.assert_not_called()
        # But last_used_tool should still be updated
        assert handler.last_used_tool == tool_call

    def test_reset_last_used_tool(self):
        """Test resetting last_used_tool to None."""
        handler = ToolsHandler()

        # First set a tool
        tool_call = ToolCalling(tool_name="test_tool", arguments={"arg": "value"})
        handler.on_tool_use(tool_call, "output")
        assert handler.last_used_tool == tool_call

        # Now reset it
        handler.last_used_tool = None
        assert handler.last_used_tool is None

    def test_multiple_tool_uses(self):
        """Test that last_used_tool is updated correctly with multiple uses."""
        handler = ToolsHandler()

        # First tool
        tool1 = ToolCalling(tool_name="tool1", arguments={"a": 1})
        handler.on_tool_use(tool1, "output1")
        assert handler.last_used_tool == tool1

        # Second tool
        tool2 = ToolCalling(tool_name="tool2", arguments={"b": 2})
        handler.on_tool_use(tool2, "output2")
        assert handler.last_used_tool == tool2

        # Third tool (InstructorToolCalling)
        tool3 = InstructorToolCalling(tool_name="tool3", arguments={"c": 3})
        handler.on_tool_use(tool3, "output3")
        assert handler.last_used_tool == tool3

    def test_type_safety(self):
        """Test that type annotations are correct."""
        handler = ToolsHandler()

        # Test that None is a valid value
        handler.last_used_tool = None
        assert handler.last_used_tool is None

        # Test with ToolCalling
        tool_calling = ToolCalling(tool_name="test", arguments={})
        handler.last_used_tool = tool_calling
        assert isinstance(handler.last_used_tool, ToolCalling)

        # Test with InstructorToolCalling
        instructor_calling = InstructorToolCalling(tool_name="test", arguments={})
        handler.last_used_tool = instructor_calling
        assert isinstance(handler.last_used_tool, InstructorToolCalling)

    def test_arguments_can_be_none(self):
        """Test that tool arguments can be None."""
        handler = ToolsHandler()

        tool_call = ToolCalling(tool_name="no_args_tool", arguments=None)

        handler.on_tool_use(tool_call, "output")

        assert handler.last_used_tool == tool_call
        assert handler.last_used_tool.arguments is None
