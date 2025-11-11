"""Tests for the tool decorator compatibility alias in crewai_tools."""

import warnings

import pytest
from crewai.tools import BaseTool


def test_tool_import_from_crewai_tools():
    """Test that tool can be imported from crewai_tools."""
    from crewai_tools import tool

    assert callable(tool)


def test_tool_decorator_basic_usage():
    """Test that the tool decorator works with basic usage."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @tool
        def my_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer to: {question}"

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "from crewai.tools import tool" in str(w[0].message)

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "my_tool"
    assert "Answer a question" in my_tool.description
    assert my_tool.func("test") == "Answer to: test"


def test_tool_decorator_with_name():
    """Test that the tool decorator works with a custom name."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @tool("Custom Tool Name")
        def my_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer to: {question}"

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "Custom Tool Name"
    assert "Answer a question" in my_tool.description
    assert my_tool.func("test") == "Answer to: test"


def test_tool_decorator_with_result_as_answer():
    """Test that the tool decorator works with result_as_answer parameter."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @tool("My Tool", result_as_answer=True)
        def my_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer to: {question}"

        assert len(w) == 1

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "My Tool"
    assert my_tool.result_as_answer is True


def test_tool_decorator_with_max_usage_count():
    """Test that the tool decorator works with max_usage_count parameter."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @tool("My Tool", max_usage_count=5)
        def my_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer to: {question}"

        assert len(w) == 1

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "My Tool"
    assert my_tool.max_usage_count == 5
    assert my_tool.current_usage_count == 0


def test_tool_decorator_with_all_parameters():
    """Test that the tool decorator works with all parameters."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @tool("Custom Name", result_as_answer=True, max_usage_count=3)
        def my_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer to: {question}"

        assert len(w) == 1

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "Custom Name"
    assert my_tool.result_as_answer is True
    assert my_tool.max_usage_count == 3


def test_tool_alias_matches_core_behavior():
    """Test that the alias behaves identically to the core tool decorator."""
    from crewai.tools import tool as core_tool
    from crewai_tools import tool as alias_tool

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        @alias_tool
        def my_test_tool(question: str) -> str:
            """Answer a question."""
            return f"Answer: {question}"

    @core_tool
    def my_test_tool_core(question: str) -> str:
        """Answer a question."""
        return f"Answer: {question}"

    assert type(my_test_tool) == type(my_test_tool_core)
    assert my_test_tool.func("test") == my_test_tool_core.func("test")
    assert my_test_tool.result_as_answer == my_test_tool_core.result_as_answer
    assert my_test_tool.max_usage_count == my_test_tool_core.max_usage_count


def test_tool_requires_docstring():
    """Test that the tool decorator requires a docstring."""
    from crewai_tools import tool

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        with pytest.raises(ValueError, match="Function must have a docstring"):

            @tool
            def my_tool(question: str) -> str:
                return f"Answer to: {question}"
