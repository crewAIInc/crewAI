import pytest
from pydantic import BaseModel, Field
from typing import Dict, Any, Union

from crewai.tools.structured_tool import CrewStructuredTool


class StringInputSchema(BaseModel):
    """Schema with a string input field."""
    query: str = Field(description="A string input parameter")


class IntInputSchema(BaseModel):
    """Schema with an integer input field."""
    number: int = Field(description="An integer input parameter")


class ComplexInputSchema(BaseModel):
    """Schema with multiple fields of different types."""
    text: str = Field(description="A string parameter")
    number: int = Field(description="An integer parameter")
    flag: bool = Field(description="A boolean parameter")


def test_parse_args_with_string_input():
    """Test that string inputs are parsed correctly."""
    def test_func(query: str) -> str:
        return f"Processed: {query}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="StringTool",
        description="A tool that processes string input"
    )
    
    # Test with direct string input
    result = tool._parse_args({"query": "test string"})
    assert result["query"] == "test string"
    assert isinstance(result["query"], str)
    
    # Test with JSON string input
    result = tool._parse_args('{"query": "json string"}')
    assert result["query"] == "json string"
    assert isinstance(result["query"], str)


def test_parse_args_with_nested_dict_for_string():
    """Test that nested dictionaries with 'value' field are handled correctly for string fields."""
    def test_func(query: str) -> str:
        return f"Processed: {query}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="StringTool",
        description="A tool that processes string input"
    )
    
    # Test with nested dict input (simulating the issue from different LLM providers)
    nested_input = {"query": {"description": "A string input parameter", "value": "test value"}}
    result = tool._parse_args(nested_input)
    assert result["query"] == "test value"
    assert isinstance(result["query"], str)


def test_parse_args_with_nested_dict_for_int():
    """Test that nested dictionaries with 'value' field are handled correctly for int fields."""
    def test_func(number: int) -> str:
        return f"Processed: {number}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="IntTool",
        description="A tool that processes integer input"
    )
    
    # Test with nested dict input for int field
    nested_input = {"number": {"description": "An integer input parameter", "value": 42}}
    result = tool._parse_args(nested_input)
    assert result["number"] == 42
    assert isinstance(result["number"], int)


def test_parse_args_with_complex_input():
    """Test that complex inputs with multiple fields are handled correctly."""
    def test_func(text: str, number: int, flag: bool) -> str:
        return f"Processed: {text}, {number}, {flag}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="ComplexTool",
        description="A tool that processes complex input"
    )
    
    # Test with mixed nested dict input
    complex_input = {
        "text": {"description": "A string parameter", "value": "test text"},
        "number": 42,
        "flag": True
    }
    result = tool._parse_args(complex_input)
    assert result["text"] == "test text"
    assert isinstance(result["text"], str)
    assert result["number"] == 42
    assert isinstance(result["number"], int)
    assert result["flag"] is True
    assert isinstance(result["flag"], bool)


def test_invoke_with_nested_dict():
    """Test that invoking a tool with nested dict input works correctly."""
    def test_func(query: str) -> str:
        return f"Processed: {query}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="StringTool",
        description="A tool that processes string input"
    )
    
    # Test invoking with nested dict input
    nested_input = {"query": {"description": "A string input parameter", "value": "test value"}}
    result = tool.invoke(nested_input)
    assert result == "Processed: test value"


def test_nested_dict_without_value_key():
    """Test that nested dictionaries without 'value' field raise appropriate errors."""
    def test_func(query: str) -> str:
        return f"Processed: {query}"
    
    tool = CrewStructuredTool.from_function(
        func=test_func,
        name="StringTool",
        description="A tool that processes string input"
    )
    
    # Test with nested dict without 'value' key
    invalid_input = {"query": {"description": "A string input parameter", "other_key": "test"}}
    with pytest.raises(ValueError):
        tool._parse_args(invalid_input)
