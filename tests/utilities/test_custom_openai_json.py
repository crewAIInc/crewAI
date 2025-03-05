import json
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.utilities.converter import Converter, ConverterError


class SimpleModel(BaseModel):
    name: str
    age: int


@pytest.fixture
def mock_llm_with_function_calling():
    """Create a mock LLM that supports function calling."""
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = True
    llm.call.return_value = '{"name": "John", "age": 30}'
    return llm


@pytest.fixture
def mock_instructor_with_error():
    """Create a mock Instructor that raises the specific error."""
    mock_instructor = Mock()
    mock_instructor.to_json.side_effect = Exception(
        "Instructor does not support multiple tool calls, use List[Model] instead"
    )
    return mock_instructor


class TestCustomOpenAIJson:
    def test_custom_openai_json_conversion_with_instructor_error(self, mock_llm_with_function_calling, mock_instructor_with_error):
        """Test that JSON conversion works with custom OpenAI backends when Instructor raises an error."""
        # Create converter with mocked dependencies
        converter = Converter(
            llm=mock_llm_with_function_calling,
            text="Convert this to JSON",
            model=SimpleModel,
            instructions="Convert to JSON",
        )
        
        # Mock the _create_instructor method to return our mocked instructor
        with patch.object(converter, '_create_instructor', return_value=mock_instructor_with_error):
            # Call to_json method
            result = converter.to_json()
            
            # Verify that the fallback mechanism was used
            mock_llm_with_function_calling.call.assert_called_once()
            
            # The result should be a JSON string
            assert isinstance(result, str)
            
            # The result might be a string representation of a JSON string
            # Try to parse it directly first, and if that fails, try to parse it as a string representation
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                # If it's a string representation of a JSON string, it will be surrounded by quotes
                # and have escaped quotes inside
                if result.startswith('"') and result.endswith('"'):
                    # Remove the surrounding quotes and unescape the string
                    unescaped = result[1:-1].replace('\\"', '"')
                    parsed_result = json.loads(unescaped)
            
            assert isinstance(parsed_result, dict)
            assert parsed_result.get("name") == "John"
            assert parsed_result.get("age") == 30
    
    def test_custom_openai_json_conversion_without_error(self, mock_llm_with_function_calling):
        """Test that JSON conversion works normally when Instructor doesn't raise an error."""
        # Mock Instructor that returns JSON without error
        mock_instructor = Mock()
        mock_instructor.to_json.return_value = '{"name": "John", "age": 30}'
        
        # Create converter with mocked dependencies
        converter = Converter(
            llm=mock_llm_with_function_calling,
            text="Convert this to JSON",
            model=SimpleModel,
            instructions="Convert to JSON",
        )
        
        # Mock the _create_instructor method to return our mocked instructor
        with patch.object(converter, '_create_instructor', return_value=mock_instructor):
            # Call to_json method
            result = converter.to_json()
            
            # Verify that the normal path was used (no fallback)
            mock_llm_with_function_calling.call.assert_not_called()
            
            # Verify the result matches the expected output
            assert result == '{"name": "John", "age": 30}'
            
    def test_custom_openai_json_conversion_with_invalid_json(self, mock_llm_with_function_calling):
        """Test that JSON conversion handles invalid JSON gracefully."""
        # Mock LLM to return invalid JSON
        mock_llm_with_function_calling.call.return_value = 'invalid json'
        
        # Mock Instructor that raises the specific error
        mock_instructor = Mock()
        mock_instructor.to_json.side_effect = Exception(
            "Instructor does not support multiple tool calls, use List[Model] instead"
        )
        
        # Create converter with mocked dependencies
        converter = Converter(
            llm=mock_llm_with_function_calling,
            text="Convert this to JSON",
            model=SimpleModel,
            instructions="Convert to JSON",
            max_attempts=1,  # Set max_attempts to 1 to avoid retries
        )
        
        # Mock the _create_instructor method to return our mocked instructor
        with patch.object(converter, '_create_instructor', return_value=mock_instructor):
            # Call to_json method
            result = converter.to_json()
            
            # The result should be a ConverterError instance
            assert isinstance(result, ConverterError)
            assert "invalid json" in str(result).lower() or "expecting value" in str(result).lower()
