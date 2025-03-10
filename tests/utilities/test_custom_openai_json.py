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
            
            # The result should be a dictionary
            assert isinstance(result, dict)
            assert result.get("name") == "John"
            assert result.get("age") == 30
    
    def test_custom_openai_json_conversion_without_error(self, mock_llm_with_function_calling):
        """Test that JSON conversion works normally when Instructor doesn't raise an error."""
        # Mock Instructor that returns JSON without error
        mock_instructor = Mock()
        mock_instructor.to_json.return_value = {"name": "John", "age": 30}
        
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
            assert isinstance(result, dict)
            assert result == {"name": "John", "age": 30}
            
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
            # Call to_json method and expect it to raise a ConverterError
            with pytest.raises(ConverterError) as excinfo:
                converter.to_json()
            
            # Check the error message
            assert "invalid json" in str(excinfo.value).lower() or "expecting value" in str(excinfo.value).lower()
