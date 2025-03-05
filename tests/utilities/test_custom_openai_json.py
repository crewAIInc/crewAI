import json
import pytest
from pydantic import BaseModel
from unittest.mock import Mock, patch

from crewai.llm import LLM
from crewai.utilities.converter import Converter


class SimpleModel(BaseModel):
    name: str
    age: int


class TestCustomOpenAIJson:
    def test_custom_openai_json_conversion_with_instructor_error(self):
        """Test that JSON conversion works with custom OpenAI backends when Instructor raises an error."""
        # Mock LLM that supports function calling
        llm = Mock(spec=LLM)
        llm.supports_function_calling.return_value = True
        llm.call.return_value = '{"name": "John", "age": 30}'
        
        # Mock Instructor that raises the specific error
        mock_instructor = Mock()
        mock_instructor.to_json.side_effect = Exception(
            "Instructor does not support multiple tool calls, use List[Model] instead"
        )
        
        # Create converter with mocked dependencies
        converter = Converter(
            llm=llm,
            text="Convert this to JSON",
            model=SimpleModel,
            instructions="Convert to JSON",
        )
        
        # Mock the _create_instructor method to return our mocked instructor
        with patch.object(converter, '_create_instructor', return_value=mock_instructor):
            # Call to_json method
            result = converter.to_json()
            
            # Verify that the fallback mechanism was used
            llm.call.assert_called_once()
            # The result is a JSON string, so we need to parse it
            parsed_result = json.loads(result)
            assert parsed_result == '{"name": "John", "age": 30}' or parsed_result == {"name": "John", "age": 30}
    
    def test_custom_openai_json_conversion_without_error(self):
        """Test that JSON conversion works normally when Instructor doesn't raise an error."""
        # Mock LLM that supports function calling
        llm = Mock(spec=LLM)
        llm.supports_function_calling.return_value = True
        
        # Mock Instructor that returns JSON without error
        mock_instructor = Mock()
        mock_instructor.to_json.return_value = '{"name": "John", "age": 30}'
        
        # Create converter with mocked dependencies
        converter = Converter(
            llm=llm,
            text="Convert this to JSON",
            model=SimpleModel,
            instructions="Convert to JSON",
        )
        
        # Mock the _create_instructor method to return our mocked instructor
        with patch.object(converter, '_create_instructor', return_value=mock_instructor):
            # Call to_json method
            result = converter.to_json()
            
            # Verify that the normal path was used (no fallback)
            llm.call.assert_not_called()
            assert json.loads(result) == {"name": "John", "age": 30}
