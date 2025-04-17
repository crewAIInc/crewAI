import unittest
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from crewai.utilities.internal_instructor import InternalInstructor


class TestOutput(BaseModel):
    value: str


class TestInternalInstructor(unittest.TestCase):
    @patch("instructor.from_litellm")
    def test_tools_mode_for_regular_models(self, mock_from_litellm):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        mock_instructor = MagicMock()
        mock_from_litellm.return_value = mock_instructor
        
        instructor = InternalInstructor(
            content="Test content",
            model=TestOutput,
            llm=mock_llm
        )
        
        import instructor
        mock_from_litellm.assert_called_once_with(
            unittest.mock.ANY,
            mode=instructor.Mode.TOOLS
        )
    
    @patch("instructor.from_litellm")
    def test_parallel_tools_mode_for_custom_openai(self, mock_from_litellm):
        mock_llm = MagicMock()
        mock_llm.model = "custom_openai/some-model"
        mock_instructor = MagicMock()
        mock_from_litellm.return_value = mock_instructor
        
        instructor = InternalInstructor(
            content="Test content",
            model=TestOutput,
            llm=mock_llm
        )
        
        import instructor
        mock_from_litellm.assert_called_once_with(
            unittest.mock.ANY,
            mode=instructor.Mode.PARALLEL_TOOLS
        )
    
    @patch("instructor.from_litellm")
    def test_handling_list_response_in_to_pydantic(self, mock_from_litellm):
        mock_llm = MagicMock()
        mock_llm.model = "custom_openai/some-model"
        mock_instructor = MagicMock()
        mock_chat = MagicMock()
        mock_instructor.chat.completions.create.return_value = [
            TestOutput(value="test value")
        ]
        mock_from_litellm.return_value = mock_instructor
        
        instructor = InternalInstructor(
            content="Test content",
            model=TestOutput,
            llm=mock_llm
        )
        result = instructor.to_pydantic()
        
        assert isinstance(result, TestOutput)
        assert result.value == "test value"
