from typing import List
from unittest.mock import MagicMock, patch

from crewai.utilities.converter import ConverterError
from crewai.utilities.training_converter import TrainingConverter
from pydantic import BaseModel, Field


class TestModel(BaseModel):
    string_field: str = Field(description="A simple string field")
    list_field: List[str] = Field(description="A list of strings")
    number_field: float = Field(description="A number field")


class TestTrainingConverter:
    def setup_method(self):
        self.llm_mock = MagicMock()
        self.test_text = "Sample text for evaluation"
        self.test_instructions = "Convert to JSON format"
        self.converter = TrainingConverter(
            llm=self.llm_mock,
            text=self.test_text,
            model=TestModel,
            instructions=self.test_instructions,
        )

    @patch("crewai.utilities.converter.Converter.to_pydantic")
    def test_fallback_to_field_by_field(self, parent_to_pydantic_mock):
        parent_to_pydantic_mock.side_effect = ConverterError(
            "Failed to convert directly"
        )

        llm_responses = {
            "string_field": "test string value",
            "list_field": "- item1\n- item2\n- item3",
            "number_field": "8.5",
        }

        def llm_side_effect(messages):
            prompt = messages[1]["content"]
            if "string_field" in prompt:
                return llm_responses["string_field"]
            if "list_field" in prompt:
                return llm_responses["list_field"]
            if "number_field" in prompt:
                return llm_responses["number_field"]
            return "unknown field"

        self.llm_mock.call.side_effect = llm_side_effect

        result = self.converter.to_pydantic()

        assert result.string_field == "test string value"
        assert result.list_field == ["item1", "item2", "item3"]
        assert result.number_field == 8.5

        parent_to_pydantic_mock.assert_called_once()
        assert self.llm_mock.call.call_count == 3

    def test_ask_llm_for_field(self):
        field_name = "test_field"
        field_description = "This is a test field description"
        expected_response = "Test response"
        self.llm_mock.call.return_value = expected_response
        response = self.converter._ask_llm_for_field(field_name, field_description)

        assert response == expected_response
        self.llm_mock.call.assert_called_once()

        call_args = self.llm_mock.call.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert f"Extract the {field_name}" in call_args[0]["content"]
        assert call_args[1]["role"] == "user"
        assert field_name in call_args[1]["content"]
        assert field_description in call_args[1]["content"]

    def test_process_field_value_string(self):
        response = "  This is a string with extra whitespace  "
        result = self.converter._process_field_value(response, str)
        assert result == "This is a string with extra whitespace"

    def test_process_field_value_list_with_bullet_points(self):
        response = "- Item 1\n- Item 2\n- Item 3"
        result = self.converter._process_field_value(response, List[str])
        assert result == ["Item 1", "Item 2", "Item 3"]

    def test_process_field_value_list_with_json(self):
        response = '["Item 1", "Item 2", "Item 3"]'
        with patch("crewai.utilities.training_converter.json.loads") as json_mock:
            json_mock.return_value = ["Item 1", "Item 2", "Item 3"]
            result = self.converter._process_field_value(response, List[str])
            assert result == ["Item 1", "Item 2", "Item 3"]

    def test_process_field_value_float(self):
        response = "The quality score is 8.5 out of 10"
        result = self.converter._process_field_value(response, float)
        assert result == 8.5
