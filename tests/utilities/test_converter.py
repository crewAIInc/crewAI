import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from crewai.utilities.converter import (
    Converter,
    ConverterError,
    convert_to_model,
    convert_with_instructions,
    create_converter,
    get_conversion_instructions,
    handle_partial_json,
    is_gpt,
    validate_model,
)
from pydantic import BaseModel


# Sample Pydantic models for testing
class EmailResponse(BaseModel):
    previous_message_content: str


class EmailResponses(BaseModel):
    responses: list[EmailResponse]


class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    id: int
    data: SimpleModel


# Fixtures
@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.function_calling_llm = None
    agent.llm = Mock()
    return agent


# Tests for convert_to_model
def test_convert_to_model_with_valid_json():
    result = '{"name": "John", "age": 30}'
    output = convert_to_model(result, SimpleModel, None, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "John"
    assert output.age == 30


def test_convert_to_model_with_invalid_json():
    result = '{"name": "John", "age": "thirty"}'
    with patch("crewai.utilities.converter.handle_partial_json") as mock_handle:
        mock_handle.return_value = "Fallback result"
        output = convert_to_model(result, SimpleModel, None, None)
        assert output == "Fallback result"


def test_convert_to_model_with_no_model():
    result = "Plain text"
    output = convert_to_model(result, None, None, None)
    assert output == "Plain text"


def test_convert_to_model_with_special_characters():
    json_string_test = """
    {
        "responses": [
            {
                "previous_message_content": "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
            }
        ]
    }
    """
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
    )


def test_convert_to_model_with_escaped_special_characters():
    json_string_test = json.dumps(
        {
            "responses": [
                {
                    "previous_message_content": "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
                }
            ]
        }
    )
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Hi Tom,\r\n\r\nNiamh has chosen the Mika phonics on"
    )


def test_convert_to_model_with_multiple_special_characters():
    json_string_test = """
    {
        "responses": [
            {
                "previous_message_content": "Line 1\r\nLine 2\tTabbed\nLine 3\r\n\rEscaped newline"
            }
        ]
    }
    """
    output = convert_to_model(json_string_test, EmailResponses, None, None)
    assert isinstance(output, EmailResponses)
    assert len(output.responses) == 1
    assert (
        output.responses[0].previous_message_content
        == "Line 1\r\nLine 2\tTabbed\nLine 3\r\n\rEscaped newline"
    )


# Tests for validate_model
def test_validate_model_pydantic_output():
    result = '{"name": "Alice", "age": 25}'
    output = validate_model(result, SimpleModel, False)
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice"
    assert output.age == 25


def test_validate_model_json_output():
    result = '{"name": "Bob", "age": 40}'
    output = validate_model(result, SimpleModel, True)
    assert isinstance(output, dict)
    assert output == {"name": "Bob", "age": 40}


# Tests for handle_partial_json
def test_handle_partial_json_with_valid_partial():
    result = 'Some text {"name": "Charlie", "age": 35} more text'
    output = handle_partial_json(result, SimpleModel, False, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "Charlie"
    assert output.age == 35


def test_handle_partial_json_with_invalid_partial(mock_agent):
    result = "No valid JSON here"
    with patch("crewai.utilities.converter.convert_with_instructions") as mock_convert:
        mock_convert.return_value = "Converted result"
        output = handle_partial_json(result, SimpleModel, False, mock_agent)
        assert output == "Converted result"


# Tests for convert_with_instructions
@patch("crewai.utilities.converter.create_converter")
@patch("crewai.utilities.converter.get_conversion_instructions")
def test_convert_with_instructions_success(
    mock_get_instructions, mock_create_converter, mock_agent
):
    mock_get_instructions.return_value = "Instructions"
    mock_converter = Mock()
    mock_converter.to_pydantic.return_value = SimpleModel(name="David", age=50)
    mock_create_converter.return_value = mock_converter

    result = "Some text to convert"
    output = convert_with_instructions(result, SimpleModel, False, mock_agent)

    assert isinstance(output, SimpleModel)
    assert output.name == "David"
    assert output.age == 50


@patch("crewai.utilities.converter.create_converter")
@patch("crewai.utilities.converter.get_conversion_instructions")
def test_convert_with_instructions_failure(
    mock_get_instructions, mock_create_converter, mock_agent
):
    mock_get_instructions.return_value = "Instructions"
    mock_converter = Mock()
    mock_converter.to_pydantic.return_value = ConverterError("Conversion failed")
    mock_create_converter.return_value = mock_converter

    result = "Some text to convert"
    with patch("crewai.utilities.converter.Printer") as mock_printer:
        output = convert_with_instructions(result, SimpleModel, False, mock_agent)
        assert output == result
        mock_printer.return_value.print.assert_called_once()


# Tests for get_conversion_instructions
def test_get_conversion_instructions_gpt():
    mock_llm = Mock()
    mock_llm.openai_api_base = None
    with patch("crewai.utilities.converter.is_gpt", return_value=True):
        instructions = get_conversion_instructions(SimpleModel, mock_llm)
        assert instructions == "I'm gonna convert this raw text into valid JSON."


def test_get_conversion_instructions_non_gpt():
    mock_llm = Mock()
    with patch("crewai.utilities.converter.is_gpt", return_value=False):
        with patch("crewai.utilities.converter.PydanticSchemaParser") as mock_parser:
            mock_parser.return_value.get_schema.return_value = "Sample schema"
            instructions = get_conversion_instructions(SimpleModel, mock_llm)
            assert "Sample schema" in instructions


# Tests for is_gpt
def test_is_gpt_true():
    from langchain_openai import ChatOpenAI

    mock_llm = Mock(spec=ChatOpenAI)
    mock_llm.openai_api_base = None
    assert is_gpt(mock_llm) is True


def test_is_gpt_false():
    mock_llm = Mock()
    assert is_gpt(mock_llm) is False


class CustomConverter(Converter):
    pass


def test_create_converter_with_mock_agent():
    mock_agent = MagicMock()
    mock_agent.get_output_converter.return_value = MagicMock(spec=Converter)

    converter = create_converter(
        agent=mock_agent,
        llm=Mock(),
        text="Sample",
        model=SimpleModel,
        instructions="Convert",
    )

    assert isinstance(converter, Converter)
    mock_agent.get_output_converter.assert_called_once()


def test_create_converter_with_custom_converter():
    converter = create_converter(
        converter_cls=CustomConverter,
        llm=Mock(),
        text="Sample",
        model=SimpleModel,
        instructions="Convert",
    )

    assert isinstance(converter, CustomConverter)


def test_create_converter_fails_without_agent_or_converter_cls():
    with pytest.raises(
        ValueError, match="Either agent or converter_cls must be provided"
    ):
        create_converter(
            llm=Mock(), text="Sample", model=SimpleModel, instructions="Convert"
        )
