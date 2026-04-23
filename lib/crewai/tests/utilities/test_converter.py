# Tests for enums
from enum import Enum
import json
import os
from unittest.mock import MagicMock, Mock, patch

from crewai.llm import LLM
from crewai.utilities.converter import (
    Converter,
    ConverterError,
    convert_to_model,
    convert_with_instructions,
    create_converter,
    generate_model_description,
    get_conversion_instructions,
    handle_partial_json,
    validate_model,
)
from pydantic import BaseModel
import pytest


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


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class CustomConverter(Converter):
    pass


# Fixtures
@pytest.fixture
def mock_agent() -> Mock:
    agent = Mock()
    agent.function_calling_llm = None
    agent.llm = Mock()
    return agent


# Tests for convert_to_model
def test_convert_to_model_with_valid_json() -> None:
    result = '{"name": "John", "age": 30}'
    output = convert_to_model(result, SimpleModel, None, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "John"
    assert output.age == 30


def test_convert_to_model_with_invalid_json() -> None:
    result = '{"name": "John", "age": "thirty"}'
    with patch("crewai.utilities.converter.handle_partial_json") as mock_handle:
        mock_handle.return_value = "Fallback result"
        output = convert_to_model(result, SimpleModel, None, None)
        assert output == "Fallback result"


def test_convert_to_model_with_no_model() -> None:
    result = "Plain text"
    output = convert_to_model(result, None, None, None)
    assert output == "Plain text"


def test_convert_to_model_with_special_characters() -> None:
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


def test_convert_to_model_with_escaped_special_characters() -> None:
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


def test_convert_to_model_with_multiple_special_characters() -> None:
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
def test_validate_model_pydantic_output() -> None:
    result = '{"name": "Alice", "age": 25}'
    output = validate_model(result, SimpleModel, False)
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice"
    assert output.age == 25


def test_validate_model_json_output() -> None:
    result = '{"name": "Bob", "age": 40}'
    output = validate_model(result, SimpleModel, True)
    assert isinstance(output, dict)
    assert output == {"name": "Bob", "age": 40}


# Tests for handle_partial_json
def test_handle_partial_json_with_valid_partial() -> None:
    result = 'Some text {"name": "Charlie", "age": 35} more text'
    output = handle_partial_json(result, SimpleModel, False, None)
    assert isinstance(output, SimpleModel)
    assert output.name == "Charlie"
    assert output.age == 35


def test_handle_partial_json_with_invalid_partial(mock_agent: Mock) -> None:
    result = "No valid JSON here"
    with patch("crewai.utilities.converter.convert_with_instructions") as mock_convert:
        mock_convert.return_value = "Converted result"
        output = handle_partial_json(result, SimpleModel, False, mock_agent)
        assert output == "Converted result"


# Tests for convert_with_instructions
@patch("crewai.utilities.converter.create_converter")
@patch("crewai.utilities.converter.get_conversion_instructions")
def test_convert_with_instructions_success(
    mock_get_instructions: Mock, mock_create_converter: Mock, mock_agent: Mock
) -> None:
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
    mock_get_instructions: Mock, mock_create_converter: Mock, mock_agent: Mock
) -> None:
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
def test_get_conversion_instructions_gpt() -> None:
    llm = LLM(model="gpt-4o-mini")
    with patch.object(LLM, "supports_function_calling") as supports_function_calling:
        supports_function_calling.return_value = True
        instructions = get_conversion_instructions(SimpleModel, llm)
        # Now using OpenAPI schema format for all models
        assert "Format your final answer according to the following OpenAPI schema:" in instructions
        assert '"type": "json_schema"' in instructions
        assert '"name": "SimpleModel"' in instructions
        assert "Do not include the OpenAPI schema in the final output" in instructions


def test_get_conversion_instructions_non_gpt() -> None:
    llm = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")
    with patch.object(LLM, "supports_function_calling", return_value=False):
        instructions = get_conversion_instructions(SimpleModel, llm)
        # Now using OpenAPI schema format for all models
        assert "Format your final answer according to the following OpenAPI schema:" in instructions
        assert '"type": "json_schema"' in instructions
        assert '"name": "SimpleModel"' in instructions
        assert "Do not include the OpenAPI schema in the final output" in instructions


# Tests for is_gpt
def test_supports_function_calling_true() -> None:
    llm = LLM(model="gpt-4o")
    assert llm.supports_function_calling() is True


def test_supports_function_calling_false() -> None:
    llm = LLM(model="non-existent-model", is_litellm=True)
    assert llm.supports_function_calling() is False


def test_create_converter_with_mock_agent() -> None:
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


def test_create_converter_with_custom_converter() -> None:
    converter = create_converter(
        converter_cls=CustomConverter,
        llm=LLM(model="gpt-4o-mini"),
        text="Sample",
        model=SimpleModel,
        instructions="Convert",
    )

    assert isinstance(converter, CustomConverter)


def test_create_converter_fails_without_agent_or_converter_cls() -> None:
    with pytest.raises(
        ValueError, match="Either agent or converter_cls must be provided"
    ):
        create_converter(
            llm=Mock(), text="Sample", model=SimpleModel, instructions="Convert"
        )


def test_generate_model_description_simple_model() -> None:
    description = generate_model_description(SimpleModel)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "SimpleModel"
    assert description["json_schema"]["strict"] is True
    assert "name" in description["json_schema"]["schema"]["properties"]
    assert "age" in description["json_schema"]["schema"]["properties"]


def test_generate_model_description_nested_model() -> None:
    description = generate_model_description(NestedModel)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "NestedModel"
    assert description["json_schema"]["strict"] is True
    assert "id" in description["json_schema"]["schema"]["properties"]
    assert "data" in description["json_schema"]["schema"]["properties"]


def test_generate_model_description_optional_field() -> None:
    class ModelWithOptionalField(BaseModel):
        name: str
        age: int | None

    description = generate_model_description(ModelWithOptionalField)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "ModelWithOptionalField"
    assert description["json_schema"]["strict"] is True


def test_generate_model_description_list_field() -> None:
    class ModelWithListField(BaseModel):
        items: list[int]

    description = generate_model_description(ModelWithListField)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "ModelWithListField"
    assert description["json_schema"]["strict"] is True


def test_generate_model_description_dict_field() -> None:
    class ModelWithDictField(BaseModel):
        attributes: dict[str, int]

    description = generate_model_description(ModelWithDictField)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "ModelWithDictField"
    assert description["json_schema"]["strict"] is True


@pytest.mark.vcr()
def test_convert_with_instructions() -> None:
    llm = LLM(model="gpt-4o-mini")
    sample_text = "Name: Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    # Act
    output = converter.to_pydantic()

    # Assert
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice"
    assert output.age == 30


@pytest.mark.vcr()
def test_converter_with_llama3_2_model() -> None:
    llm = LLM(model="openrouter/meta-llama/llama-3.2-3b-instruct")
    sample_text = "Name: Alice Llama, Age: 30"
    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )
    output = converter.to_pydantic()
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice Llama"
    assert output.age == 30


def test_converter_with_llama3_1_model() -> None:
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = True
    llm.call.return_value = '{"name": "Alice Llama", "age": 30}'

    sample_text = "Name: Alice Llama, Age: 30"
    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )
    output = converter.to_pydantic()
    assert isinstance(output, SimpleModel)
    assert output.name == "Alice Llama"
    assert output.age == 30


@pytest.mark.vcr()
def test_converter_with_nested_model() -> None:
    llm = LLM(model="gpt-4o-mini")
    sample_text = "Name: John Doe\nAge: 30\nAddress: 123 Main St, Anytown, 12345"

    instructions = get_conversion_instructions(Person, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=Person,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, Person)
    assert output.name == "John Doe"
    assert output.age == 30
    assert isinstance(output.address, Address)
    assert output.address.street == "123 Main St"
    assert output.address.city == "Anytown"
    assert output.address.zip_code == "12345"


# ---------------------------------------------------------------------------
# InternalInstructor — base_url forwarding (Anthropic + Azure + regression)
# ---------------------------------------------------------------------------

from crewai.utilities.internal_instructor import InternalInstructor  # noqa: E402
import instructor  # noqa: E402 — ensure cached before sys.modules patches in tests below


def _make_llm(
    provider: str,
    model: str = "test-model",
    base_url: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    azure_deployment: str | None = None,
    is_litellm: bool = False,
) -> Mock:
    llm = Mock()
    llm.provider = provider
    llm.model = model
    llm.base_url = base_url
    llm.api_base = api_base
    llm.api_key = api_key
    llm.api_version = api_version
    llm.azure_deployment = azure_deployment
    llm.is_litellm = is_litellm
    return llm


class TestInternalInstructorBaseUrl:
    """Verify _create_instructor_client routes to the correct SDK client.

    Note: anthropic is an optional dependency that may not be installed in the
    test environment. Tests that exercise the Anthropic path inject a fake
    anthropic module via sys.modules and use create=True on instructor.from_anthropic
    since that attribute is conditionally exported only when anthropic is installed.
    """

    def test_anthropic_with_base_url(self) -> None:
        llm = _make_llm("anthropic", base_url="https://my-anthropic-proxy.example.com")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None

        fake_anthropic = MagicMock()
        mock_from_anthropic = Mock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             patch("instructor.from_anthropic", mock_from_anthropic, create=True):
            client = inst._create_instructor_client()

        fake_anthropic.Anthropic.assert_called_once_with(base_url="https://my-anthropic-proxy.example.com")
        mock_from_anthropic.assert_called_once_with(fake_anthropic.Anthropic.return_value)
        assert client is mock_from_anthropic.return_value

    def test_anthropic_with_base_url_and_api_key(self) -> None:
        llm = _make_llm(
            "anthropic",
            base_url="https://my-anthropic-proxy.example.com",
            api_key="sk-ant-test",
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None

        fake_anthropic = MagicMock()
        mock_from_anthropic = Mock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             patch("instructor.from_anthropic", mock_from_anthropic, create=True):
            client = inst._create_instructor_client()

        fake_anthropic.Anthropic.assert_called_once_with(
            base_url="https://my-anthropic-proxy.example.com", api_key="sk-ant-test"
        )
        mock_from_anthropic.assert_called_once_with(fake_anthropic.Anthropic.return_value)
        assert client is mock_from_anthropic.return_value

    def test_anthropic_api_base_normalised_to_base_url(self) -> None:
        """api_base should be used as base_url when base_url is absent."""
        llm = _make_llm("anthropic", base_url=None, api_base="https://proxy.example.com")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None

        fake_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             patch("instructor.from_anthropic", Mock(), create=True):
            inst._create_instructor_client()

        fake_anthropic.Anthropic.assert_called_once_with(base_url="https://proxy.example.com")

    @patch("instructor.from_provider")
    def test_anthropic_without_base_url_uses_from_provider(
        self, mock_from_provider: Mock
    ) -> None:
        llm = _make_llm("anthropic", model="claude-3-5-sonnet")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with("anthropic/claude-3-5-sonnet")

    @patch("openai.AzureOpenAI")
    @patch("instructor.from_openai")
    def test_azure_with_base_url(
        self, mock_from_openai: Mock, mock_azure_cls: Mock
    ) -> None:
        llm = _make_llm(
            "azure", model="gpt-4o", base_url="https://my-resource.openai.azure.com"
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_azure_cls.assert_called_once_with(azure_endpoint="https://my-resource.openai.azure.com")
        mock_from_openai.assert_called_once_with(mock_azure_cls.return_value)

    @patch("openai.AzureOpenAI")
    @patch("instructor.from_openai")
    def test_azure_openai_alias_with_api_version(
        self, mock_from_openai: Mock, mock_azure_cls: Mock
    ) -> None:
        llm = _make_llm(
            "azure_openai",
            model="gpt-4o",
            base_url="https://my-resource.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        client = inst._create_instructor_client()

        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://my-resource.openai.azure.com",
            api_key="azure-key",
            api_version="2024-02-01",
        )
        mock_from_openai.assert_called_once_with(mock_azure_cls.return_value)
        assert client is mock_from_openai.return_value

    @patch("instructor.from_provider")
    def test_azure_without_base_url_uses_from_provider(
        self, mock_from_provider: Mock
    ) -> None:
        llm = _make_llm("azure", model="gpt-4o")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with("azure/gpt-4o")

    @patch("openai.OpenAI")
    @patch("instructor.from_openai")
    def test_openai_compatible_with_base_url(
        self, mock_from_openai: Mock, mock_openai_cls: Mock
    ) -> None:
        """Any OpenAI-compatible provider (vLLM, Ollama, Groq…) with base_url."""
        llm = _make_llm(
            "openai", model="llama3", base_url="http://localhost:8000/v1", api_key="x"
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_openai_cls.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="x")
        mock_from_openai.assert_called_once_with(mock_openai_cls.return_value)

    @patch("instructor.from_provider")
    def test_no_base_url_regression(
        self, mock_from_provider: Mock
    ) -> None:
        """Existing from_provider() path must be unchanged when no base_url is set."""
        llm = _make_llm("openai", model="gpt-4o")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with("openai/gpt-4o")

    @patch("openai.OpenAI")
    @patch("instructor.from_openai")
    def test_base_url_wins_over_api_base(
        self, mock_from_openai: Mock, mock_openai_cls: Mock
    ) -> None:
        """When both base_url and api_base are set, base_url takes precedence."""
        llm = _make_llm(
            "openai",
            model="gpt-4o",
            base_url="http://primary:8000/v1",
            api_base="http://secondary:8000/v1",
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_openai_cls.assert_called_once_with(base_url="http://primary:8000/v1")

    @patch("instructor.from_provider")
    def test_string_llm_strips_provider_prefix(
        self, mock_from_provider: Mock
    ) -> None:
        """String LLMs like 'anthropic/claude-3-5-sonnet' should not produce a double prefix."""
        inst = object.__new__(InternalInstructor)
        inst.llm = "anthropic/claude-3-5-sonnet"
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with("anthropic/claude-3-5-sonnet")

    def test_provider_routing_is_case_insensitive(self) -> None:
        """Provider names with non-standard casing should still route correctly."""
        llm = _make_llm("Anthropic", base_url="https://proxy.example.com")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None

        fake_anthropic = MagicMock()
        mock_from_anthropic = Mock()
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
             patch("instructor.from_anthropic", mock_from_anthropic, create=True):
            inst._create_instructor_client()

        fake_anthropic.Anthropic.assert_called_once_with(base_url="https://proxy.example.com")

    def test_anthropic_import_error_has_friendly_message(self) -> None:
        """Missing anthropic package raises ImportError with pip install hint."""
        llm = _make_llm("anthropic", base_url="https://proxy.example.com")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install anthropic"):
                inst._create_instructor_client()

    @patch("openai.AzureOpenAI")
    @patch("instructor.from_openai")
    def test_azure_with_deployment(
        self, mock_from_openai: Mock, mock_azure_cls: Mock
    ) -> None:
        """azure_deployment is forwarded to AzureOpenAI when set on the LLM."""
        llm = _make_llm(
            "azure",
            model="gpt-4o",
            base_url="https://my-resource.openai.azure.com",
            azure_deployment="my-gpt4o-deployment",
        )
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_azure_cls.assert_called_once_with(
            azure_endpoint="https://my-resource.openai.azure.com",
            azure_deployment="my-gpt4o-deployment",
        )
        mock_from_openai.assert_called_once_with(mock_azure_cls.return_value)

    @patch("instructor.from_provider")
    def test_object_llm_strips_model_prefix(
        self, mock_from_provider: Mock
    ) -> None:
        """An LLM object whose .model contains 'provider/model' must not double-prefix."""
        llm = _make_llm("anthropic", model="anthropic/claude-3-5-sonnet-20241022")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with(
            "anthropic/claude-3-5-sonnet-20241022"
        )

    @patch("instructor.from_provider")
    def test_empty_string_base_url_treated_as_absent(
        self, mock_from_provider: Mock
    ) -> None:
        """Empty string base_url must not trigger the custom-client path."""
        llm = _make_llm("openai", model="gpt-4o", base_url="")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._create_instructor_client()

        mock_from_provider.assert_called_once_with("openai/gpt-4o")

    def test_litellm_api_base_forwarded_in_to_pydantic(self) -> None:
        """When _litellm_api_base is set, to_pydantic passes it as api_base kwarg."""
        inst = object.__new__(InternalInstructor)
        inst.llm = _make_llm("openai", model="gpt-4o")
        inst.content = "test content"
        inst.model = SimpleModel
        inst.agent = None
        inst._litellm_api_base = "http://custom:8080/v1"
        inst._instructor_model_name = "gpt-4o"
        mock_client = MagicMock()
        inst._client = mock_client

        inst.to_pydantic()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("api_base") == "http://custom:8080/v1"

    def test_to_pydantic_sends_stripped_model_name(self) -> None:
        """to_pydantic must send the bare model id, not a provider-prefixed string."""
        inst = object.__new__(InternalInstructor)
        inst.llm = _make_llm("anthropic", model="anthropic/claude-3-5-sonnet-20241022")
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._litellm_api_base = None
        inst._instructor_model_name = "claude-3-5-sonnet-20241022"  # stripped by _create_instructor_client
        mock_client = MagicMock()
        inst._client = mock_client

        inst.to_pydantic()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert "anthropic/" not in call_kwargs["model"]

    @patch("instructor.from_provider")
    def test_from_provider_path_sets_full_prefixed_model_name(
        self, mock_from_provider: Mock
    ) -> None:
        """_instructor_model_name must include provider prefix for the from_provider path.

        Regression guard: an early assignment of the stripped model_string was
        incorrectly shared by both the direct-SDK and from_provider paths, causing
        to_pydantic to send a bare 'gpt-4o' to a from_provider client that routes
        on 'openai/gpt-4o'.
        """
        llm = _make_llm("openai", model="gpt-4o")
        inst = object.__new__(InternalInstructor)
        inst.llm = llm
        inst.content = "test"
        inst.model = SimpleModel
        inst.agent = None
        inst._litellm_api_base = None
        inst._instructor_model_name = ""

        inst._create_instructor_client()

        assert inst._instructor_model_name == "openai/gpt-4o"


# Tests for error handling
def test_converter_error_handling() -> None:
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = "Invalid JSON"
    sample_text = "Name: Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    with pytest.raises(ConverterError) as exc_info:
        converter.to_pydantic()

    assert "Failed to convert text into a Pydantic model" in str(exc_info.value)


# Tests for retry logic
def test_converter_retry_logic() -> None:
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.side_effect = [
        "Invalid JSON",
        "Still invalid",
        '{"name": "Retry Alice", "age": 30}',
    ]
    sample_text = "Name: Retry Alice, Age: 30"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
        max_attempts=3,
    )

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Retry Alice"
    assert output.age == 30
    assert llm.call.call_count == 3


# Tests for optional fields
def test_converter_with_optional_fields() -> None:
    class OptionalModel(BaseModel):
        name: str
        age: int | None

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    # Simulate the LLM's response with 'age' explicitly set to null
    llm.call.return_value = '{"name": "Bob", "age": null}'
    sample_text = "Name: Bob, age: None"

    instructions = get_conversion_instructions(OptionalModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=OptionalModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, OptionalModel)
    assert output.name == "Bob"
    assert output.age is None


# Tests for list fields
def test_converter_with_list_field() -> None:
    class ListModel(BaseModel):
        items: list[int]

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"items": [1, 2, 3]}'
    sample_text = "Items: 1, 2, 3"

    instructions = get_conversion_instructions(ListModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=ListModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, ListModel)
    assert output.items == [1, 2, 3]


def test_converter_with_enum() -> None:
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    class EnumModel(BaseModel):
        name: str
        color: Color

    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"name": "Alice", "color": "red"}'
    sample_text = "Name: Alice, Color: Red"

    instructions = get_conversion_instructions(EnumModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=EnumModel,
        instructions=instructions,
    )

    output = converter.to_pydantic()

    assert isinstance(output, EnumModel)
    assert output.name == "Alice"
    assert output.color == Color.RED


# Tests for ambiguous input
def test_converter_with_ambiguous_input() -> None:
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = False
    llm.call.return_value = '{"name": "Charlie", "age": "Not an age"}'
    sample_text = "Charlie is thirty years old"

    instructions = get_conversion_instructions(SimpleModel, llm)
    converter = Converter(
        llm=llm,
        text=sample_text,
        model=SimpleModel,
        instructions=instructions,
    )

    with pytest.raises(ConverterError) as exc_info:
        converter.to_pydantic()

    assert "failed to convert text into a pydantic model" in str(exc_info.value).lower()


# Tests for function calling support
def test_converter_with_function_calling() -> None:
    llm = Mock(spec=LLM)
    llm.supports_function_calling.return_value = True
    # Mock the llm.call to return a valid JSON string
    llm.call.return_value = '{"name": "Eve", "age": 35}'

    converter = Converter(
        llm=llm,
        text="Name: Eve, Age: 35",
        model=SimpleModel,
        instructions="Convert this text.",
    )

    output = converter.to_pydantic()

    assert isinstance(output, SimpleModel)
    assert output.name == "Eve"
    assert output.age == 35

    # Verify llm.call was called with correct parameters
    llm.call.assert_called_once()
    call_args = llm.call.call_args
    assert call_args[1]["response_model"] == SimpleModel


def test_generate_model_description_union_field() -> None:
    class UnionModel(BaseModel):
        field: int | str | None

    description = generate_model_description(UnionModel)
    # generate_model_description now returns a JSON schema dict
    assert isinstance(description, dict)
    assert description["type"] == "json_schema"
    assert description["json_schema"]["name"] == "UnionModel"
    assert description["json_schema"]["strict"] is True

def test_internal_instructor_with_openai_provider() -> None:
    """Test InternalInstructor with OpenAI provider using registry pattern."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with OpenAI provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "gpt-4o"
    mock_llm.provider = "openai"

    # Mock instructor client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Test", age=25)

    # Patch the instructor import at the method level
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client

        instructor = InternalInstructor(
            content="Test content",
            model=SimpleModel,
            llm=mock_llm
        )

        result = instructor.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Test"
        assert result.age == 25
        # Verify the method was called with the correct LLM
        mock_create_client.assert_called_once()


def test_internal_instructor_with_anthropic_provider() -> None:
    """Test InternalInstructor with Anthropic provider using registry pattern."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with Anthropic provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "claude-3-5-sonnet-20241022"
    mock_llm.provider = "anthropic"

    # Mock instructor client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Bob", age=25)

    # Patch the instructor import at the method level
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client

        instructor = InternalInstructor(
            content="Name: Bob, Age: 25",
            model=SimpleModel,
            llm=mock_llm
        )

        result = instructor.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Bob"
        assert result.age == 25
        # Verify the method was called with the correct LLM
        mock_create_client.assert_called_once()


def test_factory_pattern_registry_extensibility() -> None:
    """Test that the factory pattern registry works with different providers."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Test with OpenAI provider
    mock_llm_openai = Mock()
    mock_llm_openai.is_litellm = False
    mock_llm_openai.model = "gpt-4o-mini"
    mock_llm_openai.provider = "openai"

    mock_client_openai = Mock()
    mock_client_openai.chat.completions.create.return_value = SimpleModel(name="Alice", age=30)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client_openai

        instructor_openai = InternalInstructor(
            content="Name: Alice, Age: 30",
            model=SimpleModel,
            llm=mock_llm_openai
        )

        result_openai = instructor_openai.to_pydantic()

        assert isinstance(result_openai, SimpleModel)
        assert result_openai.name == "Alice"
        assert result_openai.age == 30

    # Test with Anthropic provider
    mock_llm_anthropic = Mock()
    mock_llm_anthropic.is_litellm = False
    mock_llm_anthropic.model = "claude-3-5-sonnet-20241022"
    mock_llm_anthropic.provider = "anthropic"

    mock_client_anthropic = Mock()
    mock_client_anthropic.chat.completions.create.return_value = SimpleModel(name="Bob", age=25)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client_anthropic

        instructor_anthropic = InternalInstructor(
            content="Name: Bob, Age: 25",
            model=SimpleModel,
            llm=mock_llm_anthropic
        )

        result_anthropic = instructor_anthropic.to_pydantic()

        assert isinstance(result_anthropic, SimpleModel)
        assert result_anthropic.name == "Bob"
        assert result_anthropic.age == 25

    # Test with Bedrock provider
    mock_llm_bedrock = Mock()
    mock_llm_bedrock.is_litellm = False
    mock_llm_bedrock.model = "claude-3-5-sonnet-20241022"
    mock_llm_bedrock.provider = "bedrock"

    mock_client_bedrock = Mock()
    mock_client_bedrock.chat.completions.create.return_value = SimpleModel(name="Charlie", age=35)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client_bedrock

        instructor_bedrock = InternalInstructor(
            content="Name: Charlie, Age: 35",
            model=SimpleModel,
            llm=mock_llm_bedrock
        )

        result_bedrock = instructor_bedrock.to_pydantic()

        assert isinstance(result_bedrock, SimpleModel)
        assert result_bedrock.name == "Charlie"
        assert result_bedrock.age == 35

    # Test with Google provider
    mock_llm_google = Mock()
    mock_llm_google.is_litellm = False
    mock_llm_google.model = "gemini-1.5-flash"
    mock_llm_google.provider = "google"

    mock_client_google = Mock()
    mock_client_google.chat.completions.create.return_value = SimpleModel(name="Diana", age=28)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client_google

        instructor_google = InternalInstructor(
            content="Name: Diana, Age: 28",
            model=SimpleModel,
            llm=mock_llm_google
        )

        result_google = instructor_google.to_pydantic()

        assert isinstance(result_google, SimpleModel)
        assert result_google.name == "Diana"
        assert result_google.age == 28

    # Test with Azure provider
    mock_llm_azure = Mock()
    mock_llm_azure.is_litellm = False
    mock_llm_azure.model = "gpt-4o"
    mock_llm_azure.provider = "azure"

    mock_client_azure = Mock()
    mock_client_azure.chat.completions.create.return_value = SimpleModel(name="Eve", age=32)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client_azure

        instructor_azure = InternalInstructor(
            content="Name: Eve, Age: 32",
            model=SimpleModel,
            llm=mock_llm_azure
        )

        result_azure = instructor_azure.to_pydantic()

        assert isinstance(result_azure, SimpleModel)
        assert result_azure.name == "Eve"
        assert result_azure.age == 32


def test_internal_instructor_with_bedrock_provider() -> None:
    """Test InternalInstructor with AWS Bedrock provider using registry pattern."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with Bedrock provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "claude-3-5-sonnet-20241022"
    mock_llm.provider = "bedrock"

    # Mock instructor client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Charlie", age=35)

    # Patch the instructor import at the method level
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client

        instructor = InternalInstructor(
            content="Name: Charlie, Age: 35",
            model=SimpleModel,
            llm=mock_llm
        )

        result = instructor.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Charlie"
        assert result.age == 35
        # Verify the method was called with the correct LLM
        mock_create_client.assert_called_once()


def test_internal_instructor_with_gemini_provider() -> None:
    """Test InternalInstructor with Google Gemini provider using registry pattern."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with Gemini provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "gemini-1.5-flash"
    mock_llm.provider = "google"

    # Mock instructor client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Diana", age=28)

    # Patch the instructor import at the method level
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client

        instructor = InternalInstructor(
            content="Name: Diana, Age: 28",
            model=SimpleModel,
            llm=mock_llm
        )

        result = instructor.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Diana"
        assert result.age == 28
        # Verify the method was called with the correct LLM
        mock_create_client.assert_called_once()


def test_internal_instructor_with_azure_provider() -> None:
    """Test InternalInstructor with Azure OpenAI provider using registry pattern."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with Azure provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "gpt-4o"
    mock_llm.provider = "azure"

    # Mock instructor client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Eve", age=32)

    # Patch the instructor import at the method level
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.return_value = mock_client

        instructor = InternalInstructor(
            content="Name: Eve, Age: 32",
            model=SimpleModel,
            llm=mock_llm
        )

        result = instructor.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Eve"
        assert result.age == 32
        # Verify the method was called with the correct LLM
        mock_create_client.assert_called_once()


def test_internal_instructor_unsupported_provider() -> None:
    """Test InternalInstructor with unsupported provider raises appropriate error."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with unsupported provider
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "unsupported-model"
    mock_llm.provider = "unsupported"

    # Mock the _create_instructor_client method to raise an error for unsupported providers
    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create_client:
        mock_create_client.side_effect = Exception("Unsupported provider: unsupported")

        # This should raise an error when trying to create the instructor client
        with pytest.raises(Exception) as exc_info:
            instructor = InternalInstructor(
                content="Test content",
                model=SimpleModel,
                llm=mock_llm
            )
            instructor.to_pydantic()

        # Verify it's the expected error
        assert "Unsupported provider" in str(exc_info.value)


def test_internal_instructor_real_unsupported_provider() -> None:
    """Test InternalInstructor with real unsupported provider using actual instructor library."""
    from crewai.utilities.internal_instructor import InternalInstructor

    # Mock LLM with unsupported provider that would actually fail with instructor
    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "unsupported-model"
    mock_llm.provider = "unsupported"
    mock_llm.base_url = None
    mock_llm.api_base = None
    mock_llm.api_key = None

    # This should raise a ConfigurationError from the real instructor library
    with pytest.raises(Exception) as exc_info:
        instructor = InternalInstructor(
            content="Test content",
            model=SimpleModel,
            llm=mock_llm
        )
        instructor.to_pydantic()

    # Verify it's a configuration error about unsupported provider
    assert "Unsupported provider" in str(exc_info.value) or "unsupported" in str(exc_info.value).lower()


# =========================================================================
# base_url forwarding tests (issue #5204)
# =========================================================================


def test_internal_instructor_no_base_url_uses_from_provider() -> None:
    """When LLM has no custom base_url, use instructor.from_provider normally."""
    from crewai.utilities.internal_instructor import InternalInstructor

    mock_llm = Mock()
    mock_llm.is_litellm = False
    mock_llm.model = "gpt-4o"
    mock_llm.provider = "openai"
    mock_llm.base_url = None
    mock_llm.api_base = None
    mock_llm.api_key = None

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = SimpleModel(name="Test", age=25)

    with patch.object(InternalInstructor, '_create_instructor_client') as mock_create:
        mock_create.return_value = mock_client

        inst = InternalInstructor(
            content="Test",
            model=SimpleModel,
            llm=mock_llm,
        )

        result = inst.to_pydantic()
        assert isinstance(result, SimpleModel)
        mock_create.assert_called_once()


