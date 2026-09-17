import pytest
from pydantic import BaseModel
from crewai.llms.base_llm import BaseLLM


class CityOutput(BaseModel):
    city: str
    population: int | None = None


class OtherOutput(BaseModel):
    name: str


def test_validate_structured_output_dict_success():
    """Test validating a dictionary directly into a Pydantic model."""
    data = {"city": "London", "population": 8900000}
    result = BaseLLM._validate_structured_output(data, CityOutput)
    assert isinstance(result, CityOutput)
    assert result.city == "London"
    assert result.population == 8900000


def test_validate_structured_output_dict_validation_error():
    """Test validating a dictionary with invalid schema raises ValueError."""
    data = {"unknown_field": "test"}
    with pytest.raises(ValueError, match="Failed to parse response into CityOutput"):
        BaseLLM._validate_structured_output(data, CityOutput)


def test_validate_structured_output_none_with_model_raises_value_error():
    """Test that None response with response_format raises ValueError instead of AttributeError."""
    with pytest.raises(ValueError, match="Failed to parse response into CityOutput: response is None"):
        BaseLLM._validate_structured_output(None, CityOutput)


def test_validate_structured_output_none_without_format_returns_none():
    """Test that None response with no response_format returns None."""
    result = BaseLLM._validate_structured_output(None, None)
    assert result is None


def test_validate_structured_output_base_model_passthrough():
    """Test passing an already instantiated BaseModel returns the model."""
    instance = CityOutput(city="Paris", population=2100000)
    result = BaseLLM._validate_structured_output(instance, CityOutput)
    assert result is instance


def test_validate_structured_output_base_model_conversion():
    """Test passing a different BaseModel converts when compatible."""
    class AltCity(BaseModel):
        city: str
    instance = AltCity(city="Berlin")
    result = BaseLLM._validate_structured_output(instance, CityOutput)
    assert isinstance(result, CityOutput)
    assert result.city == "Berlin"


def test_validate_structured_output_json_string():
    """Test standard JSON string validation."""
    json_str = '{"city": "Tokyo", "population": 14000000}'
    result = BaseLLM._validate_structured_output(json_str, CityOutput)
    assert isinstance(result, CityOutput)
    assert result.city == "Tokyo"


def test_validate_structured_output_embedded_json():
    """Test JSON extracted from markdown or prose."""
    response = 'Here is the city:\n```json\n{"city": "Madrid"}\n```'
    result = BaseLLM._validate_structured_output(response, CityOutput)
    assert isinstance(result, CityOutput)
    assert result.city == "Madrid"


def test_validate_structured_output_invalid_string():
    """Test non-JSON string raises ValueError."""
    with pytest.raises(ValueError, match="Failed to parse response into CityOutput"):
        BaseLLM._validate_structured_output("No JSON here at all", CityOutput)
