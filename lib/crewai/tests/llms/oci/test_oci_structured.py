"""Unit tests for OCI provider structured output (mocked SDK)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel


class WeatherResponse(BaseModel):
    """Weather forecast response."""
    city: str
    temperature: float
    unit: str


def _make_json_response(data: dict) -> MagicMock:
    """Build a fake OCI response returning JSON text."""
    text_part = MagicMock()
    text_part.text = json.dumps(data)

    message = MagicMock()
    message.content = [text_part]
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    chat_response = MagicMock()
    chat_response.choices = [choice]
    chat_response.finish_reason = None
    chat_response.usage = MagicMock(prompt_tokens=10, completion_tokens=8, total_tokens=18)

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


def _make_cohere_json_response(data: dict) -> MagicMock:
    """Build a fake OCI Cohere response returning JSON text."""
    chat_response = MagicMock()
    chat_response.text = json.dumps(data)
    chat_response.tool_calls = None
    chat_response.finish_reason = "COMPLETE"
    chat_response.usage = MagicMock(prompt_tokens=8, completion_tokens=6, total_tokens=14)

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


def test_oci_completion_structured_output_generic(
    patch_oci_module, oci_unit_values
):
    """response_model should parse JSON response into a Pydantic model."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = _make_json_response(
        {"city": "NYC", "temperature": 72.0, "unit": "F"}
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    # Mock schema-related models
    patch_oci_module.generative_ai_inference.models.ResponseJsonSchema = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )
    patch_oci_module.generative_ai_inference.models.JsonSchemaResponseFormat = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "Weather in NYC?"}],
        response_model=WeatherResponse,
    )

    assert isinstance(result, WeatherResponse)
    assert result.city == "NYC"
    assert result.temperature == 72.0
    assert result.unit == "F"


def test_oci_completion_structured_output_cohere(
    patch_oci_module, oci_unit_values
):
    """Cohere models should use CohereResponseJsonFormat for structured output."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = _make_cohere_json_response(
        {"city": "London", "temperature": 15.0, "unit": "C"}
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    patch_oci_module.generative_ai_inference.models.ResponseJsonSchema = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )
    patch_oci_module.generative_ai_inference.models.CohereResponseJsonFormat = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "Weather in London?"}],
        response_model=WeatherResponse,
    )

    assert isinstance(result, WeatherResponse)
    assert result.city == "London"


def test_oci_completion_structured_output_with_fenced_json(
    patch_oci_module, oci_unit_values
):
    """Should handle JSON wrapped in markdown fences."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fenced = '```json\n{"city": "Tokyo", "temperature": 25.0, "unit": "C"}\n```'
    text_part = MagicMock()
    text_part.text = fenced

    message = MagicMock()
    message.content = [text_part]
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    chat_response = MagicMock()
    chat_response.choices = [choice]
    chat_response.finish_reason = None
    chat_response.usage = MagicMock(prompt_tokens=10, completion_tokens=8, total_tokens=18)

    response = MagicMock()
    response.data.chat_response = chat_response

    fake_client = MagicMock()
    fake_client.chat.return_value = response
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    patch_oci_module.generative_ai_inference.models.ResponseJsonSchema = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )
    patch_oci_module.generative_ai_inference.models.JsonSchemaResponseFormat = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        response_model=WeatherResponse,
    )

    assert isinstance(result, WeatherResponse)
    assert result.city == "Tokyo"


def test_oci_build_response_format_returns_none_without_model(
    patch_oci_module, oci_unit_values
):
    """_build_response_format should return None when no response_model."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm._build_response_format(None) is None


def test_oci_build_response_format_creates_json_schema(
    patch_oci_module, oci_unit_values
):
    """_build_response_format should create a JsonSchemaResponseFormat for generic models."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    patch_oci_module.generative_ai_inference.models.ResponseJsonSchema = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )
    patch_oci_module.generative_ai_inference.models.JsonSchemaResponseFormat = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm._build_response_format(WeatherResponse)

    assert result is not None
    patch_oci_module.generative_ai_inference.models.JsonSchemaResponseFormat.assert_called_once()
