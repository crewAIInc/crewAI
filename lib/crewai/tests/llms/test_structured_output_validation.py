"""Tests for structured output response validation."""

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel
import pytest


class Answer(BaseModel):
    """Structured response model used by the tests."""

    city: str


def test_validate_structured_output_accepts_dict_response() -> None:
    """Validate a dictionary directly against the requested response model."""
    result = BaseLLM._validate_structured_output({"city": "London"}, Answer)

    assert result == Answer(city="London")


def test_validate_structured_output_reports_none_response_as_parse_error() -> None:
    """Report a missing response with the standard structured-output error."""
    with pytest.raises(
        ValueError,
        match=r"Failed to parse response into Answer: No JSON found in response",
    ):
        BaseLLM._validate_structured_output(None, Answer)


def test_validate_structured_output_preserves_string_json_response() -> None:
    """Keep parsing JSON strings into the requested response model."""
    result = BaseLLM._validate_structured_output('{"city": "London"}', Answer)

    assert result == Answer(city="London")
