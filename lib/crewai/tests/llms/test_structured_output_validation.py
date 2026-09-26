"""Tests for structured output response validation."""

import pytest
from pydantic import BaseModel

from crewai.llms.base_llm import BaseLLM


class Answer(BaseModel):
    """Structured response model used by the tests."""

    city: str


def test_validate_structured_output_accepts_dict_response() -> None:
    result = BaseLLM._validate_structured_output({"city": "London"}, Answer)

    assert result == Answer(city="London")


def test_validate_structured_output_reports_none_response_as_parse_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"Failed to parse response into Answer: No JSON found in response",
    ):
        BaseLLM._validate_structured_output(None, Answer)


def test_validate_structured_output_preserves_string_json_response() -> None:
    result = BaseLLM._validate_structured_output('{"city": "London"}', Answer)

    assert result == Answer(city="London")
