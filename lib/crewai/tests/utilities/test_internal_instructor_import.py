"""Tests for InternalInstructor litellm import handling."""
import sys
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel


class SimpleModel(BaseModel):
    """Simple test model."""

    name: str
    age: int


def test_internal_instructor_raises_import_error_when_litellm_not_available():
    """Test that InternalInstructor raises ImportError when litellm is not available."""
    with patch.dict(sys.modules, {"litellm": None, "instructor": None}):
        with patch(
            "crewai.utilities.internal_instructor.LITELLM_AVAILABLE", False
        ):
            from crewai.utilities.internal_instructor import InternalInstructor

            with pytest.raises(ImportError) as exc_info:
                InternalInstructor(
                    content="test content",
                    model=SimpleModel,
                    llm="gpt-4o-mini",
                )

            assert "litellm" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()


def test_internal_instructor_works_when_litellm_available():
    """Test that InternalInstructor works normally when litellm is available."""
    from crewai.utilities.internal_instructor import LITELLM_AVAILABLE

    if not LITELLM_AVAILABLE:
        pytest.skip("litellm not available in test environment")

    from crewai.utilities.internal_instructor import InternalInstructor

    instructor = InternalInstructor(
        content="test content",
        model=SimpleModel,
        llm="gpt-4o-mini",
    )

    assert instructor.content == "test content"
    assert instructor.model == SimpleModel
    assert instructor.llm == "gpt-4o-mini"


def test_converter_handles_missing_litellm_gracefully():
    """Test that Converter handles missing litellm gracefully."""
    from crewai.utilities.converter import Converter, ConverterError

    mock_llm = Mock()
    mock_llm.supports_function_calling.return_value = True

    converter = Converter(
        llm=mock_llm,
        text="Name: Alice, Age: 30",
        model=SimpleModel,
        instructions="Convert this text.",
    )

    with patch(
        "crewai.utilities.internal_instructor.LITELLM_AVAILABLE", False
    ):
        with pytest.raises(ConverterError) as exc_info:
            converter.to_pydantic()

        assert "litellm" in str(exc_info.value).lower()


def test_converter_to_json_handles_missing_litellm():
    """Test that Converter.to_json handles missing litellm gracefully."""
    from crewai.utilities.converter import Converter, ConverterError

    mock_llm = Mock()
    mock_llm.supports_function_calling.return_value = True

    converter = Converter(
        llm=mock_llm,
        text="Name: Bob, Age: 25",
        model=SimpleModel,
        instructions="Convert this text.",
    )

    with patch(
        "crewai.utilities.internal_instructor.LITELLM_AVAILABLE", False
    ):
        result = converter.to_json()

        assert isinstance(result, ConverterError)
        assert "litellm" in str(result).lower()


def test_converter_fallback_when_function_calling_not_supported():
    """Test that Converter falls back to non-function-calling mode when litellm is not available."""
    from crewai.utilities.converter import Converter

    mock_llm = Mock()
    mock_llm.supports_function_calling.return_value = False
    mock_llm.call.return_value = '{"name": "Charlie", "age": 35}'

    converter = Converter(
        llm=mock_llm,
        text="Name: Charlie, Age: 35",
        model=SimpleModel,
        instructions="Convert this text.",
    )

    with patch(
        "crewai.utilities.internal_instructor.LITELLM_AVAILABLE", False
    ):
        result = converter.to_pydantic()

        assert isinstance(result, SimpleModel)
        assert result.name == "Charlie"
        assert result.age == 35
        mock_llm.call.assert_called_once()
