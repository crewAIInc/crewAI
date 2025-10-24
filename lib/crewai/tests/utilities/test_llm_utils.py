import os
from typing import Any
from unittest.mock import patch

from crewai.cli.constants import DEFAULT_LLM_MODEL
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.llm_utils import create_llm
import pytest


def test_create_llm_with_llm_instance() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        existing_llm = LLM(model="gpt-4o")
        llm = create_llm(llm_value=existing_llm)
        assert llm is existing_llm


def test_create_llm_with_valid_model_string() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        llm = create_llm(llm_value="gpt-4o")
        assert isinstance(llm, BaseLLM)
        assert llm.model == "gpt-4o"


def test_create_llm_with_invalid_model_string() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        # For invalid model strings, create_llm succeeds but call() fails with API error
        llm = create_llm(llm_value="invalid-model")
        assert llm is not None
        assert isinstance(llm, BaseLLM)

        # The error should occur when making the actual API call
        # We expect some kind of API error (NotFoundError, etc.)
        with pytest.raises(Exception):  # noqa: B017
            llm.call(messages=[{"role": "user", "content": "Hello, world!"}])


def test_create_llm_with_unknown_object_missing_attributes() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        class UnknownObject:
            pass

        unknown_obj = UnknownObject()
        llm = create_llm(llm_value=unknown_obj)

        # Should succeed because str(unknown_obj) provides a model name
        assert llm is not None
        assert isinstance(llm, BaseLLM)


def test_create_llm_with_none_uses_default_model() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        with patch("crewai.utilities.llm_utils.DEFAULT_LLM_MODEL", DEFAULT_LLM_MODEL):
            llm = create_llm(llm_value=None)
            assert isinstance(llm, BaseLLM)
            assert llm.model == DEFAULT_LLM_MODEL


def test_create_llm_with_unknown_object() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        class UnknownObject:
            model_name = "gpt-4o"
            temperature = 0.7
            max_tokens = 1500

        unknown_obj = UnknownObject()
        llm = create_llm(llm_value=unknown_obj)
        assert isinstance(llm, BaseLLM)
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.7
        if hasattr(llm, 'max_tokens'):
            assert llm.max_tokens == 1500


def test_create_llm_from_env_with_unaccepted_attributes() -> None:
    with patch.dict(
        os.environ,
        {
            "OPENAI_MODEL_NAME": "gpt-3.5-turbo",
            "OPENAI_API_KEY": "fake-key",
            "AWS_ACCESS_KEY_ID": "fake-access-key",
            "AWS_SECRET_ACCESS_KEY": "fake-secret-key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        llm = create_llm(llm_value=None)
        assert isinstance(llm, BaseLLM)
        assert llm.model == "gpt-3.5-turbo"
        assert not hasattr(llm, "AWS_ACCESS_KEY_ID")
        assert not hasattr(llm, "AWS_SECRET_ACCESS_KEY")
        assert not hasattr(llm, "AWS_REGION_NAME")


def test_create_llm_with_partial_attributes() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        class PartialAttributes:
            model_name = "gpt-4o"
            # temperature is missing

        obj = PartialAttributes()
        llm = create_llm(llm_value=obj)
        assert isinstance(llm, BaseLLM)
        assert llm.model == "gpt-4o"
        assert llm.temperature is None  # Should handle missing attributes gracefully


def test_create_llm_with_invalid_type() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        # For integers, create_llm succeeds because str(42) becomes "42"
        llm = create_llm(llm_value=42)
        assert llm is not None
        assert isinstance(llm, BaseLLM)
        assert llm.model == "42"

        # The error should occur when making the actual API call
        with pytest.raises(Exception):  # noqa: B017
            llm.call(messages=[{"role": "user", "content": "Hello, world!"}])


def test_create_llm_openai_missing_api_key() -> None:
    """Test that create_llm raises error when OpenAI API key is missing"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises((ValueError, ImportError)) as exc_info:
            create_llm(llm_value="gpt-4o")

        error_message = str(exc_info.value).lower()
        assert "openai_api_key" in error_message or "api_key" in error_message


def test_create_llm_anthropic_missing_dependency() -> None:
    """Test that create_llm raises error when Anthropic dependency is missing"""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}, clear=True):
        with patch("crewai.llm.LLM.__new__", side_effect=ImportError('Anthropic native provider not available, to install: uv add "crewai[anthropic]"')):
            with pytest.raises(ImportError) as exc_info:
                create_llm(llm_value="anthropic/claude-3-sonnet")

            assert "Anthropic native provider not available, to install: uv add \"crewai[anthropic]\"" in str(exc_info.value)
