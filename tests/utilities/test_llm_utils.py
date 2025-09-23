import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.llm_utils import create_llm

try:
    from litellm.exceptions import BadRequestError
except ImportError:
    BadRequestError = Exception


def test_create_llm_with_llm_instance():
    existing_llm = LLM(model="gpt-4o")
    llm = create_llm(llm_value=existing_llm)
    assert llm is existing_llm


def test_create_llm_with_valid_model_string():
    llm = create_llm(llm_value="gpt-4o")
    assert isinstance(llm, BaseLLM)
    assert llm.model == "gpt-4o"


def test_create_llm_with_invalid_model_string():
    # For invalid model strings, create_llm succeeds but call() fails with API error
    llm = create_llm(llm_value="invalid-model")
    assert llm is not None
    assert isinstance(llm, BaseLLM)

    # The error should occur when making the actual API call
    # We expect some kind of API error (NotFoundError, etc.)
    with pytest.raises(Exception):  # noqa: B017
        llm.call(messages=[{"role": "user", "content": "Hello, world!"}])


def test_create_llm_with_unknown_object_missing_attributes():
    class UnknownObject:
        pass

    unknown_obj = UnknownObject()
    llm = create_llm(llm_value=unknown_obj)

    # Should succeed because str(unknown_obj) provides a model name
    assert llm is not None
    assert isinstance(llm, BaseLLM)


def test_create_llm_with_none_uses_default_model():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}, clear=True):
        with patch("crewai.utilities.llm_utils.DEFAULT_LLM_MODEL", "gpt-4o-mini"):
            llm = create_llm(llm_value=None)
            assert isinstance(llm, BaseLLM)
            assert llm.model == "gpt-4o-mini"


def test_create_llm_with_unknown_object():
    class UnknownObject:
        model_name = "gpt-4o"
        temperature = 0.7
        max_tokens = 1500

    unknown_obj = UnknownObject()
    llm = create_llm(llm_value=unknown_obj)
    assert isinstance(llm, BaseLLM)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 1500


def test_create_llm_from_env_with_unaccepted_attributes():
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


def test_create_llm_with_partial_attributes():
    class PartialAttributes:
        model_name = "gpt-4o"
        # temperature is missing

    obj = PartialAttributes()
    llm = create_llm(llm_value=obj)
    assert isinstance(llm, BaseLLM)
    assert llm.model == "gpt-4o"
    assert llm.temperature is None  # Should handle missing attributes gracefully


def test_create_llm_with_invalid_type():
    # For integers, create_llm succeeds because str(42) becomes "42"
    llm = create_llm(llm_value=42)
    assert llm is not None
    assert isinstance(llm, BaseLLM)
    assert llm.model == "42"

    # The error should occur when making the actual API call
    with pytest.raises(Exception):  # noqa: B017
        llm.call(messages=[{"role": "user", "content": "Hello, world!"}])
