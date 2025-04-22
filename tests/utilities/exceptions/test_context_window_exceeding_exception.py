import pytest

from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)


def test_context_window_error_detection():
    """Test detection of different context window error formats."""
    assert LLMContextLengthExceededException("maximum context length exceeded")._is_context_limit_error(
        "maximum context length exceeded"
    )
    assert LLMContextLengthExceededException("expected a string with maximum length")._is_context_limit_error(
        "expected a string with maximum length"
    )
    
    litellm_error = "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: MistralException - Error code: 400 - {'object': 'error', 'message': \"This model's maximum context lenght is 15000 tokens. However, you requested 15018 tokens (12970) in the messages, 2048 in the completion). Please reduce the length of the messages or completion.\", type: 'BadRequestError', 'param': None, 'code': 400}"
    
    assert LLMContextLengthExceededException(litellm_error)._is_context_limit_error(
        litellm_error
    )
