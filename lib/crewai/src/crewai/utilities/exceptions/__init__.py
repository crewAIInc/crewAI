"""Exceptions for crewAI."""

from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.exceptions.task_security_exceptions import (
    IterationLimitExceeded,
    PreTaskAttestationError,
    SpendLimitExceeded,
)


__all__ = [
    "IterationLimitExceeded",
    "LLMContextLengthExceededError",
    "PreTaskAttestationError",
    "SpendLimitExceeded",
]
