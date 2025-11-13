"""LLM implementations for crewAI.

.. deprecated:: 1.4.0
    The `crewai.llms` package is deprecated. Use `crewai.llm` instead.

    This package was reorganized from `crewai.llms.*` to `crewai.llm.*`.
    All submodules are redirected to their new locations in `crewai.llm.*`.

    Migration guide:
        Old: from crewai.llms.base_llm import BaseLLM
        New: from crewai.llm.base_llm import BaseLLM

        Old: from crewai.llms.hooks.base import BaseInterceptor
        New: from crewai.llm.hooks.base import BaseInterceptor

        Old: from crewai.llms.constants import OPENAI_MODELS
        New: from crewai.llm.constants import OPENAI_MODELS

        Or use top-level imports:
        from crewai import LLM, BaseLLM
"""

import warnings

from crewai.llm import LLM
from crewai.llm.base_llm import BaseLLM


# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'crewai.llms' package is deprecated and will be removed in a future version. "
    "Please use 'crewai.llm' (singular) instead. "
    "All submodules have been reorganized from 'crewai.llms.*' to 'crewai.llm.*'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LLM", "BaseLLM"]
