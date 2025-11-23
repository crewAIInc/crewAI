"""Deprecated: Use crewai.llm.base_llm instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.base_llm is deprecated. Use crewai.llm.base_llm instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.base_llm import *  # noqa: E402, F403
