"""Deprecated: Use crewai.llm.providers instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.providers is deprecated. Use crewai.llm.providers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.providers import *  # noqa: E402, F403
