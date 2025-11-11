"""Deprecated: Use crewai.llm.hooks.base instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.hooks.base is deprecated. Use crewai.llm.hooks.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.hooks.base import *  # noqa: E402, F403
