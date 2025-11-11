"""Deprecated: Use crewai.llm.hooks.transport instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.hooks.transport is deprecated. Use crewai.llm.hooks.transport instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.hooks.transport import *  # noqa: E402, F403
