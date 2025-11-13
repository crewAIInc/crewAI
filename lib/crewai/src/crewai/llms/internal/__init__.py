"""Deprecated: Use crewai.llm.internal instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.internal is deprecated. Use crewai.llm.internal instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.internal import *  # noqa: E402, F403
