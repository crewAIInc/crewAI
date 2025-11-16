"""Deprecated: Use crewai.llm.constants instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.constants is deprecated. Use crewai.llm.constants instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.constants import *  # noqa: E402, F403
