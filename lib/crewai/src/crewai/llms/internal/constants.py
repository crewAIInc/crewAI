"""Deprecated: Use crewai.llm.internal.constants instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.internal.constants is deprecated. Use crewai.llm.internal.constants instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.internal.constants import *  # noqa: E402, F403
