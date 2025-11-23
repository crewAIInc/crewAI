"""Deprecated: Use crewai.llm.hooks instead.

.. deprecated:: 1.4.0
"""

import warnings


warnings.warn(
    "crewai.llms.hooks is deprecated. Use crewai.llm.hooks instead.",
    DeprecationWarning,
    stacklevel=2,
)

from crewai.llm.hooks import *  # noqa: E402, F403
