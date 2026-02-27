"""Backward-compatibility shim — use ``crewai_a2a.utils.agent_card_signing`` instead."""

import warnings


warnings.warn(
    "'crewai.a2a.utils.agent_card_signing' has been moved to 'crewai_a2a.utils.agent_card_signing'. "
    "Please update your imports. The old path will be removed in v2.0.0.",
    FutureWarning,
    stacklevel=2,
)

from crewai_a2a.utils.agent_card_signing import *  # noqa: E402, F403
