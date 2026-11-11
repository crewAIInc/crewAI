"""Deprecated import path for ``AgentExecutor``.

``AgentExecutor`` left ``crewai.experimental`` for ``crewai.agents.agent_executor``
in 1.15.29. This module keeps ``from crewai.experimental.agent_executor import
AgentExecutor`` working.
"""

from __future__ import annotations

import warnings

from crewai.agents.agent_executor import AgentExecutor, AgentExecutorState


warnings.warn(
    "crewai.experimental.agent_executor is deprecated. Import AgentExecutor "
    "and AgentExecutorState from crewai.agents.agent_executor instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AgentExecutor", "AgentExecutorState"]
