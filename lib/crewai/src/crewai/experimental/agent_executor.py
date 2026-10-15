"""Deprecated import path for the agent executor.

Import from ``crewai.agents.agent_executor`` instead. This module remains so
existing ``crewai.experimental.agent_executor`` imports keep working.
"""

from __future__ import annotations

import warnings

from crewai.agents.agent_executor import (
    AgentExecutor,
    AgentExecutorState,
    CrewAgentExecutorFlow,
)


__all__ = ["AgentExecutor", "AgentExecutorState", "CrewAgentExecutorFlow"]


warnings.warn(
    "crewai.experimental.agent_executor is deprecated and will be removed in a "
    "future release. Import AgentExecutor from crewai.agents.agent_executor.",
    DeprecationWarning,
    stacklevel=2,
)
