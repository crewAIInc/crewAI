"""
GuardrailProvider protocol types.

These types mirror the upstream CrewAI GuardrailProvider contract from
crewAIInc/crewAI#4877. Once the upstream PR is merged, this module will
import directly from crewai.hooks.guardrails.

Until then, this module provides a standalone, drop-in implementation
that is structurally compatible with the upstream contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GuardrailRequest:
    """Provider-agnostic context for one pre-tool-call authorization decision."""

    tool_name: str
    tool_alias: str
    tool_input: Mapping[str, Any]
    agent_id: str | None = None
    agent_role: str | None = None
    task_description: str | None = None
    crew_id: str | None = None


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """Authorization verdict returned by a GuardrailProvider."""

    allow: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class GuardrailProvider(Protocol):
    """Contract for pluggable pre-tool-call authorization providers."""

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate whether the requested tool call should proceed."""
        ...
