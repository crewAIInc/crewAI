"""Type aliases for guardrails."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from crewai.lite_agent_output import LiteAgentOutput
from crewai.tasks.task_output import TaskOutput


GuardrailCallable: TypeAlias = Callable[
    [TaskOutput | LiteAgentOutput], tuple[bool, Any]
]

GuardrailType: TypeAlias = GuardrailCallable | str

GuardrailsType: TypeAlias = Sequence[GuardrailType] | GuardrailType
