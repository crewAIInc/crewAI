"""Experimental CrewAI surface and compatibility exports."""

from __future__ import annotations

from typing import Any

# Conversational Flow graduated to ``crewai.flow``. Keep these exports so
# existing ``from crewai.experimental import ...`` imports continue to work.
# Everything still experimental is resolved lazily below to avoid Flow/Task
# import cycles.
from crewai.flow.conversational import (
    AgentMessage,
    ConversationConfig,
    ConversationEvent,
    ConversationMessage,
    ConversationState,
    RouterConfig,
)


_LAZY_FROM_AGENT_EXECUTOR = {"AgentExecutor", "CrewAgentExecutorFlow"}

_LAZY_FROM_EVALUATION = {
    "AgentEvaluationResult",
    "AgentEvaluator",
    "BaseEvaluator",
    "EvaluationScore",
    "EvaluationTraceCallback",
    "ExperimentResult",
    "ExperimentResults",
    "ExperimentRunner",
    "GoalAlignmentEvaluator",
    "MetricCategory",
    "ParameterExtractionEvaluator",
    "ReasoningEfficiencyEvaluator",
    "SemanticQualityEvaluator",
    "ToolInvocationEvaluator",
    "ToolSelectionEvaluator",
    "create_default_evaluator",
    "create_evaluation_callbacks",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve symbols whose modules import ``Flow`` or ``Task``.

    Eager re-exports would deadlock when ``Flow`` or ``Task`` is still loading.
    Callers like ``from crewai.experimental import AgentExecutor`` still work —
    the real import just runs lazily, after the original loader finishes.
    """
    if name in _LAZY_FROM_AGENT_EXECUTOR:
        from crewai.experimental.agent_executor import (
            AgentExecutor,
            CrewAgentExecutorFlow,
        )

        globals()["AgentExecutor"] = AgentExecutor
        globals()["CrewAgentExecutorFlow"] = CrewAgentExecutorFlow
        return globals()[name]

    if name in _LAZY_FROM_EVALUATION:
        from crewai.experimental import evaluation as _evaluation_mod

        for attr in _LAZY_FROM_EVALUATION:
            globals()[attr] = getattr(_evaluation_mod, attr)
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentEvaluationResult",
    "AgentEvaluator",
    "AgentExecutor",
    "AgentMessage",
    "BaseEvaluator",
    "ConversationConfig",
    "ConversationEvent",
    "ConversationMessage",
    "ConversationState",
    "CrewAgentExecutorFlow",  # Deprecated alias for AgentExecutor
    "EvaluationScore",
    "EvaluationTraceCallback",
    "ExperimentResult",
    "ExperimentResults",
    "ExperimentRunner",
    "GoalAlignmentEvaluator",
    "MetricCategory",
    "ParameterExtractionEvaluator",
    "ReasoningEfficiencyEvaluator",
    "RouterConfig",
    "SemanticQualityEvaluator",
    "ToolInvocationEvaluator",
    "ToolSelectionEvaluator",
    "create_default_evaluator",
    "create_evaluation_callbacks",
]
