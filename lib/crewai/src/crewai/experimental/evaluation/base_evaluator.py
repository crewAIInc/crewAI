from __future__ import annotations

import abc
import enum
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llm import BaseLLM
from crewai.task import Task
from crewai.utilities.llm_utils import create_llm


if TYPE_CHECKING:
    from crewai.agent import Agent


class MetricCategory(enum.Enum):
    GOAL_ALIGNMENT = "goal_alignment"
    SEMANTIC_QUALITY = "semantic_quality"
    REASONING_EFFICIENCY = "reasoning_efficiency"
    TOOL_SELECTION = "tool_selection"
    PARAMETER_EXTRACTION = "parameter_extraction"
    TOOL_INVOCATION = "tool_invocation"

    def title(self):
        return self.value.replace("_", " ").title()


class EvaluationScore(BaseModel):
    score: float | None = Field(
        default=5.0,
        description="Numeric score from 0-10 where 0 is worst and 10 is best, None if not applicable",
        ge=0.0,
        le=10.0,
    )
    feedback: str = Field(
        default="", description="Detailed feedback explaining the evaluation score"
    )
    raw_response: str | None = Field(
        default=None, description="Raw response from the evaluator (e.g., LLM)"
    )

    def __str__(self) -> str:
        if self.score is None:
            return f"Score: N/A - {self.feedback}"
        return f"Score: {self.score:.1f}/10 - {self.feedback}"


class BaseEvaluator(abc.ABC):
    def __init__(self, llm: BaseLLM | None = None):
        self.llm: BaseLLM | None = create_llm(llm)

    @property
    @abc.abstractmethod
    def metric_category(self) -> MetricCategory:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: Any,
        task: Task | None = None,
    ) -> EvaluationScore:
        pass


class AgentEvaluationResult(BaseModel):
    agent_id: str = Field(description="ID of the evaluated agent")
    task_id: str = Field(description="ID of the task that was executed")
    metrics: dict[MetricCategory, EvaluationScore] = Field(
        default_factory=dict, description="Evaluation scores for each metric category"
    )


class AggregationStrategy(Enum):
    SIMPLE_AVERAGE = "simple_average"  # Equal weight to all tasks
    WEIGHTED_BY_COMPLEXITY = "weighted_by_complexity"  # Weight by task complexity
    BEST_PERFORMANCE = "best_performance"  # Use best scores across tasks
    WORST_PERFORMANCE = "worst_performance"  # Use worst scores across tasks


class AgentAggregatedEvaluationResult(BaseModel):
    agent_id: str = Field(default="", description="ID of the agent")
    agent_role: str = Field(default="", description="Role of the agent")
    task_count: int = Field(
        default=0, description="Number of tasks included in this aggregation"
    )
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.SIMPLE_AVERAGE,
        description="Strategy used for aggregation",
    )
    metrics: dict[MetricCategory, EvaluationScore] = Field(
        default_factory=dict, description="Aggregated metrics across all tasks"
    )
    task_results: list[str] = Field(
        default_factory=list, description="IDs of tasks included in this aggregation"
    )
    overall_score: float | None = Field(
        default=None, description="Overall score for this agent"
    )

    def __str__(self) -> str:
        result = f"Agent Evaluation: {self.agent_role}\n"
        result += f"Strategy: {self.aggregation_strategy.value}\n"
        result += f"Tasks evaluated: {self.task_count}\n"

        for category, score in self.metrics.items():
            result += f"\n\n- {category.value.upper()}: {score.score}/10\n"

            if score.feedback:
                detailed_feedback = "\n  ".join(score.feedback.split("\n"))
                result += f"  {detailed_feedback}\n"

        return result
