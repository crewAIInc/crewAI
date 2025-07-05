"""Base classes for agent evaluation.

This module provides the foundational classes and models for the agent evaluation
framework, including the abstract base evaluator interface, scoring models, and
metric categories.
"""

import abc
from collections import defaultdict
import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.task import Task
from crewai.llm import BaseLLM
from crewai.utilities.llm_utils import create_llm

class MetricCategory(enum.Enum):
    """Categories of evaluation metrics."""
    GOAL_ALIGNMENT = "goal_alignment"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    SEMANTIC_QUALITY = "semantic_quality"
    TOOL_USAGE = "tool_usage"
    STEP_EFFICIENCY = "step_efficiency"
    REASONING_EFFICIENCY = "reasoning_efficiency"
    OVERALL = "overall"


class EvaluationScore(BaseModel):
    """Standardized evaluation score structure."""

    score: float = Field(
        default=5.0,
        description="Numeric score from 0-10 where 0 is worst and 10 is best",
        ge=0.0,
        le=10.0
    )
    feedback: str = Field(
        default="",
        description="Detailed feedback explaining the evaluation score"
    )
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw response from the evaluator (e.g., LLM)"
    )

    def __str__(self) -> str:
        """String representation of the score."""
        return f"Score: {self.score}/10 - {self.feedback}"


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluators.

    All evaluator implementations should inherit from this class and implement
    the evaluate method.
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the evaluator.

        Args:
            llm: Optional LLM instance for evaluations that require an LLM.
                 If not provided, a default LLM will be used.
        """
        self.llm = create_llm(llm)

    @property
    @abc.abstractmethod
    def metric_category(self) -> MetricCategory:
        """Get the evaluation metric category."""
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        """Evaluate an agent's performance on a task.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace containing step-by-step information
            final_output: The final output produced by the agent

        Returns:
            EvaluationScore: The evaluation score
        """
        pass


class AgentEvaluationResult(BaseModel):
    """Container for all evaluation results for an agent."""

    agent_id: str = Field(description="ID of the evaluated agent")
    task_id: str = Field(description="ID of the task that was executed")
    metrics: Dict[MetricCategory, EvaluationScore] = Field(
        default_factory=dict,
        description="Evaluation scores for each metric category"
    )
    overall_score: Optional[EvaluationScore] = Field(
        default=None,
        description="Holistic evaluation score"
    )

    def get_average_score(self) -> float:
        """Calculate the average score across all metrics."""
        if not self.metrics:
            return 0.0

        total = 0.0
        count = 0

        for category, score in self.metrics.items():
            if category != MetricCategory.OVERALL:
                total += score.score
                count += 1

        return total / count if count > 0 else 0.0

    def __str__(self) -> str:
        """String representation of the evaluation result."""
        result = f"Evaluation for Agent {self.agent_id} on Task {self.task_id}:\n"

        # Add individual metrics with enhanced display for tool usage
        for category, score in self.metrics.items():
            result += f"- {category.value.upper()}: {score.score}/10\n"

            # Special handling for tool usage to show detailed feedback
            if score.feedback:
                # Add indented detailed feedback
                detailed_feedback = "\n  ".join(score.feedback.split('\n'))
                result += f"  {detailed_feedback}\n"

        # Add overall score if available
        if self.overall_score:
            result += f"\nOVERALL: {self.overall_score.score}/10\n"
            result += f"Feedback: {self.overall_score.feedback}"

        return result


class AgentEvaluator:
    """Evaluator for agent execution.

    This class orchestrates the evaluation of agent execution using multiple
    evaluators and optionally a meta-evaluator.
    """

    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        meta_evaluator: Optional[BaseEvaluator] = None,
        crew = None
    ):
        """Initialize the agent evaluator.

        Args:
            evaluators: List of evaluators to use
            meta_evaluator: Optional meta-evaluator for holistic assessment
            crew: The crew to evaluate (optional)
        """
        self.crew = crew
        self.evaluators = evaluators
        self.meta_evaluator = meta_evaluator

        # Initialize for crew-based evaluation if crew is provided
        self.agent_evaluators = {}
        if crew is not None:
            for agent in crew.agents:
                self.agent_evaluators[agent.id] = self.evaluators.copy()

        # Callback will be set from outside
        self.callback = None

    def set_crew(self, crew):
        self.crew = crew
        return self

    def get_evaluation_results(self):
        """Get evaluation results for all agents in the crew.

        This method should be called after the crew has finished execution.
        The evaluator will use the traces collected by the callback.

        Returns:
            Dict[str, List[AgentEvaluationResult]]: Evaluation results by agent role
        """
        if not self.crew:
            raise ValueError("Cannot get evaluation results: no crew was provided to the evaluator.")

        if not self.callback:
            raise ValueError("Cannot get evaluation results: no callback was set. Use set_callback() method first.")

        print("\n📊 Running agent evaluations...\n")
        evaluation_results = defaultdict(list)

        # For each agent in the crew
        for agent in self.crew.agents:
            # Get the evaluator for this agent
            evaluator = self.agent_evaluators.get(agent.id)
            if not evaluator:
                continue

            # Find tasks executed by this agent
            for task in self.crew.tasks:
                if task.agent.id != agent.id:
                    continue

                # Get the trace for this agent-task pair
                trace = self.callback.get_trace(agent.id, task.id)
                if not trace:
                    print(f"Warning: No trace found for agent {agent.role} on task {task.description[:30]}...")
                    continue

                # Run evaluation directly using our evaluators
                result = self.evaluate(
                    agent=agent,
                    task=task,
                    execution_trace=trace,
                    final_output=task.output
                )
                evaluation_results[agent.role].append(result)

        return evaluation_results

    def set_callback(self, callback):
        """Set the callback used to collect execution traces.

        Args:
            callback: An EvaluationTraceCallback instance
        """
        self.callback = callback

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any
    ) -> AgentEvaluationResult:
        """Evaluate an agent's performance using all registered evaluators.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent

        Returns:
            AgentEvaluationResult: The evaluation results
        """
        # Initialize the result container
        result = AgentEvaluationResult(
            agent_id=str(agent.id),
            task_id=str(task.id)
        )

        # Run all evaluators
        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(
                    agent=agent,
                    task=task,
                    execution_trace=execution_trace,
                    final_output=final_output
                )
                result.metrics[evaluator.metric_category] = score
            except Exception as e:
                # Log error but continue with other evaluators
                print(f"Error in {evaluator.metric_category.value} evaluator: {str(e)}")

        # Run meta-evaluation if available
        if self.meta_evaluator:
            try:
                overall_score = self.meta_evaluator.evaluate(
                    agent=agent,
                    task=task,
                    execution_trace=execution_trace,
                    final_output=final_output,
                    evaluation_results=result.metrics
                )
                result.overall_score = overall_score
                result.metrics[MetricCategory.OVERALL] = overall_score
            except Exception as e:
                print(f"Error in meta-evaluation: {str(e)}")

        return result