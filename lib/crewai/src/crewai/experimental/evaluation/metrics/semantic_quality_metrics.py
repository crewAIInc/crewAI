from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.experimental.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationScore,
    MetricCategory,
)
from crewai.experimental.evaluation.json_parser import extract_json_from_llm_response
from crewai.task import Task
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent


class SemanticQualityEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.SEMANTIC_QUALITY

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: Any,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}"
        prompt: list[LLMMessage] = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing the semantic quality of an AI agent's output.

Score the semantic quality on a scale from 0-10 where:
- 0: Completely incoherent, confusing, or logically flawed output
- 5: Moderately clear and logical output with some issues
- 10: Exceptionally clear, coherent, and logically sound output

Consider:
1. Is the output well-structured and organized?
2. Is the reasoning logical and well-supported?
3. Is the language clear, precise, and appropriate for the task?
4. Are claims supported by evidence when appropriate?
5. Is the output free from contradictions and logical fallacies?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
{task_context}

Agent's final output:
{final_output}

Evaluate the semantic quality and reasoning of this output.
""",
            },
        ]

        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)

        try:
            evaluation_data: dict[str, Any] = extract_json_from_llm_response(response)
            if evaluation_data is None:
                raise ValueError("Failed to extract evaluation data from LLM response")
            return EvaluationScore(
                score=float(evaluation_data["score"])
                if evaluation_data.get("score") is not None
                else None,
                feedback=evaluation_data.get("feedback", response),
                raw_response=response,
            )
        except Exception:
            return EvaluationScore(
                score=None,
                feedback=f"Failed to parse evaluation. Raw response: {response}",
                raw_response=response,
            )
