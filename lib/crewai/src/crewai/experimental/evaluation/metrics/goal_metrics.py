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


class GoalAlignmentEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.GOAL_ALIGNMENT

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: Any,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}\nExpected output: {task.expected_output}\n"

        prompt: list[LLMMessage] = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing how well an AI agent's output aligns with its assigned task goal.

Score the agent's goal alignment on a scale from 0-10 where:
- 0: Complete misalignment, agent did not understand or attempt the task goal
- 5: Partial alignment, agent attempted the task but missed key requirements
- 10: Perfect alignment, agent fully satisfied all task requirements

Consider:
1. Did the agent correctly interpret the task goal?
2. Did the final output directly address the requirements?
3. Did the agent focus on relevant aspects of the task?
4. Did the agent provide all requested information or deliverables?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
Agent goal: {agent.goal}
{task_context}

Agent's final output:
{final_output}

Evaluate how well the agent's output aligns with the assigned task goal.
""",
            },
        ]
        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)  # type: ignore[arg-type]

        try:
            evaluation_data: dict[str, Any] = extract_json_from_llm_response(response)
            if evaluation_data is None:
                raise ValueError("Failed to extract evaluation data from LLM response")

            return EvaluationScore(
                score=evaluation_data.get("score", 0),
                feedback=evaluation_data.get("feedback", response),
                raw_response=response,
            )
        except Exception:
            return EvaluationScore(
                score=None,
                feedback=f"Failed to parse evaluation. Raw response: {response}",
                raw_response=response,
            )
