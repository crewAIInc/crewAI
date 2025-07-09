from typing import Any, Dict

from crewai.agent import Agent
from crewai.task import Task

from crewai.evaluation.base_evaluator import BaseEvaluator, EvaluationScore, MetricCategory
from crewai.evaluation.json_parser import extract_json_from_llm_response

class GoalAlignmentEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.GOAL_ALIGNMENT

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        prompt = [
            {"role": "system", "content": """You are an expert evaluator assessing how well an AI agent's output aligns with its assigned task goal.

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
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Agent goal: {agent.goal}
Task description: {task.description}
Expected output: {task.expected_output}

Agent's final output:
{final_output}

Evaluate how well the agent's output aligns with the assigned task goal.
"""}
        ]

        response = self.llm.call(prompt)

        try:
            evaluation_data = extract_json_from_llm_response(response)
            return EvaluationScore(
                score=float(evaluation_data.get("score", None)),
                feedback=evaluation_data.get("feedback", response),
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Failed to parse evaluation. Raw response: {response}",
                raw_response=response
            )
