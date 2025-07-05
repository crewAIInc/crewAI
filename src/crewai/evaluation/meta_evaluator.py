"""Meta-evaluator for double-tier agent evaluation.

This module provides the second tier of evaluation that consolidates
results from individual metric evaluators and provides holistic assessment.
"""

import json
from typing import Any, Dict, List, Optional, Union

from crewai.llm import BaseLLM, LLM
from crewai.agent import Agent
from crewai.task import Task
from crewai.evaluation.base_evaluator import BaseEvaluator, EvaluationScore, MetricCategory, AgentEvaluationResult


class MetaEvaluator(BaseEvaluator):
    """Meta-evaluator that provides holistic assessment of agent performance."""

    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.OVERALL

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
        evaluation_results: Dict[MetricCategory, EvaluationScore] = None
    ) -> EvaluationScore:
        """Evaluate the overall performance based on individual metric results.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent
            evaluation_results: Results from individual metric evaluators

        Returns:
            EvaluationScore: Overall evaluation score
        """
        if not evaluation_results:
            return EvaluationScore(
                score=5.0,
                feedback="No metric evaluations available for meta-evaluation."
            )

        # Prepare the meta-evaluation prompt
        scores_summary = {}
        feedback_summary = {}

        for category, score in evaluation_results.items():
            if category == MetricCategory.OVERALL:
                continue  # Skip ourselves

            scores_summary[category.value] = score.score
            feedback_summary[category.value] = score.feedback

        prompt = [
            {"role": "system", "content": """You are an expert meta-evaluator providing a holistic assessment of AI agent performance.
Your job is to analyze the individual metric scores and synthesize an overall evaluation.

Score the agent's overall performance on a scale from 0-10 where:
- 0: Complete failure across all dimensions
- 5: Mixed results with some strengths and weaknesses
- 10: Exceptional performance across all metrics

Provide a detailed explanation justifying your score, focusing on:
1. Pattern identification across metrics
2. Relative importance of different metrics for this specific task
3. Key strengths to maintain in future iterations
4. Critical weaknesses that should be addressed
5. Specific recommendations for improvement

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}

Individual metric scores:
{json.dumps(scores_summary, indent=2)}

Metric feedback summaries:
{json.dumps(feedback_summary, indent=2)}

Provide a holistic evaluation of this agent's performance and specific improvement recommendations.
"""
            }
        ]

        # Get evaluation from LLM
        response = self.llm.call(prompt)

        try:
            # Parse the response
            evaluation_data = self._extract_json_from_text(response)
            return EvaluationScore(
                score=float(evaluation_data.get("score", 5.0)),
                feedback=evaluation_data.get("feedback", response)
            )
        except Exception as e:
            # Fallback if parsing fails
            return EvaluationScore(
                score=5.0,
                feedback=f"Failed to parse evaluation. Raw response: {response}"
            )

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from text that might contain markdown or other formatting."""
        # Same implementation as in other evaluators
        try:
            return json.loads(text)
        except:
            import re
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```|{[\s\S]*}'
            match = re.search(json_pattern, text)

            if match:
                try:
                    json_str = match.group(1) if match.group(1) else match.group(0)
                    return json.loads(json_str)
                except:
                    pass

        score_match = re.search(r'(?:score|rating):\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 5.0

        return {"score": score, "feedback": text}
