from unittest.mock import MagicMock, patch

from crewai.experimental.evaluation.base_evaluator import EvaluationScore
from crewai.experimental.evaluation.metrics.goal_metrics import GoalAlignmentEvaluator
from crewai.utilities.llm_utils import LLM

from tests.experimental.evaluation.metrics.test_base_evaluation_metrics import (
    BaseEvaluationMetricsTest,
)


class TestGoalAlignmentEvaluator(BaseEvaluationMetricsTest):
    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluate_success(
        self, mock_create_llm, mock_agent, mock_task, execution_trace
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "score": 8.5,
            "feedback": "The agent correctly understood the task and produced relevant output."
        }
        """
        mock_create_llm.return_value = mock_llm

        evaluator = GoalAlignmentEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="This is the final output",
        )

        assert isinstance(result, EvaluationScore)
        assert result.score == 8.5
        assert "correctly understood the task" in result.feedback

        mock_llm.call.assert_called_once()
        prompt = mock_llm.call.call_args[0][0]
        assert len(prompt) >= 2
        assert "system" in prompt[0]["role"]
        assert "user" in prompt[1]["role"]
        assert mock_agent.role in prompt[1]["content"]
        assert mock_task.description in prompt[1]["content"]

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluate_error_handling(
        self, mock_create_llm, mock_agent, mock_task, execution_trace
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = "Invalid JSON response"
        mock_create_llm.return_value = mock_llm

        evaluator = GoalAlignmentEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="This is the final output",
        )

        assert isinstance(result, EvaluationScore)
        assert result.score is None
        assert "Failed to parse" in result.feedback
