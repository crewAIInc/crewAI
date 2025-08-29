from unittest.mock import patch, MagicMock

from crewai.experimental.evaluation.base_evaluator import EvaluationScore
from crewai.experimental.evaluation.metrics.semantic_quality_metrics import (
    SemanticQualityEvaluator,
)
from tests.experimental.evaluation.metrics.test_base_evaluation_metrics import (
    BaseEvaluationMetricsTest,
)
from crewai.utilities.llm_utils import LLM


class TestSemanticQualityEvaluator(BaseEvaluationMetricsTest):
    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluate_success(
        self, mock_create_llm, mock_agent, mock_task, execution_trace
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "score": 8.5,
            "feedback": "The output is clear, coherent, and logically structured."
        }
        """
        mock_create_llm.return_value = mock_llm

        evaluator = SemanticQualityEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="This is a well-structured analysis of the data.",
        )

        assert isinstance(result, EvaluationScore)
        assert result.score == 8.5
        assert "clear, coherent" in result.feedback

        mock_llm.call.assert_called_once()
        prompt = mock_llm.call.call_args[0][0]
        assert len(prompt) >= 2
        assert "system" in prompt[0]["role"]
        assert "user" in prompt[1]["role"]
        assert mock_agent.role in prompt[1]["content"]
        assert mock_task.description in prompt[1]["content"]

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluate_with_empty_output(
        self, mock_create_llm, mock_agent, mock_task, execution_trace
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "score": 2.0,
            "feedback": "The output is empty or minimal, lacking substance."
        }
        """
        mock_create_llm.return_value = mock_llm

        evaluator = SemanticQualityEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="",
        )

        assert isinstance(result, EvaluationScore)
        assert result.score == 2.0
        assert "empty or minimal" in result.feedback

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluate_error_handling(
        self, mock_create_llm, mock_agent, mock_task, execution_trace
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = "Invalid JSON response"
        mock_create_llm.return_value = mock_llm

        evaluator = SemanticQualityEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="This is the output.",
        )

        assert isinstance(result, EvaluationScore)
        assert result.score is None
        assert "Failed to parse" in result.feedback
