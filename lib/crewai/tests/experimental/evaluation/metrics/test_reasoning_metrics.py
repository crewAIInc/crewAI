from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from crewai.experimental.evaluation.base_evaluator import EvaluationScore
from crewai.experimental.evaluation.metrics.reasoning_metrics import (
    ReasoningEfficiencyEvaluator,
)
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.llm_utils import LLM

from tests.experimental.evaluation.metrics.test_base_evaluation_metrics import (
    BaseEvaluationMetricsTest,
)


class TestReasoningEfficiencyEvaluator(BaseEvaluationMetricsTest):
    @pytest.fixture
    def mock_output(self):
        output = MagicMock(spec=TaskOutput)
        output.raw = "This is the test output"
        return output

    @pytest.fixture
    def llm_calls(self) -> List[Dict[str, Any]]:
        return [
            {
                "prompt": "How should I approach this task?",
                "response": "I'll first research the topic, then compile findings.",
                "timestamp": 1626987654,
            },
            {
                "prompt": "What resources should I use?",
                "response": "I'll use relevant academic papers and reliable websites.",
                "timestamp": 1626987754,
            },
            {
                "prompt": "How should I structure the output?",
                "response": "I'll organize information clearly with headings and bullet points.",
                "timestamp": 1626987854,
            },
        ]

    def test_insufficient_llm_calls(self, mock_agent, mock_task, mock_output):
        execution_trace = {"llm_calls": []}

        evaluator = ReasoningEfficiencyEvaluator()
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output=mock_output,
        )

        assert isinstance(result, EvaluationScore)
        assert result.score is None
        assert "Insufficient LLM calls" in result.feedback

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_successful_evaluation(
        self, mock_create_llm, mock_agent, mock_task, mock_output, llm_calls
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "scores": {
                "focus": 8.0,
                "progression": 7.0,
                "decision_quality": 7.5,
                "conciseness": 8.0,
                "loop_avoidance": 9.0
            },
            "overall_score": 7.9,
            "feedback": "The agent demonstrated good reasoning efficiency.",
            "optimization_suggestions": "The agent could improve by being more concise."
        }
        """
        mock_create_llm.return_value = mock_llm

        # Setup execution trace with sufficient LLM calls
        execution_trace = {"llm_calls": llm_calls}

        # Mock the _detect_loops method to return a simple result
        evaluator = ReasoningEfficiencyEvaluator(llm=mock_llm)
        evaluator._detect_loops = MagicMock(return_value=(False, []))

        # Evaluate
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output=mock_output,
        )

        # Assertions
        assert isinstance(result, EvaluationScore)
        assert result.score == 7.9
        assert "The agent demonstrated good reasoning efficiency" in result.feedback
        assert "Reasoning Efficiency Evaluation:" in result.feedback
        assert "• Focus: 8.0/10" in result.feedback

        # Verify LLM was called
        mock_llm.call.assert_called_once()

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_parse_error_handling(
        self, mock_create_llm, mock_agent, mock_task, mock_output, llm_calls
    ):
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = "Invalid JSON response"
        mock_create_llm.return_value = mock_llm

        # Setup execution trace
        execution_trace = {"llm_calls": llm_calls}

        # Mock the _detect_loops method
        evaluator = ReasoningEfficiencyEvaluator(llm=mock_llm)
        evaluator._detect_loops = MagicMock(return_value=(False, []))

        # Evaluate
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output=mock_output,
        )

        # Assertions for error handling
        assert isinstance(result, EvaluationScore)
        assert result.score is None
        assert "Failed to parse reasoning efficiency evaluation" in result.feedback

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_loop_detection(self, mock_create_llm, mock_agent, mock_task, mock_output):
        # Setup LLM calls with a repeating pattern
        repetitive_llm_calls = [
            {
                "prompt": "How to solve?",
                "response": "I'll try method A",
                "timestamp": 1000,
            },
            {
                "prompt": "Let me try method A",
                "response": "It didn't work",
                "timestamp": 1100,
            },
            {
                "prompt": "How to solve?",
                "response": "I'll try method A again",
                "timestamp": 1200,
            },
            {
                "prompt": "Let me try method A",
                "response": "It didn't work",
                "timestamp": 1300,
            },
            {
                "prompt": "How to solve?",
                "response": "I'll try method A one more time",
                "timestamp": 1400,
            },
        ]

        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "scores": {
                "focus": 6.0,
                "progression": 3.0,
                "decision_quality": 4.0,
                "conciseness": 6.0,
                "loop_avoidance": 2.0
            },
            "overall_score": 4.2,
            "feedback": "The agent is stuck in a reasoning loop.",
            "optimization_suggestions": "The agent should try different approaches when one fails."
        }
        """
        mock_create_llm.return_value = mock_llm

        execution_trace = {"llm_calls": repetitive_llm_calls}

        evaluator = ReasoningEfficiencyEvaluator(llm=mock_llm)

        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output=mock_output,
        )

        assert isinstance(result, EvaluationScore)
        assert result.score == 4.2
        assert "• Loop Avoidance: 2.0/10" in result.feedback
