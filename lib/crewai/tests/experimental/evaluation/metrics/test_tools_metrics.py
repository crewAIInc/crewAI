from unittest.mock import MagicMock, patch

from crewai.experimental.evaluation.metrics.tools_metrics import (
    ParameterExtractionEvaluator,
    ToolInvocationEvaluator,
    ToolSelectionEvaluator,
)
from crewai.utilities.llm_utils import LLM

from tests.experimental.evaluation.metrics.test_base_evaluation_metrics import (
    BaseEvaluationMetricsTest,
)


class TestToolSelectionEvaluator(BaseEvaluationMetricsTest):
    def test_no_tools_available(self, mock_task, mock_agent):
        # Create agent with no tools
        mock_agent.tools = []

        execution_trace = {"tool_uses": []}

        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score is None
        assert "no tools available" in result.feedback.lower()

    def test_tools_available_but_none_used(self, mock_agent, mock_task):
        mock_agent.tools = ["tool1", "tool2"]
        execution_trace = {"tool_uses": []}

        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score is None
        assert "had tools available but didn't use any" in result.feedback.lower()

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_successful_evaluation(self, mock_create_llm, mock_agent, mock_task):
        # Setup mock LLM response
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "overall_score": 8.5,
            "feedback": "The agent made good tool selections."
        }
        """
        mock_create_llm.return_value = mock_llm

        # Setup execution trace with tool uses
        execution_trace = {
            "tool_uses": [
                {
                    "tool": "search_tool",
                    "input": {"query": "test query"},
                    "output": "search results",
                },
                {"tool": "calculator", "input": {"expression": "2+2"}, "output": "4"},
            ]
        }

        evaluator = ToolSelectionEvaluator(llm=mock_llm)
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score == 8.5
        assert "The agent made good tool selections" in result.feedback

        # Verify LLM was called with correct prompt
        mock_llm.call.assert_called_once()
        prompt = mock_llm.call.call_args[0][0]
        assert isinstance(prompt, list)
        assert len(prompt) >= 2
        assert "system" in prompt[0]["role"]
        assert "user" in prompt[1]["role"]


class TestParameterExtractionEvaluator(BaseEvaluationMetricsTest):
    def test_no_tool_uses(self, mock_agent, mock_task):
        execution_trace = {"tool_uses": []}

        evaluator = ParameterExtractionEvaluator()
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score is None
        assert "no tool usage" in result.feedback.lower()

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_successful_evaluation(self, mock_create_llm, mock_agent, mock_task):
        mock_agent.tools = ["tool1", "tool2"]

        # Setup mock LLM response
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "overall_score": 9.0,
            "feedback": "The agent extracted parameters correctly."
        }
        """
        mock_create_llm.return_value = mock_llm

        # Setup execution trace with tool uses
        execution_trace = {
            "tool_uses": [
                {
                    "tool": "search_tool",
                    "input": {"query": "test query"},
                    "output": "search results",
                    "error": None,
                },
                {
                    "tool": "calculator",
                    "input": {"expression": "2+2"},
                    "output": "4",
                    "error": None,
                },
            ]
        }

        evaluator = ParameterExtractionEvaluator(llm=mock_llm)
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score == 9.0
        assert "The agent extracted parameters correctly" in result.feedback


class TestToolInvocationEvaluator(BaseEvaluationMetricsTest):
    def test_no_tool_uses(self, mock_agent, mock_task):
        execution_trace = {"tool_uses": []}

        evaluator = ToolInvocationEvaluator()
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score is None
        assert "no tool usage" in result.feedback.lower()

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_successful_evaluation(self, mock_create_llm, mock_agent, mock_task):
        mock_agent.tools = ["tool1", "tool2"]
        # Setup mock LLM response
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "overall_score": 8.0,
            "feedback": "The agent invoked tools correctly."
        }
        """
        mock_create_llm.return_value = mock_llm

        # Setup execution trace with tool uses
        execution_trace = {
            "tool_uses": [
                {
                    "tool": "search_tool",
                    "input": {"query": "test query"},
                    "output": "search results",
                },
                {"tool": "calculator", "input": {"expression": "2+2"}, "output": "4"},
            ]
        }

        evaluator = ToolInvocationEvaluator(llm=mock_llm)
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score == 8.0
        assert "The agent invoked tools correctly" in result.feedback

    @patch("crewai.utilities.llm_utils.create_llm")
    def test_evaluation_with_errors(self, mock_create_llm, mock_agent, mock_task):
        mock_agent.tools = ["tool1", "tool2"]
        # Setup mock LLM response
        mock_llm = MagicMock(spec=LLM)
        mock_llm.call.return_value = """
        {
            "overall_score": 5.5,
            "feedback": "The agent had some errors in tool invocation."
        }
        """
        mock_create_llm.return_value = mock_llm

        # Setup execution trace with tool uses including errors
        execution_trace = {
            "tool_uses": [
                {
                    "tool": "search_tool",
                    "input": {"query": "test query"},
                    "output": "search results",
                    "error": None,
                },
                {
                    "tool": "calculator",
                    "input": {"expression": "2+"},
                    "output": None,
                    "error": "Invalid expression",
                },
            ]
        }

        evaluator = ToolInvocationEvaluator(llm=mock_llm)
        result = evaluator.evaluate(
            agent=mock_agent,
            task=mock_task,
            execution_trace=execution_trace,
            final_output="Final output",
        )

        assert result.score == 5.5
        assert "The agent had some errors in tool invocation" in result.feedback
