from unittest.mock import MagicMock, patch

import pytest
from crewai.crew import Crew
from crewai.experimental.evaluation.base_evaluator import (
    EvaluationScore,
    MetricCategory,
)
from crewai.experimental.evaluation.evaluation_display import (
    AgentAggregatedEvaluationResult,
)
from crewai.experimental.evaluation.experiment.result import ExperimentResults
from crewai.experimental.evaluation.experiment.runner import ExperimentRunner


class TestExperimentRunner:
    @pytest.fixture
    def mock_crew(self):
        return MagicMock(llm=Crew)

    @pytest.fixture
    def mock_evaluator_results(self):
        agent_evaluation = AgentAggregatedEvaluationResult(
            agent_id="Test Agent",
            agent_role="Test Agent Role",
            metrics={
                MetricCategory.GOAL_ALIGNMENT: EvaluationScore(
                    score=9,
                    feedback="Test feedback for goal alignment",
                    raw_response="Test raw response for goal alignment",
                ),
                MetricCategory.REASONING_EFFICIENCY: EvaluationScore(
                    score=None,
                    feedback="Reasoning efficiency not applicable",
                    raw_response="Reasoning efficiency not applicable",
                ),
                MetricCategory.PARAMETER_EXTRACTION: EvaluationScore(
                    score=7,
                    feedback="Test parameter extraction explanation",
                    raw_response="Test raw output",
                ),
                MetricCategory.TOOL_SELECTION: EvaluationScore(
                    score=8,
                    feedback="Test tool selection explanation",
                    raw_response="Test raw output",
                ),
            },
        )

        return {"Test Agent": agent_evaluation}

    @patch("crewai.experimental.evaluation.experiment.runner.create_default_evaluator")
    def test_run_success(
        self, mock_create_evaluator, mock_crew, mock_evaluator_results
    ):
        dataset = [
            {
                "identifier": "test-case-1",
                "inputs": {"query": "Test query 1"},
                "expected_score": 8,
            },
            {
                "identifier": "test-case-2",
                "inputs": {"query": "Test query 2"},
                "expected_score": {"goal_alignment": 7},
            },
            {
                "inputs": {"query": "Test query 3"},
                "expected_score": {"tool_selection": 9},
            },
        ]

        mock_evaluator = MagicMock()
        mock_evaluator.get_agent_evaluation.return_value = mock_evaluator_results
        mock_evaluator.reset_iterations_results = MagicMock()
        mock_create_evaluator.return_value = mock_evaluator

        runner = ExperimentRunner(dataset=dataset)

        results = runner.run(crew=mock_crew)

        assert isinstance(results, ExperimentResults)
        result_1, result_2, result_3 = results.results
        assert len(results.results) == 3

        assert result_1.identifier == "test-case-1"
        assert result_1.inputs == {"query": "Test query 1"}
        assert result_1.expected_score == 8
        assert result_1.passed is True

        assert result_2.identifier == "test-case-2"
        assert result_2.inputs == {"query": "Test query 2"}
        assert isinstance(result_2.expected_score, dict)
        assert "goal_alignment" in result_2.expected_score
        assert result_2.passed is True

        assert result_3.identifier == "c2ed49e63aa9a83af3ca382794134fd5"
        assert result_3.inputs == {"query": "Test query 3"}
        assert isinstance(result_3.expected_score, dict)
        assert "tool_selection" in result_3.expected_score
        assert result_3.passed is False

        assert mock_crew.kickoff.call_count == 3
        mock_crew.kickoff.assert_any_call(inputs={"query": "Test query 1"})
        mock_crew.kickoff.assert_any_call(inputs={"query": "Test query 2"})
        mock_crew.kickoff.assert_any_call(inputs={"query": "Test query 3"})

        assert mock_evaluator.reset_iterations_results.call_count == 3
        assert mock_evaluator.get_agent_evaluation.call_count == 3

    @patch("crewai.experimental.evaluation.experiment.runner.create_default_evaluator")
    def test_run_success_with_unknown_metric(
        self, mock_create_evaluator, mock_crew, mock_evaluator_results
    ):
        dataset = [
            {
                "identifier": "test-case-2",
                "inputs": {"query": "Test query 2"},
                "expected_score": {"goal_alignment": 7, "unknown_metric": 8},
            }
        ]

        mock_evaluator = MagicMock()
        mock_evaluator.get_agent_evaluation.return_value = mock_evaluator_results
        mock_evaluator.reset_iterations_results = MagicMock()
        mock_create_evaluator.return_value = mock_evaluator

        runner = ExperimentRunner(dataset=dataset)

        results = runner.run(crew=mock_crew)

        (result,) = results.results

        assert result.identifier == "test-case-2"
        assert result.inputs == {"query": "Test query 2"}
        assert isinstance(result.expected_score, dict)
        assert "goal_alignment" in result.expected_score.keys()
        assert "unknown_metric" in result.expected_score.keys()
        assert result.passed is True

    @patch("crewai.experimental.evaluation.experiment.runner.create_default_evaluator")
    def test_run_success_with_single_metric_evaluator_and_expected_specific_metric(
        self, mock_create_evaluator, mock_crew, mock_evaluator_results
    ):
        dataset = [
            {
                "identifier": "test-case-2",
                "inputs": {"query": "Test query 2"},
                "expected_score": {"goal_alignment": 7},
            }
        ]

        mock_evaluator = MagicMock()
        mock_create_evaluator["Test Agent"].metrics = {
            MetricCategory.GOAL_ALIGNMENT: EvaluationScore(
                score=9,
                feedback="Test feedback for goal alignment",
                raw_response="Test raw response for goal alignment",
            )
        }
        mock_evaluator.get_agent_evaluation.return_value = mock_evaluator_results
        mock_evaluator.reset_iterations_results = MagicMock()
        mock_create_evaluator.return_value = mock_evaluator

        runner = ExperimentRunner(dataset=dataset)

        results = runner.run(crew=mock_crew)
        (result,) = results.results

        assert result.identifier == "test-case-2"
        assert result.inputs == {"query": "Test query 2"}
        assert isinstance(result.expected_score, dict)
        assert "goal_alignment" in result.expected_score.keys()
        assert result.passed is True

    @patch("crewai.experimental.evaluation.experiment.runner.create_default_evaluator")
    def test_run_success_when_expected_metric_is_not_available(
        self, mock_create_evaluator, mock_crew, mock_evaluator_results
    ):
        dataset = [
            {
                "identifier": "test-case-2",
                "inputs": {"query": "Test query 2"},
                "expected_score": {"unknown_metric": 7},
            }
        ]

        mock_evaluator = MagicMock()
        mock_create_evaluator["Test Agent"].metrics = {
            MetricCategory.GOAL_ALIGNMENT: EvaluationScore(
                score=5,
                feedback="Test feedback for goal alignment",
                raw_response="Test raw response for goal alignment",
            )
        }
        mock_evaluator.get_agent_evaluation.return_value = mock_evaluator_results
        mock_evaluator.reset_iterations_results = MagicMock()
        mock_create_evaluator.return_value = mock_evaluator

        runner = ExperimentRunner(dataset=dataset)

        results = runner.run(crew=mock_crew)
        (result,) = results.results

        assert result.identifier == "test-case-2"
        assert result.inputs == {"query": "Test query 2"}
        assert isinstance(result.expected_score, dict)
        assert "unknown_metric" in result.expected_score.keys()
        assert result.passed is False
