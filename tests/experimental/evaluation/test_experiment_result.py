import pytest
from unittest.mock import MagicMock, patch

from crewai.experimental.evaluation.experiment.result import ExperimentResult, ExperimentResults


class TestExperimentResult:
    @pytest.fixture
    def mock_results(self):
        return [
            ExperimentResult(
                identifier="test-1",
                inputs={"query": "What is the capital of France?"},
                score=10,
                expected_score=7,
                passed=True
            ),
            ExperimentResult(
                identifier="test-2",
                inputs={"query": "Who wrote Hamlet?"},
                score={"relevance": 9, "factuality": 8},
                expected_score={"relevance": 7, "factuality": 7},
                passed=True,
                agent_evaluations={"agent1": {"metrics": {"goal_alignment": {"score": 9}}}}
            ),
            ExperimentResult(
                identifier="test-3",
                inputs={"query": "Any query"},
                score={"relevance": 9, "factuality": 8},
                expected_score={"relevance": 7, "factuality": 7},
                passed=False,
                agent_evaluations={"agent1": {"metrics": {"goal_alignment": {"score": 9}}}}
            ),
            ExperimentResult(
                identifier="test-4",
                inputs={"query": "Another query"},
                score={"relevance": 9, "factuality": 8},
                expected_score={"relevance": 7, "factuality": 7},
                passed=True,
                agent_evaluations={"agent1": {"metrics": {"goal_alignment": {"score": 9}}}}
            ),
            ExperimentResult(
                identifier="test-6",
                inputs={"query": "Yet another query"},
                score={"relevance": 9, "factuality": 8},
                expected_score={"relevance": 7, "factuality": 7},
                passed=True,
                agent_evaluations={"agent1": {"metrics": {"goal_alignment": {"score": 9}}}}
            )
        ]

    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=1)
    @patch('json.load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_experiment_results_compare_with_baseline(self, mock_open, mock_json_load, mock_path_getsize, mock_path_exists, mock_results):
        baseline_data = {
            "timestamp": "2023-01-01T00:00:00+00:00",
            "results": [
                {
                    "identifier": "test-1",
                    "inputs": {"query": "What is the capital of France?"},
                    "score": 7,
                    "expected_score": 7,
                    "passed": False
                },
                {
                    "identifier": "test-2",
                    "inputs": {"query": "Who wrote Hamlet?"},
                    "score": {"relevance": 8, "factuality": 7},
                    "expected_score": {"relevance": 7, "factuality": 7},
                    "passed": True
                },
                {
                    "identifier": "test-3",
                    "inputs": {"query": "Any query"},
                    "score": {"relevance": 8, "factuality": 7},
                    "expected_score": {"relevance": 7, "factuality": 7},
                    "passed": True
                },
                {
                    "identifier": "test-4",
                    "inputs": {"query": "Another query"},
                    "score": {"relevance": 8, "factuality": 7},
                    "expected_score": {"relevance": 7, "factuality": 7},
                    "passed": True
                },
                {
                    "identifier": "test-5",
                    "inputs": {"query": "Another query"},
                    "score": {"relevance": 8, "factuality": 7},
                    "expected_score": {"relevance": 7, "factuality": 7},
                    "passed": True
                }
            ]
        }

        mock_json_load.return_value = baseline_data

        results = ExperimentResults(results=mock_results)
        results.display = MagicMock()

        comparison = results.compare_with_baseline(baseline_filepath="baseline.json")

        assert "baseline_timestamp" in comparison
        assert comparison["baseline_timestamp"] == "2023-01-01T00:00:00+00:00"
        assert comparison["improved"] == ["test-1"]
        assert comparison["regressed"] == ["test-3"]
        assert comparison["unchanged"] == ["test-2", "test-4"]
        assert comparison["new_tests"] == ["test-6"]
        assert comparison["missing_tests"] == ["test-5"]