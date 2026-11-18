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
class TestBaselineEncoding:
    """A baseline is a file people keep and edit, so it is read as UTF-8."""

    def test_compare_with_baseline_matches_a_utf8_file_under_a_non_utf8_default(self, tmp_path):
        """The same test identifier must match, whatever the platform default is.

        `open()` without an encoding uses ``locale.getpreferredencoding(False)``:
        cp1252 or cp936 on Windows. A UTF-8 baseline read that way yields mangled
        strings, so a result whose identifier contains non-ASCII text is reported
        as both new and missing instead of unchanged.
        """
        import json
        import os
        import subprocess
        import sys
        import textwrap

        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps(
                [
                    {
                        "timestamp": "2026-01-01T00:00:00",
                        "metadata": {"agent": "café 🚀"},
                        "results": [
                            {
                                "identifier": "café-🚀",
                                "inputs": {"q": "x"},
                                "score": 1,
                                "expected_score": 1,
                                "passed": True,
                            }
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        script = textwrap.dedent(
            f"""
            import json
            from crewai.experimental.evaluation.experiment.result import ExperimentResult, ExperimentResults

            current = ExperimentResults(
                results=[
                    ExperimentResult(
                        identifier="café-🚀",
                        inputs={{"q": "x"}},
                        score=1,
                        expected_score=1,
                        passed=True,
                    )
                ],
                metadata={{}},
            )
            comparison = current.compare_with_baseline({str(baseline)!r}, save_current=False)
            print(json.dumps({{
                "unchanged": len(comparison["unchanged"]),
                "new_tests": len(comparison["new_tests"]),
                "missing_tests": len(comparison["missing_tests"]),
            }}))
            """
        )

        # The script is a file, not `-c`: argv under a C locale is decoded as
        # ASCII, so an inline script would arrive with the non-ASCII identifier
        # below mangled and the comparison would fail for the wrong reason.
        probe = tmp_path / "compare_probe.py"
        probe.write_text(script, encoding="utf-8")

        env = {key: value for key, value in os.environ.items() if key not in {"PYTHONUTF8", "PYTHONIOENCODING"}}
        env["LC_ALL"] = "C"
        env["LANG"] = "C"

        completed = subprocess.run(
            [sys.executable, "-X", "utf8=0", str(probe)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )

        assert completed.returncode == 0, completed.stderr
        outcome = json.loads(completed.stdout.strip().splitlines()[-1])
        assert outcome == {"unchanged": 1, "new_tests": 0, "missing_tests": 0}, completed.stdout
