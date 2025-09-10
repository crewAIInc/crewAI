import inspect
from pathlib import Path

from typing_extensions import Any
import warnings
from crewai.experimental.evaluation.experiment import ExperimentResults, ExperimentRunner
from crewai import Crew, Agent

def assert_experiment_successfully(experiment_results: ExperimentResults, baseline_filepath: str | None = None) -> None:
    failed_tests = [result for result in experiment_results.results if not result.passed]

    if failed_tests:
        detailed_failures: list[str] = []

        for result in failed_tests:
            expected = result.expected_score
            actual = result.score
            detailed_failures.append(f"- {result.identifier}: expected {expected}, got {actual}")

        failure_details = "\n".join(detailed_failures)
        raise AssertionError(f"The following test cases failed:\n{failure_details}")

    baseline_filepath = baseline_filepath or _get_baseline_filepath_fallback()
    comparison = experiment_results.compare_with_baseline(baseline_filepath=baseline_filepath)
    assert_experiment_no_regression(comparison)

def assert_experiment_no_regression(comparison_result: dict[str, list[str]]) -> None:
    regressed = comparison_result.get("regressed", [])
    if regressed:
        raise AssertionError(f"Regression detected! The following tests that previously passed now fail: {regressed}")

    missing_tests = comparison_result.get("missing_tests", [])
    if missing_tests:
        warnings.warn(
            f"Warning: {len(missing_tests)} tests from the baseline are missing in the current run: {missing_tests}",
            UserWarning
        )

def run_experiment(dataset: list[dict[str, Any]], crew: Crew | None = None, agents: list[Agent] | None = None, verbose: bool = False) -> ExperimentResults:
    runner = ExperimentRunner(dataset=dataset)

    return runner.run(agents=agents, crew=crew, print_summary=verbose)

def _get_baseline_filepath_fallback() -> str:
    filename = "experiment_fallback.json"
    calling_file = None

    try:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            test_func_name = current_frame.f_back.f_back.f_code.co_name # type: ignore[union-attr]
            filename = f"{test_func_name}.json"
            calling_file = current_frame.f_back.f_back.f_code.co_filename # type: ignore[union-attr]
    except Exception:
        return filename

    if not calling_file:
        return filename

    calling_path = Path(calling_file)
    try:
        baseline_dir_parts = calling_path.parts[:-1]
        baseline_dir = Path(*baseline_dir_parts) / "results"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_filepath = baseline_dir / filename
        return str(baseline_filepath)

    except (ValueError, IndexError):
        pass

    return filename
