import inspect
import warnings

from typing_extensions import Any

from crewai import Agent, Crew
from crewai.experimental.evaluation.experiment import (
    ExperimentResults,
    ExperimentRunner,
)


def assert_experiment_successfully(
    experiment_results: ExperimentResults, baseline_filepath: str | None = None
) -> None:
    failed_tests = [
        result for result in experiment_results.results if not result.passed
    ]

    if failed_tests:
        detailed_failures: list[str] = []

        for result in failed_tests:
            expected = result.expected_score
            actual = result.score
            detailed_failures.append(
                f"- {result.identifier}: expected {expected}, got {actual}"
            )

        failure_details = "\n".join(detailed_failures)
        raise AssertionError(f"The following test cases failed:\n{failure_details}")

    baseline_filepath = baseline_filepath or _get_baseline_filepath_fallback()
    comparison = experiment_results.compare_with_baseline(
        baseline_filepath=baseline_filepath
    )
    assert_experiment_no_regression(comparison)


def assert_experiment_no_regression(comparison_result: dict[str, list[str]]) -> None:
    regressed = comparison_result.get("regressed", [])
    if regressed:
        raise AssertionError(
            f"Regression detected! The following tests that previously passed now fail: {regressed}"
        )

    missing_tests = comparison_result.get("missing_tests", [])
    if missing_tests:
        warnings.warn(
            f"Warning: {len(missing_tests)} tests from the baseline are missing in the current run: {missing_tests}",
            UserWarning,
            stacklevel=2,
        )


def run_experiment(
    dataset: list[dict[str, Any]],
    crew: Crew | None = None,
    agents: list[Agent] | None = None,
    verbose: bool = False,
) -> ExperimentResults:
    runner = ExperimentRunner(dataset=dataset)

    return runner.run(agents=agents, crew=crew, print_summary=verbose)


def _get_baseline_filepath_fallback() -> str:
    test_func_name = "experiment_fallback"

    try:
        current_frame = inspect.currentframe()
        if current_frame is not None:
            test_func_name = current_frame.f_back.f_back.f_code.co_name  # type: ignore[union-attr]
    except Exception:
        ...
    return f"{test_func_name}_results.json"
