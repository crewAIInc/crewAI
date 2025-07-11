import warnings
from crewai.experimental.evaluation import ExperimentResults

def assert_experiment_successfully(experiment_results: ExperimentResults) -> None:
    """
    Assert that all experiment results passed successfully.

    Args:
        experiment_results: The experiment results to check

    Raises:
        AssertionError: If any test case failed
    """
    failed_tests = [result for result in experiment_results.results if not result.passed]

    if failed_tests:
        detailed_failures: list[str] = []

        for result in failed_tests:
            expected = result.expected_score
            actual = result.score
            detailed_failures.append(f"- {result.identifier}: expected {expected}, got {actual}")

        failure_details = "\n".join(detailed_failures)
        raise AssertionError(f"The following test cases failed:\n{failure_details}")

def assert_experiment_no_regression(comparison_result: dict[str, list[str]]) -> None:
    """
    Assert that there are no regressions in the experiment results compared to baseline.
    Also warns if there are missing tests.

    Args:
        comparison_result: The result from compare_with_baseline()

    Raises:
        AssertionError: If there are regressions
    """
    # Check for regressions
    regressed = comparison_result.get("regressed", [])
    if regressed:
        raise AssertionError(f"Regression detected! The following tests that previously passed now fail: {regressed}")

    # Check for missing tests and warn
    missing_tests = comparison_result.get("missing_tests", [])
    if missing_tests:
        warnings.warn(
            f"Warning: {len(missing_tests)} tests from the baseline are missing in the current run: {missing_tests}",
            UserWarning
        )