from datetime import datetime, timezone
import json
import os
from typing import Any

from pydantic import BaseModel


class ExperimentResult(BaseModel):
    identifier: str
    inputs: dict[str, Any]
    score: float | dict[str, float]
    expected_score: float | dict[str, float]
    passed: bool
    agent_evaluations: dict[str, Any] | None = None


class ExperimentResults:
    def __init__(
        self, results: list[ExperimentResult], metadata: dict[str, Any] | None = None
    ):
        self.results = results
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)

        from crewai.experimental.evaluation.experiment.result_display import (
            ExperimentResultsDisplay,
        )

        self.display = ExperimentResultsDisplay()

    def to_json(self, filepath: str | None = None) -> dict[str, Any]:
        data = {
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "results": [
                r.model_dump(exclude={"agent_evaluations"}) for r in self.results
            ],
        }

        if filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            self.display.console.print(f"[green]Results saved to {filepath}[/green]")

        return data

    def compare_with_baseline(
        self,
        baseline_filepath: str,
        save_current: bool = True,
        print_summary: bool = False,
    ) -> dict[str, Any]:
        baseline_runs = []

        if os.path.exists(baseline_filepath) and os.path.getsize(baseline_filepath) > 0:
            try:
                with open(baseline_filepath, "r") as f:
                    baseline_data = json.load(f)

                if isinstance(baseline_data, dict) and "timestamp" in baseline_data:
                    baseline_runs = [baseline_data]
                elif isinstance(baseline_data, list):
                    baseline_runs = baseline_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.display.console.print(
                    f"[yellow]Warning: Could not load baseline file: {e!s}[/yellow]"
                )

        if not baseline_runs:
            if save_current:
                current_data = self.to_json()
                with open(baseline_filepath, "w") as f:
                    json.dump([current_data], f, indent=2)
                self.display.console.print(
                    f"[green]Saved current results as new baseline to {baseline_filepath}[/green]"
                )
            return {"is_baseline": True, "changes": {}}

        baseline_runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        latest_run = baseline_runs[0]

        comparison = self._compare_with_run(latest_run)

        if print_summary:
            self.display.comparison_summary(comparison, latest_run["timestamp"])

        if save_current:
            current_data = self.to_json()
            baseline_runs.append(current_data)
            with open(baseline_filepath, "w") as f:
                json.dump(baseline_runs, f, indent=2)
            self.display.console.print(
                f"[green]Added current results to baseline file {baseline_filepath}[/green]"
            )

        return comparison

    def _compare_with_run(self, baseline_run: dict[str, Any]) -> dict[str, Any]:
        baseline_results = baseline_run.get("results", [])

        baseline_lookup = {}
        for result in baseline_results:
            test_identifier = result.get("identifier")
            if test_identifier:
                baseline_lookup[test_identifier] = result

        improved = []
        regressed = []
        unchanged = []
        new_tests = []

        for result in self.results:
            test_identifier = result.identifier
            if not test_identifier or test_identifier not in baseline_lookup:
                new_tests.append(test_identifier)
                continue

            baseline_result = baseline_lookup[test_identifier]
            baseline_passed = baseline_result.get("passed", False)
            if result.passed and not baseline_passed:
                improved.append(test_identifier)
            elif not result.passed and baseline_passed:
                regressed.append(test_identifier)
            else:
                unchanged.append(test_identifier)

        missing_tests = []
        current_test_identifiers = {result.identifier for result in self.results}
        for result in baseline_results:
            test_identifier = result.get("identifier")
            if test_identifier and test_identifier not in current_test_identifiers:
                missing_tests.append(test_identifier)

        return {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "new_tests": new_tests,
            "missing_tests": missing_tests,
            "total_compared": len(improved) + len(regressed) + len(unchanged),
            "baseline_timestamp": baseline_run.get("timestamp", "unknown"),
        }
