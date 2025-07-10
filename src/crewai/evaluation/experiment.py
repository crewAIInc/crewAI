"""Agent evaluation experiments for crewAI.

This module provides tools for running experiments to evaluate agent performance
across a dataset of test cases, as well as persistence and visualization of results.
"""

from collections import defaultdict
from hashlib import md5
import json
import os
from datetime import datetime
from typing import List, Dict, Union, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from crewai import Crew
from crewai.evaluation import AgentEvaluator, create_default_evaluator
from pydantic import BaseModel
from crewai.evaluation.evaluation_display import AgentAggregatedEvaluationResult

class TestCaseResult(BaseModel):
    identifier: str
    inputs: Dict[str, Any]
    score: Union[int, Dict[str, Union[int, float]]]
    expected_score: Union[int, Dict[str, Union[int, float]]]
    passed: bool
    agent_evaluations: Optional[Dict[str, Any]] = None

class ExperimentResults:
    def __init__(self, results: List[TestCaseResult], metadata: Optional[Dict[str, Any]] = None):
        self.results = results
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.console = Console()

    def to_json(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "results": [r.model_dump(exclude={"agent_evaluations"}) for r in self.results]
        }

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.console.print(f"[green]Results saved to {filepath}[/green]")

        return data

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        table = Table(title="Experiment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Test Cases", str(total))
        table.add_row("Passed", str(passed))
        table.add_row("Failed", str(total - passed))
        table.add_row("Success Rate", f"{(passed / total * 100):.1f}%" if total > 0 else "N/A")

        self.console.print(table)

    def compare_with_baseline(self, baseline_filepath: str, save_current: bool = True) -> Dict[str, Any]:
        baseline_runs = []

        if os.path.exists(baseline_filepath) and os.path.getsize(baseline_filepath) > 0:
            try:
                with open(baseline_filepath, 'r') as f:
                    baseline_data = json.load(f)

                if isinstance(baseline_data, dict) and "timestamp" in baseline_data:
                    baseline_runs = [baseline_data]
                elif isinstance(baseline_data, list):
                    baseline_runs = baseline_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.console.print(f"[yellow]Warning: Could not load baseline file: {str(e)}[/yellow]")

        if not baseline_runs:
            if save_current:
                current_data = self.to_json()
                with open(baseline_filepath, 'w') as f:
                    json.dump([current_data], f, indent=2)
                self.console.print(f"[green]Saved current results as new baseline to {baseline_filepath}[/green]")
            return {"is_baseline": True, "changes": {}}

        baseline_runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        latest_run = baseline_runs[0]

        comparison = self._compare_with_run(latest_run)

        self._print_comparison_summary(comparison, latest_run["timestamp"])

        if save_current:
            current_data = self.to_json()
            baseline_runs.append(current_data)
            with open(baseline_filepath, 'w') as f:
                json.dump(baseline_runs, f, indent=2)
            self.console.print(f"[green]Added current results to baseline file {baseline_filepath}[/green]")

        return comparison

    def _compare_with_run(self, baseline_run: Dict[str, Any]) -> Dict[str, Any]:
        baseline_results = baseline_run.get("results", [])

        baseline_lookup = {}
        for result in baseline_results:
            test_id = result.get("test_id")
            if test_id:
                baseline_lookup[test_id] = result

        improved = []
        regressed = []
        unchanged = []
        new_tests = []

        for result in self.results:
            test_id = result.identifier
            if not test_id or test_id not in baseline_lookup:
                new_tests.append(test_id)
                continue

            baseline_result = baseline_lookup[test_id]
            baseline_passed = baseline_result.get("passed", False)

            if result.passed and not baseline_passed:
                improved.append((test_id, result.score, baseline_result.get("score", 0)))
            elif not result.passed and baseline_passed:
                regressed.append((test_id, result.score, baseline_result.get("score", 0)))
            else:
                unchanged.append(test_id)

        missing_tests = []
        current_test_ids = {result.identifier for result in self.results}
        for result in baseline_results:
            test_id = result.get("identifier")
            if test_id and test_id not in current_test_ids:
                missing_tests.append(test_id)

        return {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "new_tests": new_tests,
            "missing_tests": missing_tests,
            "total_compared": len(improved) + len(regressed) + len(unchanged),
            "baseline_timestamp": baseline_run.get("timestamp", "unknown")
        }

    def _get_inputs_from_test_case(self, test_case: Dict[str, Any]) -> str:
        inputs = test_case.get("inputs")
        if inputs is None:
            return ""

        if not isinstance(inputs, str):
            return str(inputs)
        return inputs

    def _print_comparison_summary(self, comparison: Dict[str, Any], baseline_timestamp: str):
        self.console.print(Panel(f"[bold]Comparison with baseline run from {baseline_timestamp}[/bold]",
                                 expand=False))

        table = Table(title="Results Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")
        table.add_column("Details", style="dim")

        improved = comparison.get("improved", [])
        if improved:
            details = ", ".join([f"{test_id}" for test_id, _, _ in improved[:3]])
            if len(improved) > 3:
                details += f" and {len(improved) - 3} more"
            table.add_row("✅ Improved", str(len(improved)), details)
        else:
            table.add_row("✅ Improved", "0", "")

        regressed = comparison.get("regressed", [])
        if regressed:
            details = ", ".join([f"{test_id}" for test_id, _, _ in regressed[:3]])
            if len(regressed) > 3:
                details += f" and {len(regressed) - 3} more"
            table.add_row("❌ Regressed", str(len(regressed)), details, style="red")
        else:
            table.add_row("❌ Regressed", "0", "")

        unchanged = comparison.get("unchanged", [])
        table.add_row("⏺ Unchanged", str(len(unchanged)), "")

        new_tests = comparison.get("new_tests", [])
        if new_tests:
            details = ", ".join(new_tests[:3])
            if len(new_tests) > 3:
                details += f" and {len(new_tests) - 3} more"
            table.add_row("➕ New Tests", str(len(new_tests)), details)

        missing_tests = comparison.get("missing_tests", [])
        if missing_tests:
            details = ", ".join(missing_tests[:3])
            if len(missing_tests) > 3:
                details += f" and {len(missing_tests) - 3} more"
            table.add_row("➖ Missing Tests", str(len(missing_tests)), details)

        self.console.print(table)


class ExperimentRunner:
    def __init__(self, dataset: Optional[List[Dict[str, Any]]] = None, evaluator: AgentEvaluator | None = None):
        self.dataset = dataset or []
        self.evaluator = evaluator
        self.console = Console()

    def run(self, crew: Optional[Crew] = None) -> ExperimentResults:
        if not self.dataset:
            raise ValueError("No dataset provided. Use load_dataset() or provide dataset in constructor.")

        if not crew:
            raise ValueError("crew must be provided.")

        if not self.evaluator:
            self.evaluator = create_default_evaluator(crew=crew)

        results = []

        for test_case in self.dataset:
            self.evaluator.reset_iterations_results()
            result = self._run_test_case(test_case, crew)
            results.append(result)


        return ExperimentResults(results)

    def _run_test_case(self, test_case: Dict[str, Any], crew: Crew) -> TestCaseResult:
        inputs = test_case["inputs"]
        expected_score = test_case["expected_score"]
        identifier = test_case.get("identifier") or md5(str(test_case)).hexdigest()

        try:
            self.console.print(f"[dim]Running crew with input: {str(inputs)[:50]}...[/dim]")
            crew.kickoff(inputs=inputs)

            agent_evaluations = self.evaluator.get_agent_evaluation()

            actual_score = self._extract_scores(agent_evaluations)

            passed = self._compare_scores(expected_score, actual_score)
            return TestCaseResult(
                identifier=identifier,
                inputs=inputs,
                score=actual_score,
                expected_score=expected_score,
                passed=passed,
                agent_evaluations=agent_evaluations
            )

        except Exception as e:
            self.console.print(f"[red]Error running test case: {str(e)}[/red]")
            return TestCaseResult(
                identifier=identifier,
                inputs=inputs,
                score=0,
                expected_score=expected_score,
                passed=False
            )

    def _extract_scores(self, agent_evaluations: Dict[str, AgentAggregatedEvaluationResult]) -> Union[int, Dict[str, int]]:
        all_scores = defaultdict(list)
        for evaluation in agent_evaluations.values():
            for metric_name, score in evaluation.metrics.items():
                if score.score is not None:
                    all_scores[metric_name.value].append(score.score)

        avg_scores = {m: sum(s)/len(s) for m, s in all_scores.items()}

        if len(avg_scores) == 1:
            return list(avg_scores.values())[0]

        return avg_scores

    def _compare_scores(self, expected: Union[int, Dict[str, int]],
                        actual: Union[int, Dict[str, int]]) -> bool:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return actual >= expected

        elif isinstance(expected, dict) and isinstance(actual, (int, float)):
            return False

        elif isinstance(expected, (int, float)) and isinstance(actual, dict):
            avg_score = sum(actual.values()) / len(actual)
            return avg_score >= expected

        elif isinstance(expected, dict) and isinstance(actual, dict):
            for metric, exp_score in expected.items():
                if metric not in actual or actual[metric] < exp_score:
                    return False
            return True

        return False