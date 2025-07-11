from collections import defaultdict
from hashlib import md5
from typing import List, Dict, Union, Optional, Any
from rich.console import Console

from crewai import Crew
from crewai.evaluation import AgentEvaluator, create_default_evaluator
from crewai.evaluation.experiment.result_display import ExperimentResultsDisplay
from crewai.evaluation.experiment.result import ExperimentResults, ExperimentResult
from crewai.evaluation.evaluation_display import AgentAggregatedEvaluationResult

class ExperimentRunner:
    def __init__(self, dataset: List[Dict[str, Any]]):
        self.dataset = dataset or []
        self.evaluator = None
        self.display = ExperimentResultsDisplay()

    def run(self, crew: Optional[Crew] = None, print_summary: bool = False) -> ExperimentResults:
        if not crew:
            raise ValueError("crew must be provided.")

        self.evaluator = create_default_evaluator(crew=crew)

        results = []

        for test_case in self.dataset:
            self.evaluator.reset_iterations_results()
            result = self._run_test_case(test_case, crew)
            results.append(result)

        experiment_results = ExperimentResults(results)

        if print_summary:
            self.display.summary(experiment_results)

        return experiment_results

    def _run_test_case(self, test_case: Dict[str, Any], crew: Crew) -> ExperimentResult:
        inputs = test_case["inputs"]
        expected_score = test_case["expected_score"]
        identifier = test_case.get("identifier") or md5(str(test_case), usedforsecurity=False).hexdigest()

        try:
            self.display.console.print(f"[dim]Running crew with input: {str(inputs)[:50]}...[/dim]")
            self.display.console.print("\n")
            crew.kickoff(inputs=inputs)

            agent_evaluations = self.evaluator.get_agent_evaluation()

            actual_score = self._extract_scores(agent_evaluations)

            passed = self._assert_scores(expected_score, actual_score)
            return ExperimentResult(
                identifier=identifier,
                inputs=inputs,
                score=actual_score,
                expected_score=expected_score,
                passed=passed,
                agent_evaluations=agent_evaluations
            )

        except Exception as e:
            self.display.console.print(f"[red]Error running test case: {str(e)}[/red]")
            return ExperimentResult(
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

    def _assert_scores(self, expected: Union[int, Dict[str, int]],
                        actual: Union[int, Dict[str, int]]) -> bool:
        """
        Compare expected and actual scores, and return whether the test case passed.

        The rules for comparison are as follows:

        - If both expected and actual scores are single numbers, the actual score must be greater than or equal to the expected score.
        - If the expected score is a single number and the actual score is a dict, the test case fails.
        - If the expected score is a dict and the actual score is a single number, the test case fails.
        - If both expected and actual scores are dicts, the actual score must have all the same keys as the expected score, and the value for each key must be greater than or equal to the expected score.
        """

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