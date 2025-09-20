from collections import defaultdict
from hashlib import md5
from typing import Any

from crewai import Agent, Crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.experimental.evaluation import AgentEvaluator, create_default_evaluator
from crewai.experimental.evaluation.evaluation_display import (
    AgentAggregatedEvaluationResult,
)
from crewai.experimental.evaluation.experiment.result import (
    ExperimentResult,
    ExperimentResults,
)
from crewai.experimental.evaluation.experiment.result_display import (
    ExperimentResultsDisplay,
)


class ExperimentRunner:
    def __init__(self, dataset: list[dict[str, Any]]):
        self.dataset = dataset or []
        self.evaluator: AgentEvaluator | None = None
        self.display = ExperimentResultsDisplay()

    def run(
        self,
        crew: Crew | None = None,
        agents: list[Agent] | list[BaseAgent] | None = None,
        print_summary: bool = False,
    ) -> ExperimentResults:
        if crew and not agents:
            agents = crew.agents

        if agents is None:
            raise ValueError("Agents must be provided either directly or via a crew")
        self.evaluator = create_default_evaluator(agents=agents)

        results = []

        for test_case in self.dataset:
            self.evaluator.reset_iterations_results()
            result = self._run_test_case(test_case=test_case, crew=crew, agents=agents)
            results.append(result)

        experiment_results = ExperimentResults(results)

        if print_summary:
            self.display.summary(experiment_results)

        return experiment_results

    def _run_test_case(
        self,
        test_case: dict[str, Any],
        agents: list[Agent] | list[BaseAgent],
        crew: Crew | None = None,
    ) -> ExperimentResult:
        inputs = test_case["inputs"]
        expected_score = test_case["expected_score"]
        identifier = (
            test_case.get("identifier")
            or md5(str(test_case).encode(), usedforsecurity=False).hexdigest()
        )

        try:
            self.display.console.print(
                f"[dim]Running crew with input: {str(inputs)[:50]}...[/dim]"
            )
            self.display.console.print("\n")
            if crew:
                crew.kickoff(inputs=inputs)
            else:
                for agent in agents:
                    if isinstance(agent, Agent):
                        agent.kickoff(**inputs)
                    else:
                        raise TypeError(
                            f"Agent {agent} is not an instance of Agent and cannot be kicked off directly"
                        )

            if self.evaluator is None:
                raise ValueError("Evaluator must be initialized")
            agent_evaluations = self.evaluator.get_agent_evaluation()

            actual_score = self._extract_scores(agent_evaluations)

            passed = self._assert_scores(expected_score, actual_score)
            return ExperimentResult(
                identifier=identifier,
                inputs=inputs,
                score=actual_score,
                expected_score=expected_score,
                passed=passed,
                agent_evaluations=agent_evaluations,
            )

        except Exception as e:
            self.display.console.print(f"[red]Error running test case: {e!s}[/red]")
            return ExperimentResult(
                identifier=identifier,
                inputs=inputs,
                score=0.0,
                expected_score=expected_score,
                passed=False,
            )

    def _extract_scores(
        self, agent_evaluations: dict[str, AgentAggregatedEvaluationResult]
    ) -> float | dict[str, float]:
        all_scores: dict[str, list[float]] = defaultdict(list)
        for evaluation in agent_evaluations.values():
            for metric_name, score in evaluation.metrics.items():
                if score.score is not None:
                    all_scores[metric_name.value].append(score.score)

        avg_scores = {m: sum(s) / len(s) for m, s in all_scores.items()}

        if len(avg_scores) == 1:
            return next(iter(avg_scores.values()))

        return avg_scores

    def _assert_scores(
        self, expected: float | dict[str, float], actual: float | dict[str, float]
    ) -> bool:
        """
        Compare expected and actual scores, and return whether the test case passed.

        The rules for comparison are as follows:
        - If both expected and actual scores are single numbers, the actual score must be >= expected.
        - If expected is a single number and actual is a dict, compare against the average of actual values.
        - If expected is a dict and actual is a single number, actual must be >= all expected values.
        - If both are dicts, actual must have matching keys with values >= expected values.
        """

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return actual >= expected

        if isinstance(expected, dict) and isinstance(actual, (int, float)):
            return all(actual >= exp_score for exp_score in expected.values())

        if isinstance(expected, (int, float)) and isinstance(actual, dict):
            if not actual:
                return False
            avg_score = sum(actual.values()) / len(actual)
            return avg_score >= expected

        if isinstance(expected, dict) and isinstance(actual, dict):
            if not expected:
                return True
            matching_keys = set(expected.keys()) & set(actual.keys())
            if not matching_keys:
                return False

            # All matching keys must have actual >= expected
            return all(actual[key] >= expected[key] for key in matching_keys)

        return False
