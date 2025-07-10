from crewai.evaluation.base_evaluator import AgentEvaluationResult, AggregationStrategy
from crewai.agent import Agent
from crewai.task import Task
from crewai.evaluation.evaluation_display import EvaluationDisplayFormatter

from typing import Any, Dict
from collections import defaultdict
from crewai.evaluation import BaseEvaluator, create_evaluation_callbacks
from collections.abc import Sequence
from crewai.crew import Crew
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from crewai.evaluation.evaluation_display import AgentAggregatedEvaluationResult

class AgentEvaluator:
    def __init__(
        self,
        evaluators: Sequence[BaseEvaluator] | None = None,
        crew: Crew | None = None,
    ):
        self.crew: Crew | None = crew
        self.evaluators: Sequence[BaseEvaluator] | None = evaluators

        self.agent_evaluators: dict[str, Sequence[BaseEvaluator] | None] = {}
        if crew is not None:
            assert crew and crew.agents is not None
            for agent in crew.agents:
                self.agent_evaluators[str(agent.id)] = self.evaluators

        self.callback = create_evaluation_callbacks()
        self.console_formatter = ConsoleFormatter()
        self.display_formatter = EvaluationDisplayFormatter()

        self.iteration = 1
        self.iterations_results: dict[int, dict[str, list[AgentEvaluationResult]]] = {}

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def reset_iterations_results(self):
        self.iterations_results = {}

    def evaluate_current_iteration(self) -> dict[str, list[AgentEvaluationResult]]:
        if not self.crew:
            raise ValueError("Cannot evaluate: no crew was provided to the evaluator.")

        if not self.callback:
            raise ValueError("Cannot evaluate: no callback was set. Use set_callback() method first.")

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        evaluation_results: defaultdict[str, list[AgentEvaluationResult]] = defaultdict(list)

        total_evals = 0
        for agent in self.crew.agents:
            for task in self.crew.tasks:
                if task.agent and task.agent.id == agent.id and self.agent_evaluators.get(str(agent.id)):
                    total_evals += 1

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("{task.percentage:.0f}% completed"),
            console=self.console_formatter.console
        ) as progress:
            eval_task = progress.add_task(f"Evaluating agents (iteration {self.iteration})...", total=total_evals)

            for agent in self.crew.agents:
                evaluator = self.agent_evaluators.get(str(agent.id))
                if not evaluator:
                    continue

                for task in self.crew.tasks:

                    if task.agent and str(task.agent.id) != str(agent.id):
                        continue

                    trace = self.callback.get_trace(str(agent.id), str(task.id))
                    if not trace:
                        self.console_formatter.print(f"[yellow]Warning: No trace found for agent {agent.role} on task {task.description[:30]}...[/yellow]")
                        progress.update(eval_task, advance=1)
                        continue

                    with crewai_event_bus.scoped_handlers():
                        result = self.evaluate(
                            agent=agent,
                            task=task,
                            execution_trace=trace,
                            final_output=task.output
                        )
                        evaluation_results[agent.role].append(result)
                        progress.update(eval_task, advance=1)

        self.iterations_results[self.iteration] = evaluation_results
        return evaluation_results

    def get_evaluation_results(self):
        if self.iteration in self.iterations_results:
            return self.iterations_results[self.iteration]

        return self.evaluate_current_iteration()

    def display_results_with_iterations(self):
        self.display_formatter.display_summary_results(self.iterations_results)

    def get_agent_evaluation(self, strategy: AggregationStrategy = AggregationStrategy.SIMPLE_AVERAGE, include_evaluation_feedback: bool = False) -> Dict[str, AgentAggregatedEvaluationResult]:
        agent_results = {}
        with crewai_event_bus.scoped_handlers():
            task_results = self.get_evaluation_results()
            for agent_role, results in task_results.items():
                if not results:
                    continue

                agent_id = results[0].agent_id

                aggregated_result = self.display_formatter._aggregate_agent_results(
                    agent_id=agent_id,
                    agent_role=agent_role,
                    results=results,
                    strategy=strategy
                )

                agent_results[agent_role] = aggregated_result


            if self.iteration == max(self.iterations_results.keys()):
                self.display_results_with_iterations()

            if include_evaluation_feedback:
                self.display_evaluation_with_feedback()

        return agent_results

    def display_evaluation_with_feedback(self):
        self.display_formatter.display_evaluation_with_feedback(self.iterations_results)

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any
    ) -> AgentEvaluationResult:
        result = AgentEvaluationResult(
            agent_id=str(agent.id),
            task_id=str(task.id)
        )
        assert self.evaluators is not None
        for evaluator in self.evaluators:
            try:
                score = evaluator.evaluate(
                    agent=agent,
                    task=task,
                    execution_trace=execution_trace,
                    final_output=final_output
                )
                result.metrics[evaluator.metric_category] = score
            except Exception as e:
                self.console_formatter.print(f"Error in {evaluator.metric_category.value} evaluator: {str(e)}")

        return result

def create_default_evaluator(crew, llm=None):
    from crewai.evaluation import (
        GoalAlignmentEvaluator,
        SemanticQualityEvaluator,
        ToolSelectionEvaluator,
        ParameterExtractionEvaluator,
        ToolInvocationEvaluator,
        ReasoningEfficiencyEvaluator
    )

    evaluators = [
        GoalAlignmentEvaluator(llm=llm),
        SemanticQualityEvaluator(llm=llm),
        ToolSelectionEvaluator(llm=llm),
        ParameterExtractionEvaluator(llm=llm),
        ToolInvocationEvaluator(llm=llm),
        ReasoningEfficiencyEvaluator(llm=llm),
    ]

    return AgentEvaluator(evaluators=evaluators, crew=crew)
