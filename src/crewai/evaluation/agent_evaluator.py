from crewai.evaluation.base_evaluator import AgentEvaluationResult, AgentAggregatedEvaluationResult, AggregationStrategy
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.agent import Agent
from crewai.task import Task
from crewai.utilities.llm_utils import create_llm
from crewai.evaluation.evaluation_display import EvaluationDisplayFormatter

from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from crewai.evaluation import EvaluationScore, BaseEvaluator, create_evaluation_callbacks
from crewai.crew import Crew
from rich.table import Table
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter

class AgentEvaluator:
    def __init__(
        self,
        evaluators: Optional[List[BaseEvaluator]] = None,
        crew: Optional[Any] = None,
    ):
        self.crew: Crew = crew
        self.evaluators = evaluators

        self.agent_evaluators = {}
        if crew is not None:
            for agent in crew.agents:
                self.agent_evaluators[agent.id] = self.evaluators.copy()

        self.callback = create_evaluation_callbacks()
        self.console_formatter = ConsoleFormatter()
        self.display_formatter = EvaluationDisplayFormatter()

        self.iteration = 1
        self.iterations_results = {}

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def evaluate_current_iteration(self):
        if not self.crew:
            raise ValueError("Cannot evaluate: no crew was provided to the evaluator.")

        if not self.callback:
            raise ValueError("Cannot evaluate: no callback was set. Use set_callback() method first.")

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        self.console_formatter.print(f"\n[bold blue]📊 Running agent evaluations for iteration {self.iteration}...[/bold blue]\n")

        evaluation_results = defaultdict(list)

        total_evals = 0
        for agent in self.crew.agents:
            for task in self.crew.tasks:
                if task.agent.id == agent.id and self.agent_evaluators.get(agent.id):
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
                evaluator = self.agent_evaluators.get(agent.id)
                if not evaluator:
                    continue

                for task in self.crew.tasks:
                    if task.agent.id != agent.id:
                        continue

                    trace = self.callback.get_trace(agent.id, task.id)
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

    def get_agent_evaluation(self, strategy: AggregationStrategy = AggregationStrategy.SIMPLE_AVERAGE):
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

            if len(self.iterations_results) > 1 and self.iteration == max(self.iterations_results.keys()):
                self.display_results_with_iterations()
            elif agent_results:
                self.display_evaluation_results(agent_results)

        return agent_results

    def display_evaluation_results(self, agent_results: Dict[str, AgentAggregatedEvaluationResult]):
        self.display_formatter.display_evaluation_results(agent_results)

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
