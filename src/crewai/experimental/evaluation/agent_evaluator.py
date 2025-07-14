from crewai.experimental.evaluation.base_evaluator import AgentEvaluationResult, AggregationStrategy
from crewai.agent import Agent
from crewai.task import Task
from crewai.experimental.evaluation.evaluation_display import EvaluationDisplayFormatter

from typing import Any, Dict
from collections import defaultdict
from crewai.experimental.evaluation import BaseEvaluator, create_evaluation_callbacks
from collections.abc import Sequence
from crewai.crew import Crew
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from crewai.experimental.evaluation.evaluation_display import AgentAggregatedEvaluationResult
from contextlib import contextmanager
import threading

class ExecutionState:
    def __init__(self):
        self.traces: dict[str, Any] = {}
        self.current_agent_id: str | None = None
        self.current_task_id: str | None = None
        self.iteration: int = 1
        self.iterations_results: dict[int, dict[str, list[AgentEvaluationResult]]] = {}

class AgentEvaluator:
    def __init__(
        self,
        evaluators: Sequence[BaseEvaluator] | None = None,
        crew: Crew | None = None,
    ):
        self.crew: Crew | None = crew
        self.evaluators: Sequence[BaseEvaluator] | None = evaluators

        self.callback = create_evaluation_callbacks()
        self.console_formatter = ConsoleFormatter()
        self.display_formatter = EvaluationDisplayFormatter()

        self._thread_local: threading.local = threading.local()

        target_agents = []
        if crew is not None:
            assert crew and crew.agents is not None
            target_agents = crew.agents
        elif agents is not None:
            target_agents = agents

        for agent in target_agents:
            self.agent_evaluators[str(agent.id)] = self.evaluators

        self._subscribe_to_events()

    @property
    def _execution_state(self) -> ExecutionState:
        if not hasattr(self._thread_local, 'execution_state'):
            self._thread_local.execution_state = ExecutionState()
        return self._thread_local.execution_state

    def _subscribe_to_events(self) -> None:
        crewai_event_bus.register_handler(TaskCompletedEvent, self._handle_task_completed)

    def _handle_task_completed(self, source: object, event: TaskCompletedEvent) -> None:
        agent = event.task.agent
        if agent and str(getattr(agent, 'id', 'unknown')) in self.agent_evaluators:
            state = ExecutionState()
            state.current_agent_id = str(agent.id)
            state.current_task_id = str(event.task.id)

            trace = self.callback.get_trace(state.current_agent_id, state.current_task_id)

            if not trace:
                return

            result = self.evaluate(
                agent=agent,
                task=event.task,
                execution_trace=trace,
                final_output=event.output,
                state=state
            )

            current_iteration = self._execution_state.iteration
            if current_iteration not in self._execution_state.iterations_results:
                self._execution_state.iterations_results[current_iteration] = {}

            if agent.role not in self._execution_state.iterations_results[current_iteration]:
                self._execution_state.iterations_results[current_iteration][agent.role] = []

            self._execution_state.iterations_results[current_iteration][agent.role].append(result)

    @contextmanager
    def execution_context(self):
        state = ExecutionState()
        try:
            yield state
        finally:
            pass

    @property
    def _execution_state(self) -> ExecutionState:
        if not hasattr(self._thread_local, 'execution_state'):
            self._thread_local.execution_state = ExecutionState()
        return self._thread_local.execution_state

    def set_iteration(self, iteration: int) -> None:
        self._execution_state.iteration = iteration

    def reset_iterations_results(self) -> None:
        self._execution_state.iterations_results = {}

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
            eval_task = progress.add_task(f"Evaluating agents (iteration {self._execution_state.iteration})...", total=total_evals)

            with self.execution_context() as state:
                state.iteration = self._execution_state.iteration

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

                        state.current_agent_id = str(agent.id)
                        state.current_task_id = str(task.id)

                        with crewai_event_bus.scoped_handlers():
                            result = self.evaluate(
                                agent=agent,
                                task=task,
                                execution_trace=trace,
                                final_output=task.output,
                                state=state
                            )
                            evaluation_results[agent.role].append(result)
                            progress.update(eval_task, advance=1)

        self._execution_state.iterations_results[self._execution_state.iteration] = evaluation_results
        return evaluation_results

    def get_evaluation_results(self) -> dict[str, list[AgentEvaluationResult]]:
        if self._execution_state.iterations_results and self._execution_state.iteration in self._execution_state.iterations_results:
            return self._execution_state.iterations_results[self._execution_state.iteration]
        return {}

    def display_results_with_iterations(self) -> None:
        self.display_formatter.display_summary_results(self._execution_state.iterations_results)

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


            if self._execution_state.iterations_results and self._execution_state.iteration == max(self._execution_state.iterations_results.keys(), default=0):
                self.display_results_with_iterations()

            if include_evaluation_feedback:
                self.display_evaluation_with_feedback()

        return agent_results

    def display_evaluation_with_feedback(self) -> None:
        self.display_formatter.display_evaluation_with_feedback(self._execution_state.iterations_results)

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: dict[str, Any],
        final_output: Any,
        state: ExecutionState
    ) -> AgentEvaluationResult:
        result = AgentEvaluationResult(
            agent_id=state.current_agent_id or str(agent.id),
            task_id=state.current_task_id or str(task.id)
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

def create_default_evaluator(crew: Crew | None = None, agents: list[Agent] | None = None, llm: None = None):
    from crewai.experimental.evaluation import (
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

    return AgentEvaluator(evaluators=evaluators, crew=crew, agents=agents)
