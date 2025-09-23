import threading
from collections.abc import Sequence
from typing import Any

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentEvaluationCompletedEvent,
    AgentEvaluationFailedEvent,
    AgentEvaluationStartedEvent,
    LiteAgentExecutionCompletedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent
from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.experimental.evaluation.base_evaluator import (
    AgentAggregatedEvaluationResult,
    AgentEvaluationResult,
    AggregationStrategy,
    BaseEvaluator,
    EvaluationScore,
    MetricCategory,
)
from crewai.experimental.evaluation.evaluation_display import EvaluationDisplayFormatter
from crewai.experimental.evaluation.evaluation_listener import (
    create_evaluation_callbacks,
)
from crewai.task import Task


class ExecutionState:
    current_agent_id: str | None = None
    current_task_id: str | None = None

    def __init__(self):
        self.traces = {}
        self.iteration = 1
        self.iterations_results = {}
        self.agent_evaluators = {}


class AgentEvaluator:
    def __init__(
        self,
        agents: list[Agent] | list[BaseAgent],
        evaluators: Sequence[BaseEvaluator] | None = None,
    ):
        self.agents: list[Agent] | list[BaseAgent] = agents
        self.evaluators: Sequence[BaseEvaluator] | None = evaluators

        self.callback = create_evaluation_callbacks()
        self.console_formatter = ConsoleFormatter()
        self.display_formatter = EvaluationDisplayFormatter()

        self._thread_local: threading.local = threading.local()

        for agent in self.agents:
            self._execution_state.agent_evaluators[str(agent.id)] = self.evaluators

        self._subscribe_to_events()

    @property
    def _execution_state(self) -> ExecutionState:
        if not hasattr(self._thread_local, "execution_state"):
            self._thread_local.execution_state = ExecutionState()
        return self._thread_local.execution_state

    def _subscribe_to_events(self) -> None:
        from typing import cast

        crewai_event_bus.register_handler(
            TaskCompletedEvent, cast(Any, self._handle_task_completed)
        )
        crewai_event_bus.register_handler(
            LiteAgentExecutionCompletedEvent,
            cast(Any, self._handle_lite_agent_completed),
        )

    def _handle_task_completed(self, source: Any, event: TaskCompletedEvent) -> None:
        if event.task is None:
            raise ValueError("TaskCompletedEvent must have a task")
        agent = event.task.agent
        if (
            agent
            and str(getattr(agent, "id", "unknown"))
            in self._execution_state.agent_evaluators
        ):
            self.emit_evaluation_started_event(
                agent_role=agent.role,
                agent_id=str(agent.id),
                task_id=str(event.task.id),
            )

            state = ExecutionState()
            state.current_agent_id = str(agent.id)
            state.current_task_id = str(event.task.id)

            if state.current_agent_id is None or state.current_task_id is None:
                raise ValueError("Agent ID and Task ID must not be None")
            trace = self.callback.get_trace(
                state.current_agent_id, state.current_task_id
            )

            if not trace:
                return

            result = self.evaluate(
                agent=agent,
                task=event.task,
                execution_trace=trace,
                final_output=event.output,
                state=state,
            )

            current_iteration = self._execution_state.iteration
            if current_iteration not in self._execution_state.iterations_results:
                self._execution_state.iterations_results[current_iteration] = {}

            if (
                agent.role
                not in self._execution_state.iterations_results[current_iteration]
            ):
                self._execution_state.iterations_results[current_iteration][
                    agent.role
                ] = []

            self._execution_state.iterations_results[current_iteration][
                agent.role
            ].append(result)

    def _handle_lite_agent_completed(
        self, source: object, event: LiteAgentExecutionCompletedEvent
    ) -> None:
        agent_info = event.agent_info
        agent_id = str(agent_info["id"])

        if agent_id in self._execution_state.agent_evaluators:
            state = ExecutionState()
            state.current_agent_id = agent_id
            state.current_task_id = "lite_task"

            target_agent = None
            for agent in self.agents:
                if str(agent.id) == agent_id:
                    target_agent = agent
                    break

            if not target_agent:
                return

            if state.current_agent_id is None or state.current_task_id is None:
                raise ValueError("Agent ID and Task ID must not be None")
            trace = self.callback.get_trace(
                state.current_agent_id, state.current_task_id
            )

            if not trace:
                return

            result = self.evaluate(
                agent=target_agent,
                execution_trace=trace,
                final_output=event.output,
                state=state,
            )

            current_iteration = self._execution_state.iteration
            if current_iteration not in self._execution_state.iterations_results:
                self._execution_state.iterations_results[current_iteration] = {}

            agent_role = target_agent.role
            if (
                agent_role
                not in self._execution_state.iterations_results[current_iteration]
            ):
                self._execution_state.iterations_results[current_iteration][
                    agent_role
                ] = []

            self._execution_state.iterations_results[current_iteration][
                agent_role
            ].append(result)

    def set_iteration(self, iteration: int) -> None:
        self._execution_state.iteration = iteration

    def reset_iterations_results(self) -> None:
        self._execution_state.iterations_results = {}

    def get_evaluation_results(self) -> dict[str, list[AgentEvaluationResult]]:
        if (
            self._execution_state.iterations_results
            and self._execution_state.iteration
            in self._execution_state.iterations_results
        ):
            return self._execution_state.iterations_results[
                self._execution_state.iteration
            ]
        return {}

    def display_results_with_iterations(self) -> None:
        self.display_formatter.display_summary_results(
            self._execution_state.iterations_results
        )

    def get_agent_evaluation(
        self,
        strategy: AggregationStrategy = AggregationStrategy.SIMPLE_AVERAGE,
        include_evaluation_feedback: bool = True,
    ) -> dict[str, AgentAggregatedEvaluationResult]:
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
                    strategy=strategy,
                )

                agent_results[agent_role] = aggregated_result

            if (
                self._execution_state.iterations_results
                and self._execution_state.iteration
                == max(self._execution_state.iterations_results.keys(), default=0)
            ):
                self.display_results_with_iterations()

            if include_evaluation_feedback:
                self.display_evaluation_with_feedback()

        return agent_results

    def display_evaluation_with_feedback(self) -> None:
        self.display_formatter.display_evaluation_with_feedback(
            self._execution_state.iterations_results
        )

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: Any,
        state: ExecutionState,
        task: Task | None = None,
    ) -> AgentEvaluationResult:
        result = AgentEvaluationResult(
            agent_id=state.current_agent_id or str(agent.id),
            task_id=state.current_task_id or (str(task.id) if task else "unknown_task"),
        )

        if self.evaluators is None:
            raise ValueError("Evaluators must be initialized")
        task_id = str(task.id) if task else None
        for evaluator in self.evaluators:
            try:
                self.emit_evaluation_started_event(
                    agent_role=agent.role, agent_id=str(agent.id), task_id=task_id
                )
                score = evaluator.evaluate(
                    agent=agent,
                    task=task,
                    execution_trace=execution_trace,
                    final_output=final_output,
                )
                result.metrics[evaluator.metric_category] = score
                self.emit_evaluation_completed_event(
                    agent_role=agent.role,
                    agent_id=str(agent.id),
                    task_id=task_id,
                    metric_category=evaluator.metric_category,
                    score=score,
                )
            except Exception as e:  # noqa: PERF203
                self.emit_evaluation_failed_event(
                    agent_role=agent.role,
                    agent_id=str(agent.id),
                    task_id=task_id,
                    error=str(e),
                )
                self.console_formatter.print(
                    f"Error in {evaluator.metric_category.value} evaluator: {e!s}"
                )

        return result

    def emit_evaluation_started_event(
        self, agent_role: str, agent_id: str, task_id: str | None = None
    ):
        crewai_event_bus.emit(
            self,
            AgentEvaluationStartedEvent(
                agent_role=agent_role,
                agent_id=agent_id,
                task_id=task_id,
                iteration=self._execution_state.iteration,
            ),
        )

    def emit_evaluation_completed_event(
        self,
        agent_role: str,
        agent_id: str,
        task_id: str | None = None,
        metric_category: MetricCategory | None = None,
        score: EvaluationScore | None = None,
    ):
        crewai_event_bus.emit(
            self,
            AgentEvaluationCompletedEvent(
                agent_role=agent_role,
                agent_id=agent_id,
                task_id=task_id,
                iteration=self._execution_state.iteration,
                metric_category=metric_category,
                score=score,
            ),
        )

    def emit_evaluation_failed_event(
        self, agent_role: str, agent_id: str, error: str, task_id: str | None = None
    ):
        crewai_event_bus.emit(
            self,
            AgentEvaluationFailedEvent(
                agent_role=agent_role,
                agent_id=agent_id,
                task_id=task_id,
                iteration=self._execution_state.iteration,
                error=error,
            ),
        )


def create_default_evaluator(agents: list[Agent] | list[BaseAgent], llm: None = None):
    from crewai.experimental.evaluation import (
        GoalAlignmentEvaluator,
        ParameterExtractionEvaluator,
        ReasoningEfficiencyEvaluator,
        SemanticQualityEvaluator,
        ToolInvocationEvaluator,
        ToolSelectionEvaluator,
    )

    evaluators = [
        GoalAlignmentEvaluator(llm=llm),
        SemanticQualityEvaluator(llm=llm),
        ToolSelectionEvaluator(llm=llm),
        ParameterExtractionEvaluator(llm=llm),
        ToolInvocationEvaluator(llm=llm),
        ReasoningEfficiencyEvaluator(llm=llm),
    ]

    return AgentEvaluator(evaluators=evaluators, agents=agents)
