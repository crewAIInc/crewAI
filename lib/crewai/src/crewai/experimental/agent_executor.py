from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars
from datetime import datetime
import inspect
import json
import threading
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from rich.console import Console
from rich.text import Text

from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.core.providers.human_input import get_provider
from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.utils import (
    is_tracing_enabled_in_context,
)
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.events.types.observation_events import (
    GoalAchievedEarlyEvent,
    PlanRefinementEvent,
    PlanReplanTriggeredEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.flow.flow import Flow, StateProxy, listen, or_, router, start
from crewai.flow.types import FlowMethodName
from crewai.hooks.llm_hooks import (
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.hooks.types import (
    AfterLLMCallHookCallable,
    AfterLLMCallHookType,
    BeforeLLMCallHookCallable,
    BeforeLLMCallHookType,
)
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.agent_utils import (
    check_native_tool_support,
    enforce_rpm_limit,
    extract_tool_call_info,
    format_message_for_llm,
    get_llm_response,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    is_inside_event_loop,
    is_tool_call_list,
    parse_tool_call_args,
    process_llm_response,
    setup_native_tools,
    track_delegation_if_needed,
)
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.planning_types import (
    PlanStep,
    StepObservation,
    TodoItem,
    TodoList,
)
from crewai.utilities.printer import Printer
from crewai.utilities.step_execution_context import StepExecutionContext
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.training_handler import CrewTrainingHandler
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.tools.tool_types import ToolResult
    from crewai.utilities.prompts import StandardPromptResult, SystemPromptResult


class AgentExecutorState(BaseModel):
    """Structured state for agent executor flow.

    Holds both ReAct iteration state and Plan-and-Execute state
    (todos, observations, replan tracking) in a single validated model.
    """

    messages: list[LLMMessage] = Field(default_factory=list)
    iterations: int = Field(default=0)
    current_answer: AgentAction | AgentFinish | None = Field(default=None)
    is_finished: bool = Field(default=False)
    ask_for_human_input: bool = Field(default=False)
    use_native_tools: bool = Field(default=False)
    pending_tool_calls: list[Any] = Field(default_factory=list)
    plan: str | None = Field(default=None, description="Generated execution plan")
    plan_ready: bool = Field(
        default=False, description="Whether agent is ready to execute"
    )
    todos: TodoList = Field(
        default_factory=TodoList, description="Todo list for tracking plan execution"
    )
    replan_count: int = Field(
        default=0, description="Number of times the plan has been regenerated"
    )
    last_replan_reason: str | None = Field(
        default=None, description="Reason for the last replan, if any"
    )
    observations: dict[int, StepObservation] = Field(
        default_factory=dict,
        description="Planner's observation per step (keyed by step_number)",
    )
    execution_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Audit trail for debugging (NOT used for LLM calls)",
    )


class AgentExecutor(Flow[AgentExecutorState], CrewAgentExecutorMixin):
    """Agent Executor for both standalone agents and crew-bound agents.

    Inherits from:
    - Flow[AgentExecutorState]: Provides flow orchestration capabilities
    - CrewAgentExecutorMixin: Provides memory methods (short/long/external term)

    This executor can operate in two modes:
    - Standalone mode: When crew and task are None (used by Agent.kickoff())
    - Crew mode: When crew and task are provided (used by Agent.execute_task())

    Note: Multiple instances may be created during agent initialization
    (cache setup, RPM controller setup, etc.) but only the final instance
    should execute tasks via invoke().
    """

    def __init__(
        self,
        llm: BaseLLM,
        agent: Agent,
        prompt: SystemPromptResult | StandardPromptResult,
        max_iter: int,
        tools: list[CrewStructuredTool],
        tools_names: str,
        stop_words: list[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        task: Task | None = None,
        crew: Crew | None = None,
        step_callback: Any = None,
        original_tools: list[BaseTool] | None = None,
        function_calling_llm: BaseLLM | Any | None = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Callable[[], bool] | None = None,
        callbacks: list[Any] | None = None,
        response_model: type[BaseModel] | None = None,
        i18n: I18N | None = None,
    ) -> None:
        """Initialize the flow-based agent executor.

        Args:
            llm: Language model instance.
            agent: Agent to execute.
            prompt: Prompt templates.
            max_iter: Maximum iterations.
            tools: Available tools.
            tools_names: Tool names string.
            stop_words: Stop word list.
            tools_description: Tool descriptions.
            tools_handler: Tool handler instance.
            task: Optional task to execute (None for standalone agent execution).
            crew: Optional crew instance (None for standalone agent execution).
            step_callback: Optional step callback.
            original_tools: Original tool list.
            function_calling_llm: Optional function calling LLM.
            respect_context_window: Respect context limits.
            request_within_rpm_limit: RPM limit check function.
            callbacks: Optional callbacks list.
            response_model: Optional Pydantic model for structured outputs.
        """
        self._i18n: I18N = i18n or get_i18n()
        self.llm = llm
        self.task: Task | None = task
        self.agent = agent
        self.crew: Crew | None = crew
        self.prompt = prompt
        self.tools = tools
        self.tools_names = tools_names
        self.stop = stop_words
        self.max_iter = max_iter
        self.callbacks = callbacks or []
        self._printer: Printer = Printer()
        self.tools_handler = tools_handler
        self.original_tools = original_tools or []
        self.step_callback = step_callback
        self.tools_description = tools_description
        self.function_calling_llm = function_calling_llm
        self.respect_context_window = respect_context_window
        self.request_within_rpm_limit = request_within_rpm_limit
        self.response_model = response_model
        self.log_error_after = 3
        self._console: Console = Console()

        # Error context storage for recovery
        self._last_parser_error: OutputParserError | None = None
        self._last_context_error: Exception | None = None

        # Execution guard to prevent concurrent/duplicate executions
        self._execution_lock = threading.Lock()
        self._finalize_lock = threading.Lock()
        self._finalize_called: bool = False
        self._is_executing: bool = False
        self._has_been_invoked: bool = False
        self._flow_initialized: bool = False

        self._instance_id = str(uuid4())[:8]

        self.before_llm_call_hooks: list[
            BeforeLLMCallHookType | BeforeLLMCallHookCallable
        ] = []
        self.after_llm_call_hooks: list[
            AfterLLMCallHookType | AfterLLMCallHookCallable
        ] = []
        self.before_llm_call_hooks.extend(get_before_llm_call_hooks())
        self.after_llm_call_hooks.extend(get_after_llm_call_hooks())

        if self.llm:
            existing_stop = getattr(self.llm, "stop", [])
            self.llm.stop = list(
                set(
                    existing_stop + self.stop
                    if isinstance(existing_stop, list)
                    else self.stop
                )
            )
        self._state = AgentExecutorState()

        # Plan-and-Execute components (Phase 2)
        # Lazy-imported to avoid circular imports during module load
        self._step_executor: Any = None
        self._planner_observer: Any = None

    def _ensure_flow_initialized(self) -> None:
        """Ensure Flow.__init__() has been called.

        This is deferred from __init__ to prevent FlowCreatedEvent emission
        during agent setup when multiple executor instances are created.
        Only the instance that actually executes via invoke() will emit events.
        """
        if not self._flow_initialized:
            current_tracing = is_tracing_enabled_in_context()
            # Now call Flow's __init__ which will replace self._state
            # with Flow's managed state. Suppress flow events since this is
            # an agent executor, not a user-facing flow.
            super().__init__(
                suppress_flow_events=True,
                tracing=current_tracing if current_tracing else None,
                max_method_calls=self.max_iter * 10,
            )
            self._flow_initialized = True

    def _check_native_tool_support(self) -> bool:
        """Check if LLM supports native function calling."""
        return check_native_tool_support(self.llm, self.original_tools)

    def _setup_native_tools(self) -> None:
        """Convert tools to OpenAI schema format for native function calling."""
        if self.original_tools:
            (
                self._openai_tools,
                self._available_functions,
                self._tool_name_mapping,
            ) = setup_native_tools(self.original_tools)

    def _is_tool_call_list(self, response: list[Any]) -> bool:
        """Check if a response is a list of tool calls."""
        return is_tool_call_list(response)

    @property
    def use_stop_words(self) -> bool:
        """Check to determine if stop words are being used.

        Returns:
            bool: True if stop words should be used.
        """
        return self.llm.supports_stop_words() if self.llm else False

    @property
    def state(self) -> AgentExecutorState:
        """Get state - returns temporary state if Flow not yet initialized.

        Flow initialization is deferred to prevent event emission during agent setup.
        Returns the temporary state until invoke() is called.
        """
        if self._flow_initialized and hasattr(self, "_state_lock"):
            return StateProxy(self._state, self._state_lock)  # type: ignore[return-value]
        return self._state

    @property
    def iterations(self) -> int:
        """Compatibility property for mixin - returns state iterations."""
        return self._state.iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        """Set state iterations."""
        self._state.iterations = value

    @property
    def messages(self) -> list[LLMMessage]:
        """Compatibility property - returns state messages."""
        return self._state.messages

    @messages.setter
    def messages(self, value: list[LLMMessage]) -> None:
        """Set state messages."""
        self._state.messages = value

    @start()
    def generate_plan(self) -> None:
        """Generate execution plan if planning is enabled.

        This is the entry point for the agent execution flow. If planning is
        enabled on the agent, it generates a plan before execution begins.
        The plan is stored in state and todos are created from the steps.
        """
        if not getattr(self.agent, "planning_enabled", False):
            return

        try:
            from crewai.utilities.reasoning_handler import AgentReasoning

            if self.task:
                planning_handler = AgentReasoning(agent=self.agent, task=self.task)
            else:
                # For kickoff() path - use input text directly, no Task needed
                input_text = getattr(self, "_kickoff_input", "")
                planning_handler = AgentReasoning(
                    agent=self.agent,
                    description=input_text or "Complete the requested task",
                    expected_output="Complete the task successfully",
                )

            output = planning_handler.handle_agent_reasoning()

            self.state.plan = output.plan.plan
            self.state.plan_ready = output.plan.ready

            if self.state.plan_ready and output.plan.steps:
                self._create_todos_from_plan(output.plan.steps)

            # Plan is stored in state.plan and used by the execution flow.
            # Do NOT mutate task.description — it's a shared object that
            # accumulates plan text on re-invoke.

        except Exception as e:
            if hasattr(self.agent, "_logger"):
                self.agent._logger.log("error", f"Error during planning: {e!s}")

    def _create_todos_from_plan(self, steps: list[PlanStep]) -> None:
        """Convert plan steps into trackable todo items.

        Args:
            steps: List of PlanStep objects from the reasoning handler.
        """
        todos: list[TodoItem] = []
        for step in steps:
            todo = TodoItem(
                step_number=step.step_number,
                description=step.description,
                tool_to_use=step.tool_to_use,
                depends_on=step.depends_on,
                status="pending",
            )
            todos.append(todo)

        self.state.todos = TodoList(items=todos)

    # -------------------------------------------------------------------------
    # Plan-and-Execute: Component Initialization
    # -------------------------------------------------------------------------

    def _ensure_step_executor(self) -> Any:
        """Lazily create the StepExecutor (avoids circular imports)."""
        if self._step_executor is None:
            from crewai.agents.step_executor import StepExecutor

            self._step_executor = StepExecutor(
                llm=self.llm,
                tools=self.tools,
                agent=self.agent,
                original_tools=self.original_tools,
                tools_handler=self.tools_handler,
                task=self.task,
                crew=self.crew,
                function_calling_llm=self.function_calling_llm,
                request_within_rpm_limit=self.request_within_rpm_limit,
                callbacks=self.callbacks,
                i18n=self._i18n,
            )
        return self._step_executor

    def _ensure_planner_observer(self) -> Any:
        """Lazily create the PlannerObserver (avoids circular imports)."""
        if self._planner_observer is None:
            from crewai.agents.planner_observer import PlannerObserver

            self._planner_observer = PlannerObserver(
                agent=self.agent,
                task=self.task,
                kickoff_input=getattr(self, "_kickoff_input", ""),
            )
        return self._planner_observer

    def _get_reasoning_effort(self) -> str:
        """Get the reasoning effort level from the agent's planning config.

        Returns:
            The reasoning effort level: "low", "medium", or "high".
            Defaults to "medium" if no planning config is set so that
            step failures reliably trigger replanning rather than being
            silently ignored.
        """
        config = getattr(self.agent, "planning_config", None)
        if config is not None and hasattr(config, "reasoning_effort"):
            return config.reasoning_effort
        return "medium"

    def _get_max_replans(self) -> int:
        """Get max replans from planning config or default to 3."""
        config = getattr(self.agent, "planning_config", None)
        if config is not None and hasattr(config, "max_replans"):
            return config.max_replans
        return 3

    def _get_max_step_iterations(self) -> int:
        """Get max step iterations from planning config or default to 15."""
        config = getattr(self.agent, "planning_config", None)
        if config is not None and hasattr(config, "max_step_iterations"):
            return config.max_step_iterations
        return 15

    def _get_step_timeout(self) -> int | None:
        """Get per-step timeout from planning config or default to None."""
        config = getattr(self.agent, "planning_config", None)
        if config is not None and hasattr(config, "step_timeout"):
            return config.step_timeout
        return None

    def _build_context_for_todo(self, todo: TodoItem) -> StepExecutionContext:
        """Build an isolated execution context for a single todo.

        Passes only final results from completed dependencies — never
        execution traces, tool calls, or LLM message history.

        Args:
            todo: The todo item to build context for.

        Returns:
            Immutable StepExecutionContext with dependency results.
        """
        dependency_results: dict[int, str] = {}
        for dep_num in todo.depends_on:
            dep_todo = self.state.todos.get_by_step_number(dep_num)
            if dep_todo and dep_todo.result:
                dependency_results[dep_num] = dep_todo.result

        task_description = ""
        task_goal = ""
        if self.task:
            task_description = self.task.description or ""
            task_goal = self.task.expected_output or ""
        else:
            task_description = getattr(self, "_kickoff_input", "")
            task_goal = "Complete the task successfully"

        return StepExecutionContext(
            task_description=task_description,
            task_goal=task_goal,
            dependency_results=dependency_results,
        )

    # -------------------------------------------------------------------------
    # Plan-and-Execute: New Observation-Driven Flow Methods
    # -------------------------------------------------------------------------

    @router("step_executed")
    def observe_step_result(
        self,
    ) -> Literal["step_observed_low", "step_observed_medium", "step_observed_high"]:
        """Observe step result and route based on reasoning_effort level.

        Always runs PlannerObserver.observe() to validate whether the step
        succeeded. Then routes to the appropriate handler based on the
        agent's reasoning_effort setting:

        - "low": observe → mark complete → continue (no replan/refine)
        - "medium": observe → replan on failure only (no refine)
        - "high": observe → full decide pipeline (replan/refine/goal-achieved)

        Based on PLAN-AND-ACT Section 3.3.
        """
        current_todo = self.state.todos.current_todo
        effort = self._get_reasoning_effort()

        if not current_todo:
            # No todo — route to low handler which will just continue
            return "step_observed_low"

        observer = self._ensure_planner_observer()
        all_completed = self.state.todos.get_completed_todos()
        remaining = self.state.todos.get_pending_todos()

        observation = observer.observe(
            completed_step=current_todo,
            result=current_todo.result or "",
            all_completed=all_completed,
            remaining_todos=remaining,
        )

        self.state.observations[current_todo.step_number] = observation

        # Log observation for debugging
        self.state.execution_log.append(
            {
                "type": "observation",
                "step_number": current_todo.step_number,
                "step_completed_successfully": observation.step_completed_successfully,
                "key_information_learned": observation.key_information_learned,
                "remaining_plan_still_valid": observation.remaining_plan_still_valid,
                "needs_full_replan": observation.needs_full_replan,
                "goal_already_achieved": observation.goal_already_achieved,
                "reasoning_effort": effort,
            }
        )

        if self.agent.verbose:
            self._printer.print(
                content=(
                    f"[Observe] Step {current_todo.step_number} "
                    f"(effort={effort}): "
                    f"success={observation.step_completed_successfully}, "
                    f"plan_valid={observation.remaining_plan_still_valid}, "
                    f"learned={observation.key_information_learned[:80]}..."
                ),
                color="cyan",
            )

        if effort == "high":
            return "step_observed_high"
        if effort == "medium":
            return "step_observed_medium"
        return "step_observed_low"

    # -- Low effort: observe → mark complete → continue (no replan/refine) --

    @router("step_observed_low")
    def handle_step_observed_low(
        self,
    ) -> Literal["continue_plan", "replan_now"]:
        """Low reasoning effort: mark step complete and continue linearly.

        Skips the refine/goal-achieved pipeline but still gates on hard
        failures: if the observer says the step failed AND a full replan is
        needed, we route to ``replan_now`` rather than blindly continuing.
        This prevents cascading failures where every subsequent step builds
        on a broken foundation.
        """
        current_todo = self.state.todos.current_todo
        if not current_todo:
            return "continue_plan"

        observation = self.state.observations.get(current_todo.step_number)

        # Even at low effort, don't ignore a hard step failure.
        # A hard failure is one where the step did not succeed AND a replan
        # is explicitly required (e.g. required tool not found, permission
        # denied, environment misconfiguration).
        if (
            observation
            and not observation.step_completed_successfully
            and observation.needs_full_replan
        ):
            self.state.todos.mark_failed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content=(
                        f"[Low] Step {current_todo.step_number} hard-failed "
                        f"— triggering replan: {observation.replan_reason}"
                    ),
                    color="yellow",
                )
            self.state.last_replan_reason = (
                observation.replan_reason or "Step did not complete successfully"
            )
            return "replan_now"

        self.state.todos.mark_completed(
            current_todo.step_number, result=current_todo.result
        )

        if self.agent.verbose:
            completed = self.state.todos.completed_count
            total = len(self.state.todos.items)
            self._printer.print(
                content=f"[Low] Step {current_todo.step_number} done ({completed}/{total}) — continuing",
                color="green",
            )

        return "continue_plan"

    # -- Medium effort: observe → replan on failure only (no refine) --

    @router("step_observed_medium")
    def handle_step_observed_medium(
        self,
    ) -> Literal["continue_plan", "replan_now"]:
        """Medium reasoning effort: replan only when a step fails.

        On success, marks the step complete and continues without
        refinement or early goal detection. On failure, triggers replanning
        so the agent can recover.
        """
        current_todo = self.state.todos.current_todo
        if not current_todo:
            return "continue_plan"

        observation = self.state.observations.get(current_todo.step_number)

        # If observation is missing or step succeeded — continue
        if not observation or observation.step_completed_successfully:
            self.state.todos.mark_completed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                completed = self.state.todos.completed_count
                total = len(self.state.todos.items)
                self._printer.print(
                    content=f"[Medium] Step {current_todo.step_number} succeeded ({completed}/{total}) — continuing",
                    color="green",
                )
            return "continue_plan"

        # Step failed — only replan if observer explicitly requires it,
        # otherwise mark done and continue (same gate as low-effort).
        if observation.needs_full_replan:
            self.state.todos.mark_failed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content=(
                        f"[Medium] Step {current_todo.step_number} failed + replan required "
                        f"— triggering replan: {observation.replan_reason}"
                    ),
                    color="yellow",
                )
            self.state.last_replan_reason = (
                observation.replan_reason or "Step did not complete successfully"
            )
            return "replan_now"

        # Step failed but observer does not require a full replan — mark as
        # failed (not completed) so get_failed_todos() tracks it correctly.
        self.state.todos.mark_failed(
            current_todo.step_number, result=current_todo.result
        )
        if self.agent.verbose:
            failed = len(self.state.todos.get_failed_todos())
            total = len(self.state.todos.items)
            self._printer.print(
                content=(
                    f"[Medium] Step {current_todo.step_number} failed but no replan needed "
                    f"({failed} failed/{total} total) — continuing"
                ),
                color="yellow",
            )
        return "continue_plan"

    # -- High effort: full observation pipeline (existing behavior) --

    @router("step_observed_high")
    def decide_next_action(
        self,
    ) -> Literal[
        "goal_achieved",
        "replan_now",
        "refine_and_continue",
        "continue_plan",
    ]:
        """High reasoning effort: full observation-driven routing.

        Routes based on the Planner's observation. Can trigger early goal
        achievement, full replanning, lightweight refinement, or simple
        continuation. This is the most adaptive but highest-latency path.
        """
        current_todo = self.state.todos.current_todo
        if not current_todo:
            return "continue_plan"

        observation = self.state.observations.get(current_todo.step_number)
        if not observation:
            # No observation available — default to continue
            self.state.todos.mark_completed(current_todo.step_number)
            return "continue_plan"

        # Goal already achieved — early termination
        if observation.goal_already_achieved:
            self.state.todos.mark_completed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content="[Decide] Goal achieved early — finalizing",
                    color="green",
                )
            return "goal_achieved"

        # Full replan needed
        if observation.needs_full_replan:
            self.state.todos.mark_failed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content=f"[Decide] Full replan needed: {observation.replan_reason}",
                    color="yellow",
                )
            self.state.last_replan_reason = observation.replan_reason
            return "replan_now"

        # Step failed — also trigger replan
        if not observation.step_completed_successfully:
            self.state.todos.mark_failed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content="[Decide] Step failed — triggering replan",
                    color="yellow",
                )
            self.state.last_replan_reason = "Step did not complete successfully"
            return "replan_now"

        # Plan still valid but needs refinement
        if observation.remaining_plan_still_valid and observation.suggested_refinements:
            self.state.todos.mark_completed(
                current_todo.step_number, result=current_todo.result
            )
            if self.agent.verbose:
                self._printer.print(
                    content="[Decide] Plan valid but refining upcoming steps",
                    color="cyan",
                )
            return "refine_and_continue"

        # Plan still valid, no refinements needed — just continue
        self.state.todos.mark_completed(
            current_todo.step_number, result=current_todo.result
        )
        if self.agent.verbose:
            completed = self.state.todos.completed_count
            total = len(self.state.todos.items)
            self._printer.print(
                content=f"[Decide] Continue plan ({completed}/{total} done)",
                color="green",
            )
        return "continue_plan"

    @router("refine_and_continue")
    def handle_refine_and_continue(self) -> Literal["has_todos"]:
        """Lightweight plan refinement — update pending todo descriptions.

        The Planner sharpens upcoming step descriptions based on what was
        learned, without regenerating the entire plan.
        """
        # Find the most recent observation with refinements
        recent_observation: StepObservation | None = None
        last_step: int = 0
        if self.state.observations:
            last_step = max(self.state.observations.keys())
            recent_observation = self.state.observations[last_step]

        if recent_observation and recent_observation.suggested_refinements:
            observer = self._ensure_planner_observer()
            remaining = self.state.todos.get_pending_todos()

            observer.apply_refinements(recent_observation, remaining)

            refinement_summaries = [
                f"Step {r.step_number}: {r.new_description}"
                for r in recent_observation.suggested_refinements
            ]

            crewai_event_bus.emit(
                self.agent,
                event=PlanRefinementEvent(
                    agent_role=self.agent.role,
                    step_number=last_step,
                    step_description="",
                    refined_step_count=len(remaining),
                    refinements=refinement_summaries,
                    from_task=self.task,
                    from_agent=self.agent,
                ),
            )

            if self.agent.verbose:
                self._printer.print(
                    content=f"[Refine] Updated {len(remaining)} pending step(s)",
                    color="cyan",
                )

        return "has_todos"

    @router("continue_plan")
    def handle_continue_plan(self) -> Literal["has_todos", "all_todos_complete"]:
        """Continue to the next todo after a successful step."""
        if self.state.todos.is_complete:
            return "all_todos_complete"
        return "has_todos"

    @router("goal_achieved")
    def handle_goal_achieved(self) -> Literal["all_todos_complete"]:
        """Handle early goal achievement — skip remaining todos."""
        completed = self.state.todos.get_completed_todos()
        remaining = self.state.todos.get_pending_todos()

        # Emit goal achieved early event
        crewai_event_bus.emit(
            self.agent,
            event=GoalAchievedEarlyEvent(
                agent_role=self.agent.role,
                step_number=completed[-1].step_number if completed else 0,
                step_description="",
                steps_completed=len(completed),
                steps_remaining=len(remaining),
                from_task=self.task,
                from_agent=self.agent,
            ),
        )

        if self.agent.verbose:
            self._printer.print(
                content="Goal achieved early — skipping remaining steps",
                color="green",
            )
        return "all_todos_complete"

    @router("replan_now")
    def handle_replan_now(
        self,
    ) -> Literal["has_todos", "all_todos_complete"]:
        """Handle full replanning — regenerate the remaining plan.

        Preserves completed todo results and replaces only pending steps.
        """
        max_replans = self._get_max_replans()

        if self.state.replan_count >= max_replans:
            if self.agent.verbose:
                self._printer.print(
                    content=f"Max replans ({max_replans}) reached — finalizing with current results",
                    color="yellow",
                )
            return "all_todos_complete"

        self.state.replan_count += 1
        reason = self.state.last_replan_reason or "Dynamic replan triggered"
        completed = self.state.todos.get_completed_todos()

        # Emit replan triggered event
        crewai_event_bus.emit(
            self.agent,
            event=PlanReplanTriggeredEvent(
                agent_role=self.agent.role,
                step_number=completed[-1].step_number if completed else 0,
                step_description="",
                replan_reason=reason,
                replan_count=self.state.replan_count,
                completed_steps_preserved=len(completed),
                from_task=self.task,
                from_agent=self.agent,
            ),
        )

        self._trigger_replan(reason)

        if self.state.todos.get_pending_todos():
            return "has_todos"
        return "all_todos_complete"

    # -------------------------------------------------------------------------
    # Todo-Driven Execution Flow
    # -------------------------------------------------------------------------

    @router(generate_plan)
    def check_todos_available(
        self,
    ) -> Literal["has_todos", "no_todos", "planning_disabled"]:
        """Check if todos were created from planning.

        Routes to todo-driven execution if todos exist, otherwise falls back
        to standard execution flow.
        """
        if not getattr(self.agent, "planning_enabled", False):
            return "planning_disabled"
        if not self.state.todos.items:
            return "no_todos"
        return "has_todos"

    @router("has_todos")
    def get_ready_todos_method(
        self,
    ) -> Literal[
        "single_todo_ready",
        "multiple_todos_ready",
        "all_todos_complete",
        "needs_replan",
    ]:
        """Find todos whose dependencies are satisfied.

        Determines if we can execute a single todo sequentially or multiple
        todos in parallel.
        """
        ready = self.state.todos.get_ready_todos()

        if not ready:
            if self.state.todos.is_complete:
                return "all_todos_complete"
            # Stuck state: pending todos exist but none are ready (unsatisfied
            # dependencies, e.g. a dependency was never completed). Trigger a
            # replan so the planner can generate a new plan that unblocks
            # execution rather than erroneously finalizing.
            self.state.last_replan_reason = (
                "No todos are ready but plan is not complete — "
                "likely a dependency deadlock or missing completion"
            )
            return "needs_replan"

        if len(ready) == 1:
            # Mark the single ready todo as running
            self.state.todos.mark_running(ready[0].step_number)
            return "single_todo_ready"

        # Multiple todos ready - can parallelize
        return "multiple_todos_ready"

    @router("single_todo_ready")
    def execute_todo_sequential(
        self,
    ) -> Literal["step_executed", "todo_injected"]:
        """Execute a single todo using StepExecutor (Plan-and-Execute mode)
        or fall back to the old ReAct injection (legacy mode).

        In Plan-and-Execute mode: executes the step in isolation via
        StepExecutor, stores the result, and routes to the observation step.

        In legacy mode: injects context into the shared message list and
        routes to the ReAct loop.
        """
        current = self.state.todos.current_todo
        if not current:
            return "todo_injected"  # Fall through to legacy

        # Plan-and-Execute path: use StepExecutor for isolated execution
        if getattr(self.agent, "planning_enabled", False):
            if self.agent.verbose:
                self._printer.print(
                    content=(
                        f"[Execute] Step {current.step_number}: "
                        f"{current.description[:60]}..."
                    ),
                    color="cyan",
                )

            step_executor = self._ensure_step_executor()
            context = self._build_context_for_todo(current)
            result = step_executor.execute(
                current,
                context,
                max_step_iterations=self._get_max_step_iterations(),
                step_timeout=self._get_step_timeout(),
            )

            # Store result on the todo (do NOT mark completed — observation decides)
            current.result = result.result

            # Log to audit trail
            self.state.execution_log.append(
                {
                    "type": "step_execution",
                    "step_number": current.step_number,
                    "success": result.success,
                    "result_preview": result.result[:200] if result.result else "",
                    "error": result.error,
                    "tool_calls": result.tool_calls_made,
                    "execution_time": result.execution_time,
                }
            )

            if self.agent.verbose:
                status = "success" if result.success else "failed"
                self._printer.print(
                    content=(
                        f"[Execute] Step {current.step_number} {status} "
                        f"({result.execution_time:.1f}s, "
                        f"{len(result.tool_calls_made)} tool calls)"
                    ),
                    color="green" if result.success else "red",
                )

            return "step_executed"

        # Legacy path: inject context into shared messages for ReAct loop
        self._inject_todo_context(current)
        return "todo_injected"

    def _inject_todo_context(self, todo: TodoItem) -> None:
        """Inject todo-specific context into the conversation.

        Args:
            todo: The todo item to inject context for.
        """
        # Build focused task prompt. Context from previous steps is already
        # in self.state.messages as SYSTEM messages (added by _mark_todo_as_completed)
        prompt = self._build_todo_prompt(todo, include_dependencies=False)
        todo_message: LLMMessage = {
            "role": "user",
            "content": prompt,
        }
        self.state.messages.append(todo_message)

    def _build_todo_prompt(
        self, todo: TodoItem, include_dependencies: bool = True
    ) -> str:
        """Build a focused prompt for executing a single todo.

        Args:
            todo: The todo item to build a prompt for.
            include_dependencies: Whether to include dependency results in this prompt.

        Returns:
            A prompt string focused on this specific step.
        """
        total = len(self.state.todos.items)
        parts = [f"**Current Step {todo.step_number}/{total}**"]
        parts.append(f"Task: {todo.description}")

        if todo.tool_to_use:
            parts.append(f"Suggested tool: {todo.tool_to_use}")

        # Include results from completed dependencies if requested (used for parallel execution)
        if include_dependencies and todo.depends_on:
            dep_results = []
            for dep_num in todo.depends_on:
                dep = self.state.todos.get_by_step_number(dep_num)
                if dep and dep.result:
                    dep_results.append(f"Step {dep_num} result: {dep.result}")
            if dep_results:
                parts.append("\nContext from previous steps:")
                parts.extend(dep_results)

        parts.append("\nComplete this step. Once done, provide your result.")
        return "\n".join(parts)

    @router("multiple_todos_ready")
    async def execute_todos_parallel(self) -> Literal["parallel_todos_complete"]:
        """Execute multiple independent todos concurrently via StepExecutor.

        Uses the same StepExecutor path as sequential execution so that
        parallel steps get: multi-turn action loops, tool usage events,
        security context, vision sentinel handling, and hooks.

        After all steps complete, each result is observed sequentially
        through PlannerObserver so the planning system stays informed.
        """

        ready = self.state.todos.get_ready_todos()

        # Mark all ready todos as running
        for todo in ready:
            self.state.todos.mark_running(todo.step_number)

        # Build context and executor for each todo, then run in parallel
        async def _run_step(todo: TodoItem) -> tuple[TodoItem, object]:
            step_executor = self._ensure_step_executor()
            context = self._build_context_for_todo(todo)
            result = await asyncio.to_thread(
                step_executor.execute,
                todo,
                context,
                self._get_max_step_iterations(),
                self._get_step_timeout(),
            )
            return todo, result

        gathered = await asyncio.gather(
            *[_run_step(todo) for todo in ready],
            return_exceptions=True,
        )

        # Process results: store on todos and log, then observe each.
        # asyncio.gather preserves input order, so zip gives us the exact
        # todo ↔ result (or exception) mapping.
        step_results: list[tuple[TodoItem, object]] = []
        for todo, item in zip(ready, gathered, strict=True):
            if isinstance(item, Exception):
                error_msg = f"Error: {item!s}"
                todo.result = error_msg
                self.state.todos.mark_failed(todo.step_number, result=error_msg)
                if self.agent.verbose:
                    self._printer.print(
                        content=f"Todo {todo.step_number} failed: {error_msg}",
                        color="red",
                    )
            else:
                _returned_todo, result = item
                todo.result = result.result

                self.state.execution_log.append(
                    {
                        "type": "step_execution",
                        "step_number": todo.step_number,
                        "success": result.success,
                        "result_preview": result.result[:200] if result.result else "",
                        "error": result.error,
                        "tool_calls": result.tool_calls_made,
                        "execution_time": result.execution_time,
                    }
                )

                if self.agent.verbose:
                    status = "success" if result.success else "failed"
                    self._printer.print(
                        content=(
                            f"[Execute] Step {todo.step_number} {status} "
                            f"({result.execution_time:.1f}s, "
                            f"{len(result.tool_calls_made)} tool calls)"
                        ),
                        color="green" if result.success else "red",
                    )
                step_results.append((todo, result))

        # Observe each completed step sequentially (observation updates shared state)
        effort = self._get_reasoning_effort()
        observer = self._ensure_planner_observer()

        for todo, _result in step_results:
            all_completed = self.state.todos.get_completed_todos()
            remaining = self.state.todos.get_pending_todos()

            observation = observer.observe(
                completed_step=todo,
                result=todo.result or "",
                all_completed=all_completed,
                remaining_todos=remaining,
            )

            self.state.observations[todo.step_number] = observation

            self.state.execution_log.append(
                {
                    "type": "observation",
                    "step_number": todo.step_number,
                    "step_completed_successfully": observation.step_completed_successfully,
                    "key_information_learned": observation.key_information_learned,
                    "remaining_plan_still_valid": observation.remaining_plan_still_valid,
                    "needs_full_replan": observation.needs_full_replan,
                    "goal_already_achieved": observation.goal_already_achieved,
                    "reasoning_effort": effort,
                }
            )

            # Mark based on observation result
            if observation.step_completed_successfully:
                self.state.todos.mark_completed(todo.step_number, result=todo.result)
            else:
                self.state.todos.mark_failed(todo.step_number, result=todo.result)

            if self.agent.verbose:
                self._printer.print(
                    content=(
                        f"[Observe] Step {todo.step_number} "
                        f"(effort={effort}): "
                        f"success={observation.step_completed_successfully}, "
                        f"plan_valid={observation.remaining_plan_still_valid}, "
                        f"learned={observation.key_information_learned[:80]}..."
                    ),
                    color="cyan",
                )

        return "parallel_todos_complete"

    @router("parallel_todos_complete")
    def after_parallel_execution(
        self,
    ) -> Literal["has_todos", "all_todos_complete", "needs_replan"]:
        """Check for more todos after parallel execution completes.

        Also checks if replanning is needed based on execution results.
        """
        # Check if replanning is needed before continuing
        should_replan, reason = self._should_replan()
        if should_replan:
            self.state.last_replan_reason = reason
            return "needs_replan"

        if self.state.todos.is_complete:
            return "all_todos_complete"
        return "has_todos"

    @router(or_("todo_injected", "no_todos", "planning_disabled"))
    def initialize_reasoning(self) -> Literal["initialized"]:
        """Initialize the reasoning flow and emit agent start logs.

        This is called either after todo context is injected, or when
        there are no todos (falling back to standard execution).
        """
        self._show_start_logs()
        # Check for native tool support on first iteration
        if self.state.iterations == 0:
            self.state.use_native_tools = self._check_native_tool_support()
            if self.state.use_native_tools:
                self._setup_native_tools()
        return "initialized"

    @router("force_final_answer")
    def force_final_answer(self) -> Literal["agent_finished"]:
        """Force agent to provide final answer when max iterations exceeded."""
        formatted_answer = handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=self._printer,
            i18n=self._i18n,
            messages=list(self.state.messages),
            llm=self.llm,
            callbacks=self.callbacks,
            verbose=self.agent.verbose,
        )

        self.state.current_answer = formatted_answer
        self.state.is_finished = True

        return "agent_finished"

    @router("continue_reasoning")
    def call_llm_and_parse(self) -> Literal["parsed", "parser_error", "context_error"]:
        """Execute LLM call with hooks and parse the response.

        Returns routing decision based on parsing result.
        """
        if self.state.is_finished:
            return "parsed"

        try:
            enforce_rpm_limit(self.request_within_rpm_limit)

            answer = get_llm_response(
                llm=self.llm,
                messages=list(self.state.messages),
                callbacks=self.callbacks,
                printer=self._printer,
                from_task=self.task,
                from_agent=self.agent,
                response_model=self.response_model,
                executor_context=self,
                verbose=self.agent.verbose,
            )

            # If response is structured output (BaseModel), store it directly
            if isinstance(answer, BaseModel):
                self.state.current_answer = AgentFinish(
                    thought="",
                    output=answer,
                    text=str(answer),
                )
                return "parsed"

            # Parse the LLM response
            formatted_answer = process_llm_response(answer, self.use_stop_words)

            self.state.current_answer = formatted_answer

            if "Final Answer:" in answer and isinstance(formatted_answer, AgentAction):
                warning_text = Text()
                warning_text.append("⚠️ ", style="yellow bold")
                warning_text.append(
                    f"LLM returned 'Final Answer:' but parsed as AgentAction (tool: {formatted_answer.tool})",
                    style="yellow",
                )
                self._console.print(warning_text)
                preview_text = Text()
                preview_text.append("Answer preview: ", style="yellow")
                preview_text.append(f"{answer[:200]}...", style="yellow dim")
                self._console.print(preview_text)

            return "parsed"

        except OutputParserError as e:
            # Store error context for recovery
            self._last_parser_error = e or OutputParserError(
                error="Unknown parser error"
            )
            return "parser_error"

        except Exception as e:
            if is_context_length_exceeded(e):
                self._last_context_error = e
                return "context_error"
            if e.__class__.__module__.startswith("litellm"):
                raise e
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise

    @router("continue_reasoning_native")
    def call_llm_native_tools(
        self,
    ) -> Literal[
        "native_tool_calls", "native_finished", "context_error", "todo_satisfied"
    ]:
        """Execute LLM call with native function calling.

        Always calls the LLM so it can read reflection prompts and decide
        whether to provide a final answer or request more tools.

        When todos are active and the LLM produces a final answer, we treat it
        as completing the current todo rather than finishing the entire task.

        Returns routing decision based on whether tool calls or final answer.
        """
        if self.state.is_finished:
            return "native_finished"

        try:
            # Clear pending tools - LLM will decide what to do next after reading
            # the reflection prompt. It can either:
            # 1. Return a final answer (string) if it has enough info
            # 2. Return tool calls (possibly same ones, or different ones)
            self.state.pending_tool_calls.clear()

            enforce_rpm_limit(self.request_within_rpm_limit)

            # Call LLM with native tools
            answer = get_llm_response(
                llm=self.llm,
                messages=list(self.state.messages),
                callbacks=self.callbacks,
                printer=self._printer,
                tools=self._openai_tools,
                available_functions=None,
                from_task=self.task,
                from_agent=self.agent,
                response_model=self.response_model,
                executor_context=self,
                verbose=self.agent.verbose,
            )

            # Check if the response is a list of tool calls
            if isinstance(answer, list) and answer and self._is_tool_call_list(answer):
                # Store tool calls for sequential processing
                self.state.pending_tool_calls = list(answer)
                return "native_tool_calls"

            if isinstance(answer, BaseModel):
                self.state.current_answer = AgentFinish(
                    thought="",
                    output=answer,
                    text=answer.model_dump_json(),
                )
                self._invoke_step_callback(self.state.current_answer)
                self._append_message_to_state(answer.model_dump_json())
                return self._route_finish_with_todos("native_finished")

            # Text response - this is the final answer
            if isinstance(answer, str):
                self.state.current_answer = AgentFinish(
                    thought="",
                    output=answer,
                    text=answer,
                )
                self._invoke_step_callback(self.state.current_answer)
                self._append_message_to_state(answer)

                return self._route_finish_with_todos("native_finished")

            # Unexpected response type, treat as final answer
            self.state.current_answer = AgentFinish(
                thought="",
                output=str(answer),
                text=str(answer),
            )
            self._invoke_step_callback(self.state.current_answer)
            self._append_message_to_state(str(answer))

            return self._route_finish_with_todos("native_finished")

        except Exception as e:
            if is_context_length_exceeded(e):
                self._last_context_error = e
                return "context_error"
            if e.__class__.__module__.startswith("litellm"):
                raise e
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise

    def _route_finish_with_todos(
        self, default_route: str
    ) -> Literal["native_finished", "agent_finished", "todo_satisfied"]:
        """Helper to route finish events, checking for pending todos first.

        If there are pending todos, route to todo_satisfied instead of the
        default finish event to continue processing todos.

        Args:
            default_route: The default route to use if no todos are pending.

        Returns:
            "todo_satisfied" if todos need processing, otherwise the default route.
        """
        if self.state.todos.items and not self.state.todos.is_complete:
            current_todo = self.state.todos.current_todo
            if current_todo:
                return "todo_satisfied"
        return default_route  # type: ignore[return-value]

    @router(call_llm_and_parse)
    def route_by_answer_type(
        self,
    ) -> Literal["execute_tool", "agent_finished", "todo_satisfied"]:
        """Route based on whether answer is AgentAction or AgentFinish.

        When todos are active and the LLM produces a final answer, we treat it
        as completing the current todo rather than finishing the entire task.
        """
        if isinstance(self.state.current_answer, AgentAction):
            return "execute_tool"

        return self._route_finish_with_todos("agent_finished")

    @router("execute_tool")
    def execute_tool_action(self) -> Literal["tool_completed", "tool_result_is_final"]:
        """Execute the tool action and handle the result."""

        action = cast(AgentAction, self.state.current_answer)

        fingerprint_context = {}
        if (
            self.agent
            and hasattr(self.agent, "security_config")
            and hasattr(self.agent.security_config, "fingerprint")
        ):
            fingerprint_context = {
                "agent_fingerprint": str(self.agent.security_config.fingerprint)
            }

        try:
            tool_result = execute_tool_and_check_finality(
                agent_action=action,
                fingerprint_context=fingerprint_context,
                tools=self.tools,
                i18n=self._i18n,
                agent_key=self.agent.key if self.agent else None,
                agent_role=self.agent.role if self.agent else None,
                tools_handler=self.tools_handler,
                task=self.task,
                agent=self.agent,
                function_calling_llm=self.function_calling_llm,
                crew=self.crew,
            )
        except Exception as e:
            if self.agent and self.agent.verbose:
                self._printer.print(
                    content=f"Error in tool execution: {e}", color="red"
                )
            if self.task:
                self.task.increment_tools_errors()

            error_observation = f"\nObservation: Error executing tool: {e}"
            action.text += error_observation
            action.result = str(e)
            self._append_message_to_state(action.text)

            reasoning_prompt = self._i18n.slice("post_tool_reasoning")
            reasoning_message: LLMMessage = {
                "role": "user",
                "content": reasoning_prompt,
            }
            self.state.messages.append(reasoning_message)

            return "tool_completed"

        result = self._handle_agent_action(action, tool_result)
        self.state.current_answer = result

        self._invoke_step_callback(result)

        if hasattr(result, "text"):
            self._append_message_to_state(result.text)

        if isinstance(result, AgentFinish):
            self.state.is_finished = True
            return "tool_result_is_final"

        reasoning_prompt = self._i18n.slice("post_tool_reasoning")
        reasoning_message_post: LLMMessage = {
            "role": "user",
            "content": reasoning_prompt,
        }
        self.state.messages.append(reasoning_message_post)

        return "tool_completed"

    @router("native_tool_calls")
    def execute_native_tool(
        self,
    ) -> Literal["native_tool_completed", "tool_result_is_final"]:
        """Execute native tool calls in a batch.

        Processes all tools from pending_tool_calls, executes them,
        and appends results to the conversation history.

        Returns:
            "native_tool_completed" normally, or "tool_result_is_final" if
            a tool with result_as_answer=True was executed.
        """
        if not self.state.pending_tool_calls:
            return "native_tool_completed"

        pending_tool_calls = list(self.state.pending_tool_calls)
        self.state.pending_tool_calls.clear()

        # Group all tool calls into a single assistant message
        tool_calls_to_report = []
        for tool_call in pending_tool_calls:
            info = extract_tool_call_info(tool_call)
            if not info:
                continue

            call_id, func_name, func_args = info
            tool_calls_to_report.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": func_args
                        if isinstance(func_args, str)
                        else json.dumps(func_args),
                    },
                }
            )

        if tool_calls_to_report:
            assistant_message: LLMMessage = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_to_report,
            }
            if all(type(tc).__qualname__ == "Part" for tc in pending_tool_calls):
                assistant_message["raw_tool_call_parts"] = list(pending_tool_calls)
            self.state.messages.append(assistant_message)

        runnable_tool_calls = [
            tool_call
            for tool_call in pending_tool_calls
            if extract_tool_call_info(tool_call) is not None
        ]
        should_parallelize = self._should_parallelize_native_tool_calls(
            runnable_tool_calls
        )

        execution_results: list[dict[str, Any]] = []
        if should_parallelize:
            max_workers = min(8, len(runnable_tool_calls))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        contextvars.copy_context().run,
                        self._execute_single_native_tool_call,
                        tool_call,
                    ): idx
                    for idx, tool_call in enumerate(runnable_tool_calls)
                }
                ordered_results: list[dict[str, Any] | None] = [None] * len(
                    runnable_tool_calls
                )
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        ordered_results[idx] = future.result()
                    except Exception as e:
                        tool_call = runnable_tool_calls[idx]
                        info = extract_tool_call_info(tool_call)
                        call_id = info[0] if info else "unknown"
                        func_name = info[1] if info else "unknown"
                        ordered_results[idx] = {
                            "call_id": call_id,
                            "func_name": func_name,
                            "result": f"Error executing tool: {e}",
                            "from_cache": False,
                            "original_tool": None,
                        }
                execution_results = [
                    result for result in ordered_results if result is not None
                ]
        else:
            # Execute sequentially so result_as_answer tools can short-circuit
            # immediately without running remaining calls.
            for tool_call in runnable_tool_calls:
                execution_result = self._execute_single_native_tool_call(tool_call)
                call_id = cast(str, execution_result["call_id"])
                func_name = cast(str, execution_result["func_name"])
                result = cast(str, execution_result["result"])
                from_cache = cast(bool, execution_result["from_cache"])
                original_tool = execution_result["original_tool"]

                tool_message: LLMMessage = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": result,
                }
                self.state.messages.append(tool_message)

                # Log the tool execution
                if self.agent and self.agent.verbose:
                    cache_info = " (from cache)" if from_cache else ""
                    self._printer.print(
                        content=f"Tool {func_name} executed with result{cache_info}: {result[:200]}...",
                        color="green",
                    )

                if (
                    original_tool
                    and hasattr(original_tool, "result_as_answer")
                    and original_tool.result_as_answer
                ):
                    self.state.current_answer = AgentFinish(
                        thought="Tool result is the final answer",
                        output=result,
                        text=result,
                    )
                    self.state.is_finished = True
                    return "tool_result_is_final"

            return "native_tool_completed"

        for execution_result in execution_results:
            call_id = cast(str, execution_result["call_id"])
            func_name = cast(str, execution_result["func_name"])
            result = cast(str, execution_result["result"])
            from_cache = cast(bool, execution_result["from_cache"])
            original_tool = execution_result["original_tool"]

            tool_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": result,
            }
            self.state.messages.append(tool_message)

            # Log the tool execution
            if self.agent and self.agent.verbose:
                cache_info = " (from cache)" if from_cache else ""
                self._printer.print(
                    content=f"Tool {func_name} executed with result{cache_info}: {result[:200]}...",
                    color="green",
                )

            if (
                original_tool
                and hasattr(original_tool, "result_as_answer")
                and original_tool.result_as_answer
            ):
                # Set the result as the final answer
                self.state.current_answer = AgentFinish(
                    thought="Tool result is the final answer",
                    output=result,
                    text=result,
                )
                self.state.is_finished = True
                return "tool_result_is_final"

        return "native_tool_completed"

    def _should_parallelize_native_tool_calls(self, tool_calls: list[Any]) -> bool:
        """Determine if native tool calls are safe to run in parallel."""
        if len(tool_calls) <= 1:
            return False

        for tool_call in tool_calls:
            info = extract_tool_call_info(tool_call)
            if not info:
                continue
            _, func_name, _ = info

            mapping = getattr(self, "_tool_name_mapping", None)
            original_tool: BaseTool | None = None
            if mapping and func_name in mapping:
                mapped = mapping[func_name]
                if isinstance(mapped, BaseTool):
                    original_tool = mapped
            if original_tool is None:
                for tool in self.original_tools or []:
                    if sanitize_tool_name(tool.name) == func_name:
                        original_tool = tool
                        break

            if not original_tool:
                continue

            if getattr(original_tool, "result_as_answer", False):
                return False
            if getattr(original_tool, "max_usage_count", None) is not None:
                return False

        return True

    def _execute_single_native_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Execute a single native tool call and return metadata/result."""
        info = extract_tool_call_info(tool_call)
        if not info:
            call_id = (
                getattr(tool_call, "id", None)
                or (tool_call.get("id") if isinstance(tool_call, dict) else None)
                or "unknown"
            )
            return {
                "call_id": call_id,
                "func_name": "unknown",
                "result": "Error: Invalid native tool call format",
                "from_cache": False,
                "original_tool": None,
            }

        call_id, func_name, func_args = info

        # Parse arguments
        parsed_args, parse_error = parse_tool_call_args(func_args, func_name, call_id)
        if parse_error is not None:
            return parse_error
        args_dict: dict[str, Any] = parsed_args or {}

        # Get agent_key for event tracking
        agent_key = getattr(self.agent, "key", "unknown") if self.agent else "unknown"

        original_tool: BaseTool | None = None
        mapping = getattr(self, "_tool_name_mapping", None)
        if mapping and func_name in mapping:
            mapped = mapping[func_name]
            if isinstance(mapped, BaseTool):
                original_tool = mapped
        if original_tool is None:
            for tool in self.original_tools or []:
                if sanitize_tool_name(tool.name) == func_name:
                    original_tool = tool
                    break

        # Check if tool has reached max usage count
        max_usage_reached = False
        if (
            original_tool
            and original_tool.max_usage_count is not None
            and original_tool.current_usage_count >= original_tool.max_usage_count
        ):
            max_usage_reached = True

        # Check cache before executing
        from_cache = False
        input_str = json.dumps(args_dict) if args_dict else ""
        if self.tools_handler and self.tools_handler.cache:
            cached_result = self.tools_handler.cache.read(
                tool=func_name, input=input_str
            )
            if cached_result is not None:
                result = (
                    str(cached_result)
                    if not isinstance(cached_result, str)
                    else cached_result
                )
                from_cache = True

        # Emit tool usage started event
        started_at = datetime.now()
        crewai_event_bus.emit(
            self,
            event=ToolUsageStartedEvent(
                tool_name=func_name,
                tool_args=args_dict,
                from_agent=self.agent,
                from_task=self.task,
                agent_key=agent_key,
            ),
        )
        error_event_emitted = False

        track_delegation_if_needed(func_name, args_dict, self.task)

        structured_tool: CrewStructuredTool | None = None
        if original_tool is not None:
            for structured in self.tools or []:
                if getattr(structured, "_original_tool", None) is original_tool:
                    structured_tool = structured
                    break
        if structured_tool is None:
            for structured in self.tools or []:
                if sanitize_tool_name(structured.name) == func_name:
                    structured_tool = structured
                    break

        hook_blocked = False
        before_hook_context = ToolCallHookContext(
            tool_name=func_name,
            tool_input=args_dict,
            tool=structured_tool,  # type: ignore[arg-type]
            agent=self.agent,
            task=self.task,
            crew=self.crew,
        )
        before_hooks = get_before_tool_call_hooks()
        try:
            for hook in before_hooks:
                hook_result = hook(before_hook_context)
                if hook_result is False:
                    hook_blocked = True
                    break
        except Exception as hook_error:
            if self.agent.verbose:
                self._printer.print(
                    content=f"Error in before_tool_call hook: {hook_error}",
                    color="red",
                )

        if hook_blocked:
            result = f"Tool execution blocked by hook. Tool: {func_name}"
        elif not from_cache and not max_usage_reached:
            result = "Tool not found"
            if func_name in self._available_functions:
                try:
                    tool_func = self._available_functions[func_name]
                    raw_result = tool_func(**args_dict)

                    # Add to cache after successful execution (before string conversion)
                    if self.tools_handler and self.tools_handler.cache:
                        should_cache = True
                        if original_tool:
                            should_cache = original_tool.cache_function(
                                args_dict, raw_result
                            )
                        if should_cache:
                            self.tools_handler.cache.add(
                                tool=func_name, input=input_str, output=raw_result
                            )

                    # Convert to string for message
                    result = (
                        str(raw_result)
                        if not isinstance(raw_result, str)
                        else raw_result
                    )
                except Exception as e:
                    result = f"Error executing tool: {e}"
                    if self.task:
                        self.task.increment_tools_errors()
                    # Emit tool usage error event
                    crewai_event_bus.emit(
                        self,
                        event=ToolUsageErrorEvent(
                            tool_name=func_name,
                            tool_args=args_dict,
                            from_agent=self.agent,
                            from_task=self.task,
                            agent_key=agent_key,
                            error=e,
                        ),
                    )
                    error_event_emitted = True
        elif max_usage_reached:
            # Return error message when max usage limit is reached
            if original_tool:
                result = f"Tool '{func_name}' has reached its usage limit of {original_tool.max_usage_count} times and cannot be used anymore."
            else:
                result = f"Tool '{func_name}' has reached its maximum usage limit and cannot be used anymore."

        # Execute after_tool_call hooks (even if blocked, to allow logging/monitoring)
        after_hook_context = ToolCallHookContext(
            tool_name=func_name,
            tool_input=args_dict,
            tool=structured_tool,  # type: ignore[arg-type]
            agent=self.agent,
            task=self.task,
            crew=self.crew,
            tool_result=result,
        )
        after_hooks = get_after_tool_call_hooks()
        try:
            for after_hook in after_hooks:
                after_hook_result = after_hook(after_hook_context)
                if after_hook_result is not None:
                    result = after_hook_result
                    after_hook_context.tool_result = result
        except Exception as hook_error:
            if self.agent.verbose:
                self._printer.print(
                    content=f"Error in after_tool_call hook: {hook_error}",
                    color="red",
                )

        if not error_event_emitted:
            crewai_event_bus.emit(
                self,
                event=ToolUsageFinishedEvent(
                    output=result,
                    tool_name=func_name,
                    tool_args=args_dict,
                    from_agent=self.agent,
                    from_task=self.task,
                    agent_key=agent_key,
                    started_at=started_at,
                    finished_at=datetime.now(),
                ),
            )

        return {
            "call_id": call_id,
            "func_name": func_name,
            "result": result,
            "from_cache": from_cache,
            "original_tool": original_tool,
        }

    def _extract_tool_name(self, tool_call: Any) -> str:
        """Extract tool name from various tool call formats."""
        if hasattr(tool_call, "function"):
            return sanitize_tool_name(tool_call.function.name)
        if hasattr(tool_call, "function_call") and tool_call.function_call:
            return sanitize_tool_name(tool_call.function_call.name)
        if hasattr(tool_call, "name"):
            return sanitize_tool_name(tool_call.name)
        if isinstance(tool_call, dict):
            func_info = tool_call.get("function", {})
            return sanitize_tool_name(
                func_info.get("name", "") or tool_call.get("name", "unknown")
            )
        return "unknown"

    @router(execute_native_tool)
    def check_native_todo_completion(
        self,
    ) -> Literal["todo_satisfied", "todo_not_satisfied"]:
        """Check if the native tool execution satisfied the active todo.

        Similar to check_todo_completion but for native tool execution path.
        """
        current_todo = self.state.todos.current_todo

        if not current_todo:
            return "todo_not_satisfied"

        # For native tools, any tool execution satisfies the todo
        return "todo_satisfied"

    @listen("initialized")
    def continue_iteration(self) -> Literal["check_iteration"]:
        """Bridge listener that connects iteration loop back to iteration check."""
        if self._flow_initialized:
            self._discard_or_listener(FlowMethodName("continue_iteration"))
        return "check_iteration"

    @router(or_(initialize_reasoning, continue_iteration))
    def check_max_iterations(
        self,
    ) -> Literal[
        "force_final_answer", "continue_reasoning", "continue_reasoning_native"
    ]:
        """Check if max iterations reached before proceeding with reasoning."""
        if has_reached_max_iterations(self.state.iterations, self.max_iter):
            return "force_final_answer"
        if self.state.use_native_tools:
            return "continue_reasoning_native"
        return "continue_reasoning"

    @router(execute_tool_action)
    def check_todo_completion(
        self,
    ) -> Literal["todo_satisfied", "todo_not_satisfied"]:
        """Check if the current tool execution satisfied the active todo.

        After a tool is executed, this determines if the current todo
        should be marked as complete based on whether:
        1. The expected tool was used (if specified)
        2. The agent returned a final answer for this step
        """
        current_todo = self.state.todos.current_todo

        if not current_todo:
            return "todo_not_satisfied"

        action = self.state.current_answer

        # Check if the expected tool was used
        if isinstance(action, AgentAction):
            if current_todo.tool_to_use:
                if action.tool == current_todo.tool_to_use:
                    return "todo_satisfied"
            else:
                return "todo_satisfied"

        if isinstance(action, AgentFinish):
            return "todo_satisfied"

        return "todo_not_satisfied"

    @listen("todo_satisfied")
    def mark_todo_complete(self) -> Literal["todo_marked"]:
        """Mark the current todo as completed with its result."""
        current_todo = self.state.todos.current_todo

        if not current_todo:
            return "todo_marked"

        # Extract result from the current answer
        result = ""
        if isinstance(self.state.current_answer, AgentFinish):
            result = str(self.state.current_answer.output)
        elif isinstance(self.state.current_answer, AgentAction):
            # Use the tool result (last message should have it)
            if self.state.messages:
                last_msg = self.state.messages[-1]
                if (
                    last_msg.get("role") == "tool"
                    or last_msg.get("role") == "assistant"
                ):
                    result = str(last_msg.get("content", ""))
        elif not self.state.current_answer and self.state.messages:
            # For native tools, results are in the message history as 'tool' roles
            # We take the content of the most recent tool results
            tool_results = []
            for msg in reversed(self.state.messages):
                if msg.get("role") == "tool":
                    tool_results.insert(0, str(msg.get("content", "")))
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Once we hit the assistant message that triggered the tools, we stop
                    break
            result = "\n".join(tool_results)

        self._mark_todo_as_completed(current_todo.step_number, result)

        return "todo_marked"

    def _mark_todo_as_completed(self, step_number: int, result: str) -> None:
        """Helper to mark a todo as completed and update history.

        Args:
            step_number: The step number to mark.
            result: The result of the todo.
        """
        self.state.todos.mark_completed(step_number, result=result)

        if self.agent.verbose:
            completed = self.state.todos.completed_count
            total = len(self.state.todos.items)
            self._printer.print(
                content=f"✓ Todo {step_number} completed ({completed}/{total})",
                color="green",
            )

        # Add to history as a SYSTEM message for subsequent steps
        if result:
            self._append_message_to_state(
                f"**Step {step_number} result:**\n\n{result}",
                role="system",
            )

    @router(mark_todo_complete)
    def check_more_todos(
        self,
    ) -> Literal["has_todos", "all_todos_complete", "needs_replan"]:
        """Check if there are more todos to execute after marking one complete.

        Also checks if replanning is needed based on execution results.
        """
        # Check if replanning is needed before continuing
        should_replan, reason = self._should_replan()
        if should_replan:
            self.state.last_replan_reason = reason
            return "needs_replan"

        if self.state.todos.is_complete:
            return "all_todos_complete"

        return "has_todos"

    @router("todo_not_satisfied")
    def increment_and_continue(self) -> Literal["initialized"]:
        """Increment iteration counter and loop back for next iteration.

        Called when a tool execution didn't satisfy the current todo,
        allowing the agent to continue working on it.
        """
        self.state.iterations += 1
        return "initialized"

    @listen(
        or_(
            "all_todos_complete",
            "agent_finished",
            "tool_result_is_final",
            "native_finished",
        )
    )
    def finalize(self) -> Literal["completed", "skipped"]:
        """Finalize execution and emit completion logs.

        If todos were used, synthesizes a final answer from all todo results.
        Handles both the legacy ReAct path (current_answer already set) and
        the Plan-and-Execute path (synthesize from completed todos).
        """
        # Guard against duplicate finalization — the flow may trigger finalize
        # more than once when concurrent branches both reach a terminal state.
        # Use a lock to atomically check-and-set _finalize_called so only the
        # first caller proceeds.  We use a separate flag (not is_finished)
        # because is_finished should only be set when finalization succeeds.
        with self._finalize_lock:
            if self._finalize_called:
                return "completed"
            self._finalize_called = True

        if self.agent.verbose:
            self._printer.print(
                content=f"[Finalize] todos_count={len(self.state.todos.items)}, todos_with_results={sum(1 for t in self.state.todos.items if t.result)}",
                color="magenta",
            )

        if self.state.current_answer is None:
            # Plan-and-Execute path: todos may have results even if not all are
            # marked "completed" (e.g., goal_achieved early).
            todos_with_results = [t for t in self.state.todos.items if t.result]
            if todos_with_results:
                if self._can_use_last_todo_result_as_final_answer(todos_with_results):
                    last_todo = max(
                        todos_with_results, key=lambda todo: todo.step_number
                    )
                    final_text = str(last_todo.result or "")
                    self.state.current_answer = AgentFinish(
                        thought="Final answer returned directly from last completed todo",
                        output=final_text,
                        text=final_text,
                    )
                else:
                    self._synthesize_final_answer_from_todos()

        if self.state.current_answer is None:
            # Last resort: produce a fallback answer rather than leaving
            # current_answer as None, which causes a RuntimeError upstream.
            fallback_text = "Agent completed execution but produced no final output."
            if self.state.todos.items:
                partial = [
                    f"Step {t.step_number}: {t.result or '(no result)'}"
                    for t in self.state.todos.items
                    if t.status == "completed"
                ]
                if partial:
                    fallback_text = "\n\n".join(partial)
            self.state.current_answer = AgentFinish(
                thought="Finalize fallback — no explicit answer was set",
                output=fallback_text,
                text=fallback_text,
            )

        if not isinstance(self.state.current_answer, AgentFinish):
            skip_text = Text()
            skip_text.append("⚠️ ", style="yellow bold")
            skip_text.append(
                f"Finalize called with {type(self.state.current_answer).__name__} instead of AgentFinish - skipping",
                style="yellow",
            )
            self._console.print(skip_text)
            return "skipped"

        self.state.is_finished = True
        self._show_logs(self.state.current_answer)

        return "completed"

    def _can_use_last_todo_result_as_final_answer(
        self, todos_with_results: list[TodoItem]
    ) -> bool:
        """Determine whether synthesis can be skipped for planning results."""
        # Keep synthesis when structured output is requested.
        if self.response_model is not None:
            return False
        if not todos_with_results:
            return False

        last_todo = max(todos_with_results, key=lambda todo: todo.step_number)
        if last_todo.tool_to_use:
            return False

        last_result = str(last_todo.result or "").strip()
        if not last_result:
            return False

        lowered_result = last_result.lower()
        if (
            lowered_result.startswith("error:")
            or "tool execution error" in lowered_result
        ):
            return False

        word_count = len(last_result.split())
        has_sentence_punctuation = any(ch in last_result for ch in ".!?")
        return (
            len(last_result) >= 200 or word_count >= 30
        ) and has_sentence_punctuation

    def _synthesize_final_answer_from_todos(self) -> None:
        """Synthesize a coherent final answer from all todo results.

        Makes one LLM call to produce a clean, unified response from
        the accumulated step results, rather than dumping raw step outputs.

        If a response_model is set (from task.response_model or kickoff(response_format)),
        the synthesis call uses it to produce structured output matching the
        expected schema. This is the ONLY place response_model is applied in
        the Plan-and-Execute path — intermediate steps produce free-text results.

        Falls back to concatenation if the synthesis LLM call fails.
        """
        step_results: list[str] = [
            f"Step {todo.step_number} ({todo.description}):\n{todo.result}"
            for todo in self.state.todos.items
            if todo.result
        ]

        if not step_results:
            return

        combined_steps = "\n\n".join(step_results)

        # Get the original task description
        task_description = ""
        if self.task:
            task_description = self.task.description or ""
        else:
            task_description = getattr(self, "_kickoff_input", "")

        # Strip any appended planning text from the task description
        if "\n\nPlanning:\n" in task_description:
            task_description = task_description.split("\n\nPlanning:\n")[0]

        # Build synthesis prompt
        role = self.agent.role if self.agent else "Assistant"

        system_prompt = self._i18n.retrieve(
            "planning", "synthesis_system_prompt"
        ).format(role=role)
        user_prompt = self._i18n.retrieve("planning", "synthesis_user_prompt").format(
            task_description=task_description,
            combined_steps=combined_steps,
        )

        try:
            synthesis = self.llm.call(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=self.response_model,
                from_task=self.task,
                from_agent=self.agent,
            )

            if synthesis:
                # If response_model produced a BaseModel, store it directly
                if isinstance(synthesis, BaseModel):
                    self.state.current_answer = AgentFinish(
                        thought="Synthesized structured final answer from all completed steps",
                        output=synthesis,
                        text=synthesis.model_dump_json(),
                    )
                else:
                    final_text = str(synthesis)
                    self.state.current_answer = AgentFinish(
                        thought="Synthesized final answer from all completed steps",
                        output=final_text,
                        text=final_text,
                    )
                return

        except Exception as e:
            if self.agent and self.agent.verbose:
                self._printer.print(
                    content=f"Synthesis LLM call failed ({e}), falling back to concatenation",
                    color="yellow",
                )

        # Fallback: concatenate step results if synthesis fails
        fallback = "\n\n".join(step_results)
        self.state.current_answer = AgentFinish(
            thought="All planned steps completed (synthesis unavailable)",
            output=fallback,
            text=fallback,
        )

    # -------------------------------------------------------------------------
    # Dynamic Replanning Methods
    # -------------------------------------------------------------------------

    def _should_replan(self) -> tuple[bool, str]:
        """Determine if dynamic replanning is needed.

        Checks for conditions that warrant regenerating the execution plan:
        1. Multiple consecutive todo failures
        2. All todos completed but agent indicates incomplete results
        3. Agent explicitly requested a replan via tool or output

        Returns:
            Tuple of (should_replan: bool, reason: str)
        """
        max_replans = self._get_max_replans()

        # Don't replan if we've hit the limit
        if self.state.replan_count >= max_replans:
            return False, "Max replan attempts reached"

        # Check for failed todos (now actually tracked via "failed" status)
        failed_todos = self.state.todos.get_failed_todos()
        if len(failed_todos) >= 2:
            return True, f"Multiple todos failed ({len(failed_todos)} failures)"

        # Check for todos with error results
        error_todos = [
            todo
            for todo in self.state.todos.items
            if todo.result and todo.result.startswith("Error:")
        ]
        if len(error_todos) >= 2:
            return (
                True,
                f"Multiple todos encountered errors ({len(error_todos)} errors)",
            )

        # Check if agent's last message indicates need for replanning
        if self.state.messages:
            last_msg = self.state.messages[-1]
            content = str(last_msg.get("content", "")).lower()
            replan_indicators = [
                "need to reconsider",
                "approach isn't working",
                "try a different approach",
                "replan",
                "revise the plan",
                "plan needs adjustment",
            ]
            for indicator in replan_indicators:
                if indicator in content:
                    return True, f"Agent indicated replanning needed: '{indicator}'"

        return False, ""

    def _trigger_replan(self, reason: str) -> None:
        """Trigger dynamic replanning with accumulated context.

        Regenerates the execution plan based on what has been learned
        from previous attempts, including failures and partial results.

        NOTE: Callers are responsible for incrementing ``replan_count``
        before calling this method (to allow the guard check in each
        caller's own flow method).

        Args:
            reason: The reason for triggering the replan.
        """
        self.state.last_replan_reason = reason

        if self.agent.verbose:
            self._printer.print(
                content=f"Triggering replan (attempt {self.state.replan_count}): {reason}",
                color="yellow",
            )

        # Build context from previous execution attempts
        previous_context = self._build_replan_context()

        try:
            from crewai.utilities.reasoning_handler import AgentReasoning

            if self.task:
                planning_handler = AgentReasoning(agent=self.agent, task=self.task)
            else:
                input_text = getattr(self, "_kickoff_input", "")
                planning_handler = AgentReasoning(
                    agent=self.agent,
                    description=input_text or "Complete the requested task",
                    expected_output="Complete the task successfully",
                )

            # Include previous context in the planning request
            # This helps the planner learn from past failures
            enhanced_description = self._enhance_task_for_replan(previous_context)
            if self.task:
                original_description = self.task.description
                self.task.description = enhanced_description
                output = planning_handler.handle_agent_reasoning()
                self.task.description = original_description
            else:
                # description is a read-only property — recreate with enhanced text
                input_text = getattr(self, "_kickoff_input", "")
                planning_handler = AgentReasoning(
                    agent=self.agent,
                    description=enhanced_description
                    or input_text
                    or "Complete the requested task",
                    expected_output="Complete the task successfully",
                )
                output = planning_handler.handle_agent_reasoning()

            # Update plan metadata and replace only pending todos,
            # preserving completed history for context and synthesis.
            self.state.plan = output.plan.plan
            self.state.plan_ready = output.plan.ready

            if self.state.plan_ready and output.plan.steps:
                new_todos = [
                    TodoItem(
                        step_number=step.step_number,
                        description=step.description,
                        tool_to_use=step.tool_to_use,
                        depends_on=step.depends_on,
                        status="pending",
                    )
                    for step in output.plan.steps
                ]
                self.state.todos.replace_pending_todos(new_todos)

                if self.agent.verbose:
                    self._printer.print(
                        content=f"Replan: {len(new_todos)} new steps (completed history preserved)",
                        color="green",
                    )

        except Exception as e:
            if hasattr(self.agent, "_logger"):
                self.agent._logger.log("error", f"Error during replanning: {e!s}")
            # Keep existing todos if replanning fails
            self.state.last_replan_reason = f"Replan failed: {e!s}"

    def _build_replan_context(self) -> str:
        """Build context from previous execution for replanning.

        Summarizes what has been attempted, what failed, and what succeeded
        to help the planner create a better plan.

        Returns:
            A context string describing previous execution state.
        """
        context_parts = []

        # Summarize completed todos
        completed = [t for t in self.state.todos.items if t.status == "completed"]
        if completed:
            context_parts.append("Successfully completed steps:")
            for todo in completed:
                context_parts.append(f"  - Step {todo.step_number}: {todo.description}")
                if todo.result:
                    context_parts.append(f"    Result: {todo.result}")

        # Summarize failed todos
        failed = [
            t
            for t in self.state.todos.items
            if t.status == "failed" or (t.result and t.result.startswith("Error:"))
        ]
        if failed:
            context_parts.append("\nFailed or errored steps:")
            for todo in failed:
                context_parts.append(f"  - Step {todo.step_number}: {todo.description}")
                if todo.result:
                    context_parts.append(f"    Error: {todo.result}")

        # Add replan history
        if self.state.replan_count > 0:
            context_parts.append(f"\nThis is replan attempt {self.state.replan_count}.")
            if self.state.last_replan_reason:
                context_parts.append(
                    f"Previous replan reason: {self.state.last_replan_reason}"
                )

        return "\n".join(context_parts)

    def _enhance_task_for_replan(self, previous_context: str) -> str:
        """Enhance task description with context for replanning.

        Args:
            previous_context: Context from previous execution attempts.

        Returns:
            Enhanced task description for the planner.
        """
        original = (
            self.task.description if self.task else getattr(self, "_kickoff_input", "")
        )

        enhancement = self._i18n.retrieve(
            "planning", "replan_enhancement_prompt"
        ).format(previous_context=previous_context)

        return f"{original}{enhancement}"

    @router("needs_replan")
    def handle_replan(self) -> Literal["has_todos", "no_todos"]:
        """Handle replanning request and return to todo execution.

        Called when dynamic replanning is triggered. Regenerates the plan
        and routes back to todo-driven execution.
        """
        max_replans = self._get_max_replans()

        if self.state.replan_count >= max_replans:
            if self.agent.verbose:
                self._printer.print(
                    content=f"Max replans ({max_replans}) reached — finalizing with current results",
                    color="yellow",
                )
            return "no_todos"

        self.state.replan_count += 1
        reason = self.state.last_replan_reason or "Dynamic replan triggered"
        self._trigger_replan(reason)

        if self.state.todos.get_pending_todos():
            return "has_todos"
        return "no_todos"

    @router("parser_error")
    def recover_from_parser_error(self) -> Literal["initialized"]:
        """Recover from output parser errors and retry."""
        if not self._last_parser_error:
            self.state.iterations += 1
            return "initialized"

        formatted_answer = handle_output_parser_exception(
            e=self._last_parser_error,
            messages=list(self.state.messages),
            iterations=self.state.iterations,
            log_error_after=self.log_error_after,
            printer=self._printer,
            verbose=self.agent.verbose,
        )

        if formatted_answer:
            self.state.current_answer = formatted_answer

        self.state.iterations += 1

        return "initialized"

    @router("context_error")
    def recover_from_context_length(self) -> Literal["initialized"]:
        """Recover from context length errors and retry."""
        handle_context_length(
            respect_context_window=self.respect_context_window,
            printer=self._printer,
            messages=self.state.messages,
            llm=self.llm,
            callbacks=self.callbacks,
            i18n=self._i18n,
            verbose=self.agent.verbose,
        )

        self.state.iterations += 1

        return "initialized"

    def invoke(
        self, inputs: dict[str, Any]
    ) -> dict[str, Any] | Coroutine[Any, Any, dict[str, Any]]:
        """Execute agent with given inputs.

        When called from within an existing event loop (e.g., inside a Flow),
        this method returns a coroutine that should be awaited. The Flow
        framework handles this automatically.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output, or a coroutine if inside an event loop.
        """
        # Magic auto-async: if inside event loop, return coroutine for Flow to await
        if is_inside_event_loop():
            return self.invoke_async(inputs)

        self._ensure_flow_initialized()

        with self._execution_lock:
            if self._is_executing:
                raise RuntimeError(
                    "Executor is already running. "
                    "Cannot invoke the same executor instance concurrently."
                )
            self._is_executing = True
            self._has_been_invoked = True

        try:
            # Reset state for fresh execution
            self._finalize_called = False
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False
            self.state.use_native_tools = False
            self.state.pending_tool_calls = []
            self.state.plan = None
            self.state.plan_ready = False
            self.state.todos = TodoList()
            self.state.replan_count = 0
            self.state.last_replan_reason = None
            self.state.observations = {}
            self.state.execution_log = []

            self._kickoff_input = inputs.get("input", "")

            if "system" in self.prompt:
                prompt = cast("SystemPromptResult", self.prompt)
                system_prompt = self._format_prompt(prompt["system"], inputs)
                user_prompt = self._format_prompt(prompt["user"], inputs)
                self.state.messages.append(
                    format_message_for_llm(system_prompt, role="system")
                )
                self.state.messages.append(format_message_for_llm(user_prompt))
            else:
                user_prompt = self._format_prompt(self.prompt["prompt"], inputs)
                self.state.messages.append(format_message_for_llm(user_prompt))

            self._inject_files_from_inputs(inputs)

            self.state.ask_for_human_input = bool(
                inputs.get("ask_for_human_input", False)
            )

            self.kickoff()

            formatted_answer = self.state.current_answer

            if not isinstance(formatted_answer, AgentFinish):
                raise RuntimeError(
                    "Agent execution ended without reaching a final answer."
                )

            if self.state.ask_for_human_input:
                formatted_answer = self._handle_human_feedback(formatted_answer)

            self._save_to_memory(formatted_answer)

            return {"output": formatted_answer.output}

        except AssertionError:
            fail_text = Text()
            fail_text.append("❌ ", style="red bold")
            fail_text.append(
                "Agent failed to reach a final answer. This is likely a bug - please report it.",
                style="red",
            )
            self._console.print(fail_text)
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise
        finally:
            self._is_executing = False

    async def invoke_async(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute agent asynchronously with given inputs.

        This method is designed for use within async contexts, such as when
        the agent is called from within an async Flow method. It uses
        kickoff_async() directly instead of running in a separate thread.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        self._ensure_flow_initialized()

        with self._execution_lock:
            if self._is_executing:
                raise RuntimeError(
                    "Executor is already running. "
                    "Cannot invoke the same executor instance concurrently."
                )
            self._is_executing = True
            self._has_been_invoked = True

        try:
            # Reset state for fresh execution
            self._finalize_called = False
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False
            self.state.use_native_tools = False
            self.state.pending_tool_calls = []
            self.state.plan = None
            self.state.plan_ready = False
            self.state.todos = TodoList()
            self.state.replan_count = 0
            self.state.last_replan_reason = None
            self.state.observations = {}
            self.state.execution_log = []

            self._kickoff_input = inputs.get("input", "")

            if "system" in self.prompt:
                prompt = cast("SystemPromptResult", self.prompt)
                system_prompt = self._format_prompt(prompt["system"], inputs)
                user_prompt = self._format_prompt(prompt["user"], inputs)
                self.state.messages.append(
                    format_message_for_llm(system_prompt, role="system")
                )
                self.state.messages.append(format_message_for_llm(user_prompt))
            else:
                user_prompt = self._format_prompt(self.prompt["prompt"], inputs)
                self.state.messages.append(format_message_for_llm(user_prompt))

            self._inject_files_from_inputs(inputs)

            self.state.ask_for_human_input = bool(
                inputs.get("ask_for_human_input", False)
            )

            # Use async kickoff directly since we're already in an async context
            await self.kickoff_async()

            formatted_answer = self.state.current_answer

            if not isinstance(formatted_answer, AgentFinish):
                raise RuntimeError(
                    "Agent execution ended without reaching a final answer."
                )

            if self.state.ask_for_human_input:
                formatted_answer = await self._ahandle_human_feedback(formatted_answer)

            self._save_to_memory(formatted_answer)

            return {"output": formatted_answer.output}

        except AssertionError:
            fail_text = Text()
            fail_text.append("❌ ", style="red bold")
            fail_text.append(
                "Agent failed to reach a final answer. This is likely a bug - please report it.",
                style="red",
            )
            self._console.print(fail_text)
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise
        finally:
            self._is_executing = False

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async version of invoke. Alias for invoke_async."""
        return await self.invoke_async(inputs)

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> AgentAction | AgentFinish:
        """Process agent action and tool execution result.

        Args:
            formatted_answer: Agent's action to execute.
            tool_result: Result from tool execution.

        Returns:
            Updated action or final answer.
        """
        add_image_tool = self._i18n.tools("add_image")
        if (
            isinstance(add_image_tool, dict)
            and formatted_answer.tool.casefold().strip()
            == add_image_tool.get("name", "").casefold().strip()
        ):
            self.state.messages.append(
                {"role": "assistant", "content": tool_result.result}
            )
            return formatted_answer

        return handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            messages=self.state.messages,
            step_callback=self.step_callback,
            show_logs=self._show_logs,
        )

    def _invoke_step_callback(
        self, formatted_answer: AgentAction | AgentFinish
    ) -> None:
        """Invoke step callback if configured.

        Args:
            formatted_answer: Current agent response.
        """
        if self.step_callback:
            cb_result = self.step_callback(formatted_answer)
            if inspect.iscoroutine(cb_result):
                if is_inside_event_loop():
                    callback_task = asyncio.create_task(cb_result)
                    callback_task.add_done_callback(
                        self._handle_step_callback_task_result
                    )
                else:
                    asyncio.run(cb_result)

    def _handle_step_callback_task_result(self, task: asyncio.Task[Any]) -> None:
        """Surface async callback errors without crashing the flow event loop."""
        try:
            task.result()
        except Exception as e:
            if self.agent.verbose:
                self._printer.print(
                    content=f"Error in async step_callback task: {e!s}",
                    color="red",
                )

    def _append_message_to_state(
        self, text: str, role: Literal["user", "assistant", "system"] = "assistant"
    ) -> None:
        """Add message to state conversation history.

        Args:
            text: Message content.
            role: Message role (default: assistant).
        """
        self.state.messages.append(format_message_for_llm(text, role=role))

    def _show_start_logs(self) -> None:
        """Emit agent start event."""
        if self.agent is None:
            raise ValueError("Agent cannot be None")

        if self.task is None:
            return

        crewai_event_bus.emit(
            self.agent,
            AgentLogsStartedEvent(
                agent_role=self.agent.role,
                task_description=self.task.description,
                verbose=self.agent.verbose
                or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
            ),
        )

    def _show_logs(self, formatted_answer: AgentAction | AgentFinish) -> None:
        """Emit agent execution event.

        Args:
            formatted_answer: Agent's response to log.
        """
        if self.agent is None:
            raise ValueError("Agent cannot be None")

        crewai_event_bus.emit(
            self.agent,
            AgentLogsExecutionEvent(
                agent_role=self.agent.role,
                formatted_answer=formatted_answer,
                verbose=self.agent.verbose
                or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
            ),
        )

    def _handle_crew_training_output(
        self, result: AgentFinish, human_feedback: str | None = None
    ) -> None:
        """Save training data for crew training mode.

        Args:
            result: Agent's final output.
            human_feedback: Optional feedback from human.
        """
        # Early return if no crew (standalone mode)
        if self.crew is None:
            return

        agent_id = str(self.agent.id)
        train_iteration = getattr(self.crew, "_train_iteration", None)

        if train_iteration is None or not isinstance(train_iteration, int):
            train_error = Text()
            train_error.append("❌ ", style="red bold")
            train_error.append(
                "Invalid or missing train iteration. Cannot save training data.",
                style="red",
            )
            self._console.print(train_error)
            return

        training_handler = CrewTrainingHandler(TRAINING_DATA_FILE)
        training_data = training_handler.load() or {}

        # Initialize or retrieve agent's training data
        agent_training_data = training_data.get(agent_id, {})

        if human_feedback is not None:
            # Save initial output and human feedback
            agent_training_data[train_iteration] = {
                "initial_output": result.output,
                "human_feedback": human_feedback,
            }
        else:
            # Save improved output
            if train_iteration in agent_training_data:
                agent_training_data[train_iteration]["improved_output"] = result.output
            else:
                train_error = Text()
                train_error.append("❌ ", style="red bold")
                train_error.append(
                    f"No existing training data for agent {agent_id} and iteration "
                    f"{train_iteration}. Cannot save improved output.",
                    style="red",
                )
                self._console.print(train_error)
                return

        # Update the training data and save
        training_data[agent_id] = agent_training_data
        training_handler.save(training_data)

    def _inject_files_from_inputs(self, inputs: dict[str, Any]) -> None:
        """Inject files from inputs into the last user message.

        Args:
            inputs: Input dictionary that may contain a 'files' key.
        """
        files = inputs.get("files")
        if not files:
            return

        for i in range(len(self.state.messages) - 1, -1, -1):
            msg = self.state.messages[i]
            if msg.get("role") == "user":
                msg["files"] = files
                break

    @staticmethod
    def _format_prompt(prompt: str, inputs: dict[str, str]) -> str:
        """Format prompt template with input values.

        Args:
            prompt: Template string.
            inputs: Values to substitute.

        Returns:
            Formatted prompt.
        """
        prompt = prompt.replace("{input}", inputs["input"])
        prompt = prompt.replace("{tool_names}", inputs["tool_names"])
        return prompt.replace("{tools}", inputs["tools"])

    def _handle_human_feedback(self, formatted_answer: AgentFinish) -> AgentFinish:
        """Process human feedback and refine answer.

        Args:
            formatted_answer: Initial agent result.

        Returns:
            Final answer after feedback.
        """
        provider = get_provider()
        return provider.handle_feedback(formatted_answer, self)

    async def _ahandle_human_feedback(
        self, formatted_answer: AgentFinish
    ) -> AgentFinish:
        """Process human feedback asynchronously and refine answer.

        Args:
            formatted_answer: Initial agent result.

        Returns:
            Final answer after feedback.
        """
        provider = get_provider()
        return await provider.handle_feedback_async(formatted_answer, self)

    def _is_training_mode(self) -> bool:
        """Check if training mode is active.

        Returns:
            True if in training mode.
        """
        return bool(self.crew and self.crew._train)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for Protocol compatibility.

        Allows the executor to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()


# Backward compatibility alias (deprecated)
CrewAgentExecutorFlow = AgentExecutor
