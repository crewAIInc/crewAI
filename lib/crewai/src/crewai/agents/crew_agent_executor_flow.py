"""Flow-based agent executor for crew AI agents.

Implements the ReAct pattern using Flow's event-driven architecture
as an alternative to the imperative while-loop pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.hooks.llm_hooks import (
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
)
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    process_llm_response,
)
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.printer import Printer
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.training_handler import CrewTrainingHandler
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.tools.tool_types import ToolResult
    from crewai.utilities.prompts import StandardPromptResult, SystemPromptResult


class AgentReActState(BaseModel):
    """Structured state for agent ReAct flow execution.

    Replaces scattered instance variables with validated immutable state.
    Maps to: self.messages, self.iterations, formatted_answer in current executor.
    """

    messages: list[LLMMessage] = Field(default_factory=list)
    iterations: int = Field(default=0)
    current_answer: AgentAction | AgentFinish | None = Field(default=None)
    is_finished: bool = Field(default=False)
    ask_for_human_input: bool = Field(default=False)


class CrewAgentExecutorFlow(Flow[AgentReActState], CrewAgentExecutorMixin):
    """Flow-based executor matching CrewAgentExecutor interface.

    Inherits from:
    - Flow[AgentReActState]: Provides flow orchestration capabilities
    - CrewAgentExecutorMixin: Provides memory methods (short/long/external term)

    Note: Multiple instances may be created during agent initialization
    (cache setup, RPM controller setup, etc.) but only the final instance
    should execute tasks via invoke().
    """

    def __init__(
        self,
        llm: BaseLLM,
        task: Task,
        crew: Crew,
        agent: Agent,
        prompt: SystemPromptResult | StandardPromptResult,
        max_iter: int,
        tools: list[CrewStructuredTool],
        tools_names: str,
        stop_words: list[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: list[BaseTool] | None = None,
        function_calling_llm: BaseLLM | Any | None = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Callable[[], bool] | None = None,
        callbacks: list[Any] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> None:
        """Initialize with same signature as CrewAgentExecutor.

        Reference: crew_agent_executor.py lines 70-150

        Args:
            llm: Language model instance.
            task: Task to execute.
            crew: Crew instance.
            agent: Agent to execute.
            prompt: Prompt templates.
            max_iter: Maximum iterations.
            tools: Available tools.
            tools_names: Tool names string.
            stop_words: Stop word list.
            tools_description: Tool descriptions.
            tools_handler: Tool handler instance.
            step_callback: Optional step callback.
            original_tools: Original tool list.
            function_calling_llm: Optional function calling LLM.
            respect_context_window: Respect context limits.
            request_within_rpm_limit: RPM limit check function.
            callbacks: Optional callbacks list.
            response_model: Optional Pydantic model for structured outputs.
        """
        # Store all parameters as instance variables BEFORE calling super().__init__()
        # This is required because Flow.__init__ calls getattr() which may trigger
        # @property decorators that reference these attributes
        self._i18n: I18N = get_i18n()
        self.llm = llm
        self.task = task
        self.agent = agent
        self.crew = crew
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

        # Error context storage for recovery
        self._last_parser_error: OutputParserError | None = None
        self._last_context_error: Exception | None = None

        # Execution guard to prevent concurrent/duplicate executions
        self._is_executing: bool = False
        self._has_been_invoked: bool = False
        self._flow_initialized: bool = False  # Track if Flow.__init__ was called

        # Debug: Track instance creation
        self._instance_id = str(uuid4())[:8]

        # Initialize hooks
        self.before_llm_call_hooks: list[Callable] = []
        self.after_llm_call_hooks: list[Callable] = []
        self.before_llm_call_hooks.extend(get_before_llm_call_hooks())
        self.after_llm_call_hooks.extend(get_after_llm_call_hooks())

        # Configure LLM stop words
        if self.llm:
            existing_stop = getattr(self.llm, "stop", [])
            self.llm.stop = list(
                set(
                    existing_stop + self.stop
                    if isinstance(existing_stop, list)
                    else self.stop
                )
            )

        # Create a temporary minimal state for property access before Flow init
        self._state = AgentReActState()

    def _ensure_flow_initialized(self) -> None:
        """Ensure Flow.__init__() has been called.

        This is deferred from __init__ to prevent FlowCreatedEvent emission
        during agent setup when multiple executor instances are created.
        Only the instance that actually executes via invoke() will emit events.
        """
        if not self._flow_initialized:
            # Now call Flow's __init__ which will replace self._state
            # with Flow's managed state
            super().__init__()
            self._flow_initialized = True
            self._printer.print(
                content=f"🌊 Flow initialized for instance: {self._instance_id}",
                color="blue",
            )

    @property
    def use_stop_words(self) -> bool:
        """Check to determine if stop words are being used.

        Returns:
            bool: True if stop words should be used.
        """
        return self.llm.supports_stop_words() if self.llm else False

    @property
    def state(self) -> AgentReActState:
        """Get state - returns temporary state if Flow not yet initialized.

        Flow initialization is deferred to prevent event emission during agent setup.
        Returns the temporary state until invoke() is called.
        """
        return self._state

    @property
    def messages(self) -> list[LLMMessage]:
        """Compatibility property for mixin - returns state messages."""
        return list(self._state.messages)

    @property
    def iterations(self) -> int:
        """Compatibility property for mixin - returns state iterations."""
        return self._state.iterations

    @start()
    def initialize_reasoning(self) -> str:
        """Initialize flow state and messages.

        Maps to: Initial prompt formatting in current executor's invoke() method
        Reference: crew_agent_executor.py lines 170-181

        Flow Event: START -> triggers check_max_iterations_router
        """
        self._show_start_logs()
        return "initialized"

    @listen("force_final_answer")
    def force_final_answer(self) -> str:
        """Force agent to provide final answer when max iterations exceeded.

        Maps to: handle_max_iterations_exceeded at lines 217-224
        Reference: crew_agent_executor.py lines 217-224

        Flow Event: "force_final_answer" -> "agent_finished"
        """
        formatted_answer = handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=self._printer,
            i18n=self._i18n,
            messages=list(self.state.messages),
            llm=self.llm,
            callbacks=self.callbacks,
        )

        self.state.current_answer = formatted_answer
        self.state.is_finished = True

        return "agent_finished"

    @listen("continue_reasoning")
    def call_llm_and_parse(self) -> str:
        """Execute LLM call with hooks and parse response.

        Maps to: Lines 227-239 in _invoke_loop
        Reference: crew_agent_executor.py lines 227-239

        Flow Event: "continue_reasoning" -> "parsed" | "parser_error" | "context_error"

        Steps:
        1. enforce_rpm_limit
        2. get_llm_response (with before/after hooks already integrated)
        3. process_llm_response
        """
        self._printer.print(
            content=f"🤖 call_llm_and_parse: About to call LLM (iteration {self.state.iterations})",
            color="blue",
        )
        try:
            # RPM enforcement (line 227)
            enforce_rpm_limit(self.request_within_rpm_limit)

            # LLM call with hooks (lines 229-238)
            # Note: Hooks are already integrated in get_llm_response utility
            answer = get_llm_response(
                llm=self.llm,
                messages=list(self.state.messages),  # Pass copy of state messages
                callbacks=self.callbacks,
                printer=self._printer,
                from_task=self.task,
                from_agent=self.agent,
                response_model=self.response_model,
                executor_context=self,
            )
            print(f"answer for iteration: {self.state.iterations} is {answer}")

            # Parse response (line 239)
            formatted_answer = process_llm_response(answer, self.use_stop_words)
            self.state.current_answer = formatted_answer

            # Debug: Check what we parsed
            if "Final Answer:" in answer:
                self._printer.print(
                    content=f"⚠️ LLM returned Final Answer but parsed as: {type(formatted_answer).__name__}",
                    color="yellow",
                )
                if isinstance(formatted_answer, AgentAction):
                    self._printer.print(
                        content=f"Answer preview: {answer[:200]}...", color="yellow"
                    )

            return "parsed"

        except OutputParserError as e:
            # Store error context for recovery
            self._last_parser_error = e
            return "parser_error"

        except Exception as e:
            if is_context_length_exceeded(e):
                self._last_context_error = e
                return "context_error"
            # Re-raise other exceptions (including litellm errors)
            if e.__class__.__module__.startswith("litellm"):
                raise e
            handle_unknown_error(self._printer, e)
            raise

    @router(call_llm_and_parse)
    def route_by_answer_type(self) -> str:
        """Route based on whether answer is AgentAction or AgentFinish.

        Maps to: isinstance check at line 241
        Reference: crew_agent_executor.py line 241

        Flow Event: call_llm_and_parse completes -> ROUTE -> "execute_tool" | "agent_finished"
        """
        answer_type = type(self.state.current_answer).__name__
        self._printer.print(
            content=f"🚦 route_by_answer_type: Got {answer_type}",
            color="yellow",
        )
        if isinstance(self.state.current_answer, AgentAction):
            return "execute_tool"
        return "agent_finished"

    @listen("execute_tool")
    def execute_tool_action(self) -> str:
        """Execute tool and handle result.

        Maps to: Lines 242-273 in _invoke_loop
        Reference: crew_agent_executor.py lines 242-273

        Flow Event: "execute_tool" -> completion triggers router

        Steps:
        1. Extract fingerprint context
        2. Execute tool via execute_tool_and_check_finality
        3. Handle agent action (append observation)
        4. Invoke step callback
        """
        try:
            action = cast(AgentAction, self.state.current_answer)

            # Extract fingerprint context (lines 243-253)
            fingerprint_context = {}
            if (
                self.agent
                and hasattr(self.agent, "security_config")
                and hasattr(self.agent.security_config, "fingerprint")
            ):
                fingerprint_context = {
                    "agent_fingerprint": str(self.agent.security_config.fingerprint)
                }

            # Execute tool (lines 255-267)
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

            # Handle agent action - appends observation to messages (lines 268-270)
            result = self._handle_agent_action(action, tool_result)
            self.state.current_answer = result

            # Invoke step callback (line 272)
            self._invoke_step_callback(result)

            # Append message to state (line 273)
            if hasattr(result, "text"):
                self._append_message_to_state(result.text)

            # Check if tool result became a final answer (result_as_answer flag)
            if isinstance(result, AgentFinish):
                self.state.is_finished = True
                return "tool_result_is_final"

            return "tool_completed"

        except Exception as e:
            self._printer.print(content=f"Error in tool execution: {e}", color="red")
            raise

    @listen("initialized")
    def continue_iteration(self) -> str:
        """Bridge listener that catches 'initialized' event from increment_and_continue.

        This is needed because @router listens to METHODS, not EVENT STRINGS.
        increment_and_continue returns 'initialized' STRING which triggers this listener.
        """
        return "check_iteration"

    @router(or_(initialize_reasoning, continue_iteration))
    def check_max_iterations(self) -> str:
        """Check if max iterations reached before LLM call.

        Maps to: has_reached_max_iterations check at line 216
        Reference: crew_agent_executor.py lines 216-225

        Triggered by:
        - initialize_reasoning METHOD (first iteration)
        - continue_iteration METHOD (subsequent iterations after tool execution)

        Flow Event: ROUTE -> "force_final_answer" | "continue_reasoning"
        """
        self._printer.print(
            content=f"🔄 check_max_iterations: iteration {self.state.iterations}/{self.max_iter}",
            color="cyan",
        )
        if has_reached_max_iterations(self.state.iterations, self.max_iter):
            return "force_final_answer"
        return "continue_reasoning"

    @router(execute_tool_action)
    def increment_and_continue(self) -> str:
        """Increment iteration counter and loop back.

        Maps to: Loop continuation (iteration increment at line 301)
        Reference: crew_agent_executor.py line 301 (finally block)

        Flow Event: execute_tool_action completes -> ROUTER returns "loop_continue"
        """
        # Increment iterations (line 301)
        self.state.iterations += 1
        self._printer.print(
            content=f"+ increment_and_continue: Incremented to iteration {self.state.iterations}, looping back",
            color="magenta",
        )
        # Return "initialized" to trigger check_max_iterations router again (simple loop)
        return "initialized"

    @listen(or_("agent_finished", "tool_result_is_final"))
    def finalize(self) -> str:
        """Finalize execution and return result.

        Maps to: Final steps after loop (lines 307-313)
        Reference: crew_agent_executor.py lines 307-313

        Triggered by:
        - "agent_finished" (Router returns this when LLM gives Final Answer)
        - "tool_result_is_final" (Tool execution returns this when tool has result_as_answer=True)

        Flow Event: -> "completed" (END)
        """
        # Guard: Only finalize if we actually have a valid final answer
        # This prevents finalization during initialization or intermediate states
        if self.state.current_answer is None:
            self._printer.print(
                content="⚠️ Finalize called but no answer in state - skipping",
                color="yellow",
            )
            return "skipped"

        # Validate we have an AgentFinish (lines 307-311)
        if not isinstance(self.state.current_answer, AgentFinish):
            # This can happen if Flow is triggered during initialization
            # Don't raise error, just log and skip
            self._printer.print(
                content=f"⚠️ Finalize called with {type(self.state.current_answer).__name__} instead of AgentFinish - skipping",
                color="yellow",
            )
            return "skipped"

        self.state.is_finished = True

        # Show logs (line 312)
        self._show_logs(self.state.current_answer)

        return "completed"

    @listen("parser_error")
    def recover_from_parser_error(self) -> str:
        """Recover from output parser errors.

        Maps to: OutputParserError handling in _invoke_loop
        Reference: crew_agent_executor.py lines 275-282

        Flow Event: "parser_error" -> "initialized" (loops back via event)
        """
        formatted_answer = handle_output_parser_exception(
            e=self._last_parser_error,
            messages=list(self.state.messages),
            iterations=self.state.iterations,
            log_error_after=self.log_error_after,
            printer=self._printer,
        )

        # If error handler returns an answer, use it
        if formatted_answer:
            self.state.current_answer = formatted_answer

        # Increment iterations (finally block)
        self.state.iterations += 1

        # Loop back via initialized event
        return "initialized"

    @listen("context_error")
    def recover_from_context_length(self) -> str:
        """Recover from context length errors.

        Maps to: Context length exception handling in _invoke_loop
        Reference: crew_agent_executor.py lines 288-297

        Flow Event: "context_error" -> "initialized" (loops back via event)
        """
        handle_context_length(
            respect_context_window=self.respect_context_window,
            printer=self._printer,
            messages=list(self.state.messages),
            llm=self.llm,
            callbacks=self.callbacks,
            i18n=self._i18n,
        )

        # Increment iterations (finally block)
        self.state.iterations += 1

        # Loop back via initialized event
        return "initialized"

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute agent with given inputs - maintains compatibility.

        Maps to: invoke() method at lines 161-205
        Reference: crew_agent_executor.py lines 161-205

        This is the main entry point that maintains backward compatibility
        with the current CrewAgentExecutor interface.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        # Guard: Prevent concurrent executions
        # Ensure Flow is initialized before execution
        self._ensure_flow_initialized()

        if self._is_executing:
            raise RuntimeError(
                "Executor is already running. "
                "Cannot invoke the same executor instance concurrently."
            )

        self._is_executing = True
        self._has_been_invoked = True

        # Debug: Track invoke calls
        self._printer.print(
            content=f"🚀 FlowExecutor.invoke() called on instance: {self._instance_id}",
            color="green",
        )

        try:
            # Reset state for fresh execution
            # This is important because create_agent_executor may be called multiple times
            # during agent initialization, and we need clean state for each actual task execution
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False

            # Format and initialize messages (lines 170-181)
            if "system" in self.prompt:
                system_prompt = self._format_prompt(
                    cast(str, self.prompt.get("system", "")), inputs
                )
                user_prompt = self._format_prompt(
                    cast(str, self.prompt.get("user", "")), inputs
                )
                self.state.messages.append(
                    format_message_for_llm(system_prompt, role="system")
                )
                self.state.messages.append(format_message_for_llm(user_prompt))
            else:
                user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
                self.state.messages.append(format_message_for_llm(user_prompt))

            # Set human input flag (line 185)
            self.state.ask_for_human_input = bool(
                inputs.get("ask_for_human_input", False)
            )

            # Run the flow (replaces _invoke_loop call at line 188)
            self.kickoff()

            # Extract final answer from state
            formatted_answer = self.state.current_answer

            if not isinstance(formatted_answer, AgentFinish):
                raise RuntimeError(
                    "Agent execution ended without reaching a final answer."
                )

            # Handle human feedback if needed (lines 199-200)
            if self.state.ask_for_human_input:
                formatted_answer = self._handle_human_feedback(formatted_answer)

            # Create memories (lines 202-204)
            self._create_short_term_memory(formatted_answer)
            self._create_long_term_memory(formatted_answer)
            self._create_external_memory(formatted_answer)

            return {"output": formatted_answer.output}

        except AssertionError:
            self._printer.print(
                content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                color="red",
            )
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e)
            raise
        finally:
            # Always reset execution flag
            self._is_executing = False

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> AgentAction | AgentFinish:
        """Process agent action and tool execution.

        Reference: crew_agent_executor.py lines 315-343

        Args:
            formatted_answer: Agent's action to execute.
            tool_result: Result from tool execution.

        Returns:
            Updated action or final answer.
        """
        # Special case for add_image_tool
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
            messages=list(self.state.messages),  # Pass copy
            step_callback=self.step_callback,
            show_logs=self._show_logs,
        )

    def _invoke_step_callback(
        self, formatted_answer: AgentAction | AgentFinish
    ) -> None:
        """Invoke step callback.

        Reference: crew_agent_executor.py lines 345-354

        Args:
            formatted_answer: Current agent response.
        """
        if self.step_callback:
            self.step_callback(formatted_answer)

    def _append_message_to_state(
        self, text: str, role: Literal["user", "assistant", "system"] = "assistant"
    ) -> None:
        """Add message to state conversation history.

        Reference: crew_agent_executor.py lines 356-365
        Adapted to work with Flow state instead of instance variable.

        Args:
            text: Message content.
            role: Message role (default: assistant).
        """
        self.state.messages.append(format_message_for_llm(text, role=role))

    def _show_start_logs(self) -> None:
        """Emit agent start event.

        Reference: crew_agent_executor.py lines 367-380
        """
        if self.agent is None:
            raise ValueError("Agent cannot be None")

        crewai_event_bus.emit(
            self.agent,
            AgentLogsStartedEvent(
                agent_role=self.agent.role,
                task_description=(self.task.description if self.task else "Not Found"),
                verbose=self.agent.verbose
                or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
            ),
        )

    def _show_logs(self, formatted_answer: AgentAction | AgentFinish) -> None:
        """Emit agent execution event.

        Reference: crew_agent_executor.py lines 382-399

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
        """Save training data.

        Reference: crew_agent_executor.py lines 401-450

        Args:
            result: Agent's final output.
            human_feedback: Optional feedback from human.
        """
        agent_id = str(self.agent.id)
        train_iteration = (
            getattr(self.crew, "_train_iteration", None) if self.crew else None
        )

        if train_iteration is None or not isinstance(train_iteration, int):
            self._printer.print(
                content="Invalid or missing train iteration. Cannot save training data.",
                color="red",
            )
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
                self._printer.print(
                    content=(
                        f"No existing training data for agent {agent_id} and iteration "
                        f"{train_iteration}. Cannot save improved output."
                    ),
                    color="red",
                )
                return

        # Update the training data and save
        training_data[agent_id] = agent_training_data
        training_handler.save(training_data)

    @staticmethod
    def _format_prompt(prompt: str, inputs: dict[str, str]) -> str:
        """Format prompt with input values.

        Reference: crew_agent_executor.py lines 452-465

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
        """Process human feedback.

        Reference: crew_agent_executor.py lines 467-481

        Args:
            formatted_answer: Initial agent result.

        Returns:
            Final answer after feedback.
        """
        human_feedback = self._ask_human_input(formatted_answer.output)

        if self._is_training_mode():
            return self._handle_training_feedback(formatted_answer, human_feedback)

        return self._handle_regular_feedback(formatted_answer, human_feedback)

    def _is_training_mode(self) -> bool:
        """Check if training mode is active.

        Reference: crew_agent_executor.py lines 483-489

        Returns:
            True if in training mode.
        """
        return bool(self.crew and self.crew._train)

    def _handle_training_feedback(
        self, initial_answer: AgentFinish, feedback: str
    ) -> AgentFinish:
        """Process training feedback.

        Reference: crew_agent_executor.py lines 491-512

        Args:
            initial_answer: Initial agent output.
            feedback: Training feedback.

        Returns:
            Improved answer.
        """
        self._handle_crew_training_output(initial_answer, feedback)
        self.state.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )

        # Re-run flow for improved answer
        # Need to reset state appropriately
        self.state.iterations = 0
        self.state.is_finished = False
        self.state.current_answer = None

        self.kickoff()

        # Get improved answer from state
        improved_answer = self.state.current_answer
        if not isinstance(improved_answer, AgentFinish):
            raise RuntimeError(
                "Training feedback iteration did not produce final answer"
            )

        self._handle_crew_training_output(improved_answer)
        self.state.ask_for_human_input = False
        return improved_answer

    def _handle_regular_feedback(
        self, current_answer: AgentFinish, initial_feedback: str
    ) -> AgentFinish:
        """Process regular feedback iteratively.

        Reference: crew_agent_executor.py lines 514-537

        Args:
            current_answer: Current agent output.
            initial_feedback: Initial user feedback.

        Returns:
            Final answer after iterations.
        """
        feedback = initial_feedback
        answer = current_answer

        while self.state.ask_for_human_input:
            if feedback.strip() == "":
                self.state.ask_for_human_input = False
            else:
                answer = self._process_feedback_iteration(feedback)
                feedback = self._ask_human_input(answer.output)

        return answer

    def _process_feedback_iteration(self, feedback: str) -> AgentFinish:
        """Process single feedback iteration.

        Reference: crew_agent_executor.py lines 539-553

        Args:
            feedback: User feedback.

        Returns:
            Updated agent response.
        """
        self.state.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )

        # Re-run flow
        self.state.iterations = 0
        self.state.is_finished = False
        self.state.current_answer = None

        self.kickoff()

        # Get answer from state
        answer = self.state.current_answer
        if not isinstance(answer, AgentFinish):
            raise RuntimeError("Feedback iteration did not produce final answer")

        return answer

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for Protocol compatibility.

        Reference: crew_agent_executor.py lines 555-564

        This allows the Protocol to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()
