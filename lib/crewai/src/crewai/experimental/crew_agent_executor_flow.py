from __future__ import annotations

from collections.abc import Callable
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
        i18n: I18N | None = None,
    ) -> None:
        """Initialize the flow-based agent executor.

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
        self._i18n: I18N = i18n or get_i18n()
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
        self._console: Console = Console()

        # Error context storage for recovery
        self._last_parser_error: OutputParserError | None = None
        self._last_context_error: Exception | None = None

        # Execution guard to prevent concurrent/duplicate executions
        self._execution_lock = threading.Lock()
        self._is_executing: bool = False
        self._has_been_invoked: bool = False
        self._flow_initialized: bool = False

        self._instance_id = str(uuid4())[:8]

        self.before_llm_call_hooks: list[Callable] = []
        self.after_llm_call_hooks: list[Callable] = []
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

        self._state = AgentReActState()

    def _ensure_flow_initialized(self) -> None:
        """Ensure Flow.__init__() has been called.

        This is deferred from __init__ to prevent FlowCreatedEvent emission
        during agent setup when multiple executor instances are created.
        Only the instance that actually executes via invoke() will emit events.
        """
        if not self._flow_initialized:
            # Now call Flow's __init__ which will replace self._state
            # with Flow's managed state. Suppress flow events since this is
            # an agent executor, not a user-facing flow.
            super().__init__(
                suppress_flow_events=True,
            )
            self._flow_initialized = True

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
        return self._state.messages

    @property
    def iterations(self) -> int:
        """Compatibility property for mixin - returns state iterations."""
        return self._state.iterations

    @start()
    def initialize_reasoning(self) -> Literal["initialized"]:
        """Initialize the reasoning flow and emit agent start logs."""
        self._show_start_logs()
        return "initialized"

    @listen("force_final_answer")
    def force_final_answer(self) -> Literal["agent_finished"]:
        """Force agent to provide final answer when max iterations exceeded."""
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
    def call_llm_and_parse(self) -> Literal["parsed", "parser_error", "context_error"]:
        """Execute LLM call with hooks and parse the response.

        Returns routing decision based on parsing result.
        """
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
            )

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
            handle_unknown_error(self._printer, e)
            raise

    @router(call_llm_and_parse)
    def route_by_answer_type(self) -> Literal["execute_tool", "agent_finished"]:
        """Route based on whether answer is AgentAction or AgentFinish."""
        if isinstance(self.state.current_answer, AgentAction):
            return "execute_tool"
        return "agent_finished"

    @listen("execute_tool")
    def execute_tool_action(self) -> Literal["tool_completed", "tool_result_is_final"]:
        """Execute the tool action and handle the result."""
        try:
            action = cast(AgentAction, self.state.current_answer)

            # Extract fingerprint context for tool execution
            fingerprint_context = {}
            if (
                self.agent
                and hasattr(self.agent, "security_config")
                and hasattr(self.agent.security_config, "fingerprint")
            ):
                fingerprint_context = {
                    "agent_fingerprint": str(self.agent.security_config.fingerprint)
                }

            # Execute the tool
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

            # Handle agent action and append observation to messages
            result = self._handle_agent_action(action, tool_result)
            self.state.current_answer = result

            # Invoke step callback if configured
            self._invoke_step_callback(result)

            # Append result message to conversation state
            if hasattr(result, "text"):
                self._append_message_to_state(result.text)

            # Check if tool result became a final answer (result_as_answer flag)
            if isinstance(result, AgentFinish):
                self.state.is_finished = True
                return "tool_result_is_final"

            return "tool_completed"

        except Exception as e:
            error_text = Text()
            error_text.append("❌ Error in tool execution: ", style="red bold")
            error_text.append(str(e), style="red")
            self._console.print(error_text)
            raise

    @listen("initialized")
    def continue_iteration(self) -> Literal["check_iteration"]:
        """Bridge listener that connects iteration loop back to iteration check."""
        return "check_iteration"

    @router(or_(initialize_reasoning, continue_iteration))
    def check_max_iterations(
        self,
    ) -> Literal["force_final_answer", "continue_reasoning"]:
        """Check if max iterations reached before proceeding with reasoning."""
        if has_reached_max_iterations(self.state.iterations, self.max_iter):
            return "force_final_answer"
        return "continue_reasoning"

    @router(execute_tool_action)
    def increment_and_continue(self) -> Literal["initialized"]:
        """Increment iteration counter and loop back for next iteration."""
        self.state.iterations += 1
        return "initialized"

    @listen(or_("agent_finished", "tool_result_is_final"))
    def finalize(self) -> Literal["completed", "skipped"]:
        """Finalize execution and emit completion logs."""
        if self.state.current_answer is None:
            skip_text = Text()
            skip_text.append("⚠️ ", style="yellow bold")
            skip_text.append(
                "Finalize called but no answer in state - skipping", style="yellow"
            )
            self._console.print(skip_text)
            return "skipped"

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

    @listen("parser_error")
    def recover_from_parser_error(self) -> Literal["initialized"]:
        """Recover from output parser errors and retry."""
        formatted_answer = handle_output_parser_exception(
            e=self._last_parser_error,
            messages=list(self.state.messages),
            iterations=self.state.iterations,
            log_error_after=self.log_error_after,
            printer=self._printer,
        )

        if formatted_answer:
            self.state.current_answer = formatted_answer

        self.state.iterations += 1

        return "initialized"

    @listen("context_error")
    def recover_from_context_length(self) -> Literal["initialized"]:
        """Recover from context length errors and retry."""
        handle_context_length(
            respect_context_window=self.respect_context_window,
            printer=self._printer,
            messages=self.state.messages,
            llm=self.llm,
            callbacks=self.callbacks,
            i18n=self._i18n,
        )

        self.state.iterations += 1

        return "initialized"

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute agent with given inputs.

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
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False

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

            self._create_short_term_memory(formatted_answer)
            self._create_long_term_memory(formatted_answer)
            self._create_external_memory(formatted_answer)

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
            handle_unknown_error(self._printer, e)
            raise
        finally:
            self._is_executing = False

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
            self.step_callback(formatted_answer)

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
        agent_id = str(self.agent.id)
        train_iteration = (
            getattr(self.crew, "_train_iteration", None) if self.crew else None
        )

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
        human_feedback = self._ask_human_input(formatted_answer.output)

        if self._is_training_mode():
            return self._handle_training_feedback(formatted_answer, human_feedback)

        return self._handle_regular_feedback(formatted_answer, human_feedback)

    def _is_training_mode(self) -> bool:
        """Check if training mode is active.

        Returns:
            True if in training mode.
        """
        return bool(self.crew and self.crew._train)

    def _handle_training_feedback(
        self, initial_answer: AgentFinish, feedback: str
    ) -> AgentFinish:
        """Process training feedback and generate improved answer.

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
        """Process regular feedback iteratively until user is satisfied.

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
        """Process a single feedback iteration and generate updated response.

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

        Allows the executor to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()
