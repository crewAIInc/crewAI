from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import datetime
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
from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.utils import (
    is_tracing_enabled_in_context,
)
from crewai.events.types.logging_events import (
    AgentLogsExecutionEvent,
    AgentLogsStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.hooks.llm_hooks import (
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.hooks.types import AfterLLMCallHookType, BeforeLLMCallHookType
from crewai.utilities.agent_utils import (
    convert_tools_to_openai_schema,
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
    process_llm_response,
    track_delegation_if_needed,
)
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.printer import Printer
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
    use_native_tools: bool = Field(default=False)
    pending_tool_calls: list[Any] = Field(default_factory=list)


class AgentExecutor(Flow[AgentReActState], CrewAgentExecutorMixin):
    """Agent Executor for both standalone agents and crew-bound agents.

    Inherits from:
    - Flow[AgentReActState]: Provides flow orchestration capabilities
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
        self._is_executing: bool = False
        self._has_been_invoked: bool = False
        self._flow_initialized: bool = False

        self._instance_id = str(uuid4())[:8]

        self.before_llm_call_hooks: list[BeforeLLMCallHookType] = []
        self.after_llm_call_hooks: list[AfterLLMCallHookType] = []
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
            current_tracing = is_tracing_enabled_in_context()
            # Now call Flow's __init__ which will replace self._state
            # with Flow's managed state. Suppress flow events since this is
            # an agent executor, not a user-facing flow.
            super().__init__(
                suppress_flow_events=True,
                tracing=current_tracing if current_tracing else None,
            )
            self._flow_initialized = True

    def _check_native_tool_support(self) -> bool:
        """Check if LLM supports native function calling.

        Returns:
            True if the LLM supports native function calling and tools are available.
        """
        return (
            hasattr(self.llm, "supports_function_calling")
            and callable(getattr(self.llm, "supports_function_calling", None))
            and self.llm.supports_function_calling()
            and bool(self.original_tools)
        )

    def _setup_native_tools(self) -> None:
        """Convert tools to OpenAI schema format for native function calling."""
        if self.original_tools:
            self._openai_tools, self._available_functions = (
                convert_tools_to_openai_schema(self.original_tools)
            )

    def _is_tool_call_list(self, response: list[Any]) -> bool:
        """Check if a response is a list of tool calls.

        Args:
            response: The response to check.

        Returns:
            True if the response appears to be a list of tool calls.
        """
        if not response:
            return False
        first_item = response[0]
        # Check for OpenAI-style tool call structure
        if hasattr(first_item, "function") or (
            isinstance(first_item, dict) and "function" in first_item
        ):
            return True
        # Check for Anthropic-style tool call structure (ToolUseBlock)
        if (
            hasattr(first_item, "type")
            and getattr(first_item, "type", None) == "tool_use"
        ):
            return True
        if hasattr(first_item, "name") and hasattr(first_item, "input"):
            return True
        # Check for Bedrock-style tool call structure (dict with name and input keys)
        if (
            isinstance(first_item, dict)
            and "name" in first_item
            and "input" in first_item
        ):
            return True
        # Check for Gemini-style function call (Part with function_call)
        if hasattr(first_item, "function_call") and first_item.function_call:
            return True
        return False

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

    @messages.setter
    def messages(self, value: list[LLMMessage]) -> None:
        """Set state messages."""
        self._state.messages = value

    @property
    def iterations(self) -> int:
        """Compatibility property for mixin - returns state iterations."""
        return self._state.iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        """Set state iterations."""
        self._state.iterations = value

    @start()
    def initialize_reasoning(self) -> Literal["initialized"]:
        """Initialize the reasoning flow and emit agent start logs."""
        self._show_start_logs()
        # Check for native tool support on first iteration
        if self.state.iterations == 0:
            self.state.use_native_tools = self._check_native_tool_support()
            if self.state.use_native_tools:
                self._setup_native_tools()
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
            verbose=self.agent.verbose,
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

    @listen("continue_reasoning_native")
    def call_llm_native_tools(
        self,
    ) -> Literal["native_tool_calls", "native_finished", "context_error"]:
        """Execute LLM call with native function calling.

        Always calls the LLM so it can read reflection prompts and decide
        whether to provide a final answer or request more tools.

        Returns routing decision based on whether tool calls or final answer.
        """
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
                return "native_finished"

            # Text response - this is the final answer
            if isinstance(answer, str):
                self.state.current_answer = AgentFinish(
                    thought="",
                    output=answer,
                    text=answer,
                )
                self._invoke_step_callback(self.state.current_answer)
                self._append_message_to_state(answer)

                return "native_finished"

            # Unexpected response type, treat as final answer
            self.state.current_answer = AgentFinish(
                thought="",
                output=str(answer),
                text=str(answer),
            )
            self._invoke_step_callback(self.state.current_answer)
            self._append_message_to_state(str(answer))

            return "native_finished"

        except Exception as e:
            if is_context_length_exceeded(e):
                self._last_context_error = e
                return "context_error"
            if e.__class__.__module__.startswith("litellm"):
                raise e
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
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

            # Inject post-tool reasoning prompt to enforce analysis
            reasoning_prompt = self._i18n.slice("post_tool_reasoning")
            reasoning_message: LLMMessage = {
                "role": "user",
                "content": reasoning_prompt,
            }
            self.state.messages.append(reasoning_message)

            return "tool_completed"

        except Exception as e:
            error_text = Text()
            error_text.append("❌ Error in tool execution: ", style="red bold")
            error_text.append(str(e), style="red")
            self._console.print(error_text)
            raise

    @listen("native_tool_calls")
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

        # Group all tool calls into a single assistant message
        tool_calls_to_report = []
        for tool_call in self.state.pending_tool_calls:
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
            if all(
                type(tc).__qualname__ == "Part" for tc in self.state.pending_tool_calls
            ):
                assistant_message["raw_tool_call_parts"] = list(
                    self.state.pending_tool_calls
                )
            self.state.messages.append(assistant_message)

        # Now execute each tool
        while self.state.pending_tool_calls:
            tool_call = self.state.pending_tool_calls.pop(0)
            info = extract_tool_call_info(tool_call)
            if not info:
                continue

            call_id, func_name, func_args = info

            # Parse arguments
            if isinstance(func_args, str):
                try:
                    args_dict = json.loads(func_args)
                except json.JSONDecodeError:
                    args_dict = {}
            else:
                args_dict = func_args

            # Get agent_key for event tracking
            agent_key = (
                getattr(self.agent, "key", "unknown") if self.agent else "unknown"
            )

            # Find original tool by matching sanitized name (needed for cache_function and result_as_answer)
            original_tool = None
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

            track_delegation_if_needed(func_name, args_dict, self.task)

            structured_tool: CrewStructuredTool | None = None
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
            elif max_usage_reached and original_tool:
                # Return error message when max usage limit is reached
                result = f"Tool '{func_name}' has reached its usage limit of {original_tool.max_usage_count} times and cannot be used anymore."

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

            # Emit tool usage finished event
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

            # Append tool result message
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
                # Set the result as the final answer
                self.state.current_answer = AgentFinish(
                    thought="Tool result is the final answer",
                    output=result,
                    text=result,
                )
                self.state.is_finished = True
                return "tool_result_is_final"

        return "native_tool_completed"

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
    def increment_native_and_continue(self) -> Literal["initialized"]:
        """Increment iteration counter after native tool execution."""
        self.state.iterations += 1
        return "initialized"

    @listen("initialized")
    def continue_iteration(self) -> Literal["check_iteration"]:
        """Bridge listener that connects iteration loop back to iteration check."""
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
    def increment_and_continue(self) -> Literal["initialized"]:
        """Increment iteration counter and loop back for next iteration."""
        self.state.iterations += 1
        return "initialized"

    @listen(or_("agent_finished", "tool_result_is_final", "native_finished"))
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
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False
            self.state.use_native_tools = False
            self.state.pending_tool_calls = []

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
            self.state.messages.clear()
            self.state.iterations = 0
            self.state.current_answer = None
            self.state.is_finished = False
            self.state.use_native_tools = False
            self.state.pending_tool_calls = []

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
        output_str = (
            str(formatted_answer.output)
            if isinstance(formatted_answer.output, BaseModel)
            else formatted_answer.output
        )
        human_feedback = self._ask_human_input(output_str)

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
                output_str = (
                    str(answer.output)
                    if isinstance(answer.output, BaseModel)
                    else answer.output
                )
                feedback = self._ask_human_input(output_str)

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


# Backward compatibility alias (deprecated)
CrewAgentExecutorFlow = AgentExecutor
