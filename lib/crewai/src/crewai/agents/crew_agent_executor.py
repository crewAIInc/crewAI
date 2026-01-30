"""Agent executor for crew AI agents.

Handles agent execution flow including LLM interactions, tool execution,
and memory management.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, GetCoreSchemaHandler, ValidationError
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
from crewai.hooks.llm_hooks import (
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.utilities.agent_utils import (
    aget_llm_response,
    convert_tools_to_openai_schema,
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
    track_delegation_if_needed,
)
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.file_store import aget_all_files, get_all_files
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.tool_utils import (
    aexecute_tool_and_check_finality,
    execute_tool_and_check_finality,
)
from crewai.utilities.training_handler import CrewTrainingHandler


logger = logging.getLogger(__name__)

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
    from crewai.utilities.types import LLMMessage


class CrewAgentExecutor(CrewAgentExecutorMixin):
    """Executor for crew agents.

    Manages the execution lifecycle of an agent including prompt formatting,
    LLM interactions, tool execution, and feedback handling.
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
        """Initialize executor.

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
        self.ask_for_human_input = False
        self.messages: list[LLMMessage] = []
        self.iterations = 0
        self.log_error_after = 3
        self.before_llm_call_hooks: list[Callable[..., Any]] = []
        self.after_llm_call_hooks: list[Callable[..., Any]] = []
        self.before_llm_call_hooks.extend(get_before_llm_call_hooks())
        self.after_llm_call_hooks.extend(get_after_llm_call_hooks())
        if self.llm:
            # This may be mutating the shared llm object and needs further evaluation
            existing_stop = getattr(self.llm, "stop", [])
            self.llm.stop = list(
                set(
                    existing_stop + self.stop
                    if isinstance(existing_stop, list)
                    else self.stop
                )
            )

    @property
    def use_stop_words(self) -> bool:
        """Check to determine if stop words are being used.

        Returns:
            bool: True if tool should be used or not.
        """
        return self.llm.supports_stop_words() if self.llm else False

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with given inputs.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        if "system" in self.prompt:
            system_prompt = self._format_prompt(
                cast(str, self.prompt.get("system", "")), inputs
            )
            user_prompt = self._format_prompt(
                cast(str, self.prompt.get("user", "")), inputs
            )
            self.messages.append(format_message_for_llm(system_prompt, role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        else:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(format_message_for_llm(user_prompt))

        self._inject_multimodal_files(inputs)

        self._show_start_logs()

        self.ask_for_human_input = bool(inputs.get("ask_for_human_input", False))

        try:
            formatted_answer = self._invoke_loop()
        except AssertionError:
            if self.agent.verbose:
                self._printer.print(
                    content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                    color="red",
                )
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise

        if self.ask_for_human_input:
            formatted_answer = self._handle_human_feedback(formatted_answer)

        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        self._create_external_memory(formatted_answer)
        return {"output": formatted_answer.output}

    def _inject_multimodal_files(self, inputs: dict[str, Any] | None = None) -> None:
        """Attach files to the last user message for LLM-layer formatting.

        Merges files from crew/task store and inputs dict, then attaches them
        to the message's `files` field. Input files take precedence over
        crew/task files with the same name.

        Args:
            inputs: Optional inputs dict that may contain files.
        """
        files: dict[str, Any] = {}

        if self.crew and self.task:
            crew_files = get_all_files(self.crew.id, self.task.id)
            if crew_files:
                files.update(crew_files)

        if inputs and inputs.get("files"):
            files.update(inputs["files"])

        if not files:
            return

        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") == "user":
                msg["files"] = files
                break

    async def _ainject_multimodal_files(
        self, inputs: dict[str, Any] | None = None
    ) -> None:
        """Async attach files to the last user message for LLM-layer formatting.

        Merges files from crew/task store and inputs dict, then attaches them
        to the message's `files` field. Input files take precedence over
        crew/task files with the same name.

        Args:
            inputs: Optional inputs dict that may contain files.
        """
        files: dict[str, Any] = {}

        if self.crew and self.task:
            crew_files = await aget_all_files(self.crew.id, self.task.id)
            if crew_files:
                files.update(crew_files)

        if inputs and inputs.get("files"):
            files.update(inputs["files"])

        if not files:
            return

        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.get("role") == "user":
                msg["files"] = files
                break

    def _invoke_loop(self) -> AgentFinish:
        """Execute agent loop until completion.

        Checks if the LLM supports native function calling and uses that
        approach if available, otherwise falls back to the ReAct text pattern.

        Returns:
            Final answer from the agent.
        """
        # Check if model supports native function calling
        use_native_tools = (
            hasattr(self.llm, "supports_function_calling")
            and callable(getattr(self.llm, "supports_function_calling", None))
            and self.llm.supports_function_calling()
            and self.original_tools
        )

        if use_native_tools:
            return self._invoke_loop_native_tools()

        # Fall back to ReAct text-based pattern
        return self._invoke_loop_react()

    def _invoke_loop_react(self) -> AgentFinish:
        """Execute agent loop using ReAct text-based pattern.

        This is the traditional approach where tool definitions are embedded
        in the prompt and the LLM outputs Action/Action Input text that is
        parsed to execute tools.

        Returns:
            Final answer from the agent.
        """
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    break

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )
                # breakpoint()
                if self.response_model is not None:
                    try:
                        if isinstance(answer, BaseModel):
                            output_json = answer.model_dump_json()
                            formatted_answer = AgentFinish(
                                thought="",
                                output=answer,
                                text=output_json,
                            )
                        else:
                            self.response_model.model_validate_json(answer)
                            formatted_answer = AgentFinish(
                                thought="",
                                output=answer,
                                text=answer,
                            )
                    except ValidationError:
                        # If validation fails, convert BaseModel to JSON string for parsing
                        answer_str = (
                            answer.model_dump_json()
                            if isinstance(answer, BaseModel)
                            else str(answer)
                        )
                        formatted_answer = process_llm_response(
                            answer_str, self.use_stop_words
                        )  # type: ignore[assignment]
                else:
                    # When no response_model, answer should be a string
                    answer_str = str(answer) if not isinstance(answer, str) else answer
                    formatted_answer = process_llm_response(
                        answer_str, self.use_stop_words
                    )  # type: ignore[assignment]

                if isinstance(formatted_answer, AgentAction):
                    # Extract agent fingerprint if available
                    fingerprint_context = {}
                    if (
                        self.agent
                        and hasattr(self.agent, "security_config")
                        and hasattr(self.agent.security_config, "fingerprint")
                    ):
                        fingerprint_context = {
                            "agent_fingerprint": str(
                                self.agent.security_config.fingerprint
                            )
                        }

                    tool_result = execute_tool_and_check_finality(
                        agent_action=formatted_answer,
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
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                self._invoke_step_callback(formatted_answer)  # type: ignore[arg-type]
                self._append_message(formatted_answer.text)  # type: ignore[union-attr]

            except OutputParserError as e:
                formatted_answer = handle_output_parser_exception(  # type: ignore[assignment]
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=self._printer,
                    verbose=self.agent.verbose,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

        # During the invoke loop, formatted_answer alternates between AgentAction
        # (when the agent is using tools) and eventually becomes AgentFinish
        # (when the agent reaches a final answer). This check confirms we've
        # reached a final answer and helps type checking understand this transition.
        if not isinstance(formatted_answer, AgentFinish):
            raise RuntimeError(
                "Agent execution ended without reaching a final answer. "
                f"Got {type(formatted_answer).__name__} instead of AgentFinish."
            )
        self._show_logs(formatted_answer)
        return formatted_answer

    def _invoke_loop_native_tools(self) -> AgentFinish:
        """Execute agent loop using native function calling.

        This method uses the LLM's native tool/function calling capability
        instead of the text-based ReAct pattern. The LLM directly returns
        structured tool calls which are executed and results fed back.

        Returns:
            Final answer from the agent.
        """
        # Convert tools to OpenAI schema format
        if not self.original_tools:
            # No tools available, fall back to simple LLM call
            return self._invoke_loop_native_no_tools()

        openai_tools, available_functions = convert_tools_to_openai_schema(
            self.original_tools
        )

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                enforce_rpm_limit(self.request_within_rpm_limit)

                # Call LLM with native tools
                # Pass available_functions=None so the LLM returns tool_calls
                # without executing them. The executor handles tool execution
                # via _handle_native_tool_calls to properly manage message history.
                answer = get_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )

                # Check if the response is a list of tool calls
                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    # Handle tool calls - execute tools and add results to messages
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    # If tool has result_as_answer=True, return immediately
                    if tool_finish is not None:
                        return tool_finish
                    # Continue loop to let LLM analyze results and decide next steps
                    continue

                # Text or other response - handle as potential final answer
                if isinstance(answer, str):
                    # Text response - this is the final answer
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=answer,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(answer)  # Save final answer to messages
                    self._show_logs(formatted_answer)
                    return formatted_answer

                if isinstance(answer, BaseModel):
                    output_json = answer.model_dump_json()
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=output_json,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(output_json)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                # Unexpected response type, treat as final answer
                formatted_answer = AgentFinish(
                    thought="",
                    output=str(answer),
                    text=str(answer),
                )
                self._invoke_step_callback(formatted_answer)
                self._append_message(str(answer))  # Save final answer to messages
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

    def _invoke_loop_native_no_tools(self) -> AgentFinish:
        """Execute a simple LLM call when no tools are available.

        Returns:
            Final answer from the agent.
        """
        enforce_rpm_limit(self.request_within_rpm_limit)

        answer = get_llm_response(
            llm=self.llm,
            messages=self.messages,
            callbacks=self.callbacks,
            printer=self._printer,
            from_task=self.task,
            from_agent=self.agent,
            response_model=self.response_model,
            executor_context=self,
            verbose=self.agent.verbose,
        )

        if isinstance(answer, BaseModel):
            output_json = answer.model_dump_json()
            formatted_answer = AgentFinish(
                thought="",
                output=answer,
                text=output_json,
            )
        else:
            answer_str = answer if isinstance(answer, str) else str(answer)
            formatted_answer = AgentFinish(
                thought="",
                output=answer_str,
                text=answer_str,
            )
        self._show_logs(formatted_answer)
        return formatted_answer

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
        # OpenAI-style
        if hasattr(first_item, "function") or (
            isinstance(first_item, dict) and "function" in first_item
        ):
            return True
        # Anthropic-style (object with attributes)
        if (
            hasattr(first_item, "type")
            and getattr(first_item, "type", None) == "tool_use"
        ):
            return True
        if hasattr(first_item, "name") and hasattr(first_item, "input"):
            return True
        # Bedrock-style (dict with name and input keys)
        if (
            isinstance(first_item, dict)
            and "name" in first_item
            and "input" in first_item
        ):
            return True
        # Gemini-style
        if hasattr(first_item, "function_call") and first_item.function_call:
            return True
        return False

    def _handle_native_tool_calls(
        self,
        tool_calls: list[Any],
        available_functions: dict[str, Callable[..., Any]],
    ) -> AgentFinish | None:
        """Handle a single native tool call from the LLM.

        Executes only the FIRST tool call and appends the result to message history.
        This enables sequential tool execution with reflection after each tool,
        allowing the LLM to reason about results before deciding on next steps.

        Args:
            tool_calls: List of tool calls from the LLM (only first is processed).
            available_functions: Dict mapping function names to callables.

        Returns:
            AgentFinish if tool has result_as_answer=True, None otherwise.
        """
        from datetime import datetime
        import json

        from crewai.events import crewai_event_bus
        from crewai.events.types.tool_usage_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        if not tool_calls:
            return None

        # Only process the FIRST tool call for sequential execution with reflection
        tool_call = tool_calls[0]

        # Extract tool call info - handle OpenAI-style, Anthropic-style, and Gemini-style
        if hasattr(tool_call, "function"):
            # OpenAI-style: has .function.name and .function.arguments
            call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
            func_name = sanitize_tool_name(tool_call.function.name)
            func_args = tool_call.function.arguments
        elif hasattr(tool_call, "function_call") and tool_call.function_call:
            # Gemini-style: has .function_call.name and .function_call.args
            call_id = f"call_{id(tool_call)}"
            func_name = sanitize_tool_name(tool_call.function_call.name)
            func_args = (
                dict(tool_call.function_call.args)
                if tool_call.function_call.args
                else {}
            )
        elif hasattr(tool_call, "name") and hasattr(tool_call, "input"):
            # Anthropic format: has .name and .input (ToolUseBlock)
            call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
            func_name = sanitize_tool_name(tool_call.name)
            func_args = tool_call.input  # Already a dict in Anthropic
        elif isinstance(tool_call, dict):
            # Support OpenAI "id", Bedrock "toolUseId", or generate one
            call_id = (
                tool_call.get("id")
                or tool_call.get("toolUseId")
                or f"call_{id(tool_call)}"
            )
            func_info = tool_call.get("function", {})
            func_name = sanitize_tool_name(
                func_info.get("name", "") or tool_call.get("name", "")
            )
            func_args = func_info.get("arguments", "{}") or tool_call.get("input", {})
        else:
            return None

        # Append assistant message with single tool call
        assistant_message: LLMMessage = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
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
            ],
        }

        self.messages.append(assistant_message)

        # Parse arguments for the single tool call
        if isinstance(func_args, str):
            try:
                args_dict = json.loads(func_args)
            except json.JSONDecodeError:
                args_dict = {}
        else:
            args_dict = func_args

        agent_key = getattr(self.agent, "key", "unknown") if self.agent else "unknown"

        # Find original tool by matching sanitized name (needed for cache_function and result_as_answer)

        original_tool = None
        for tool in self.original_tools or []:
            if sanitize_tool_name(tool.name) == func_name:
                original_tool = tool
                break

        # Check if tool has reached max usage count
        max_usage_reached = False
        if original_tool:
            if (
                hasattr(original_tool, "max_usage_count")
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

        # Find the structured tool for hook context
        structured_tool: CrewStructuredTool | None = None
        for structured in self.tools or []:
            if sanitize_tool_name(structured.name) == func_name:
                structured_tool = structured
                break

        # Execute before_tool_call hooks
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

        # If hook blocked execution, set result and skip tool execution
        if hook_blocked:
            result = f"Tool execution blocked by hook. Tool: {func_name}"
        # Execute the tool (only if not cached, not at max usage, and not blocked by hook)
        elif not from_cache and not max_usage_reached:
            result = "Tool not found"
            if func_name in available_functions:
                try:
                    tool_func = available_functions[func_name]
                    raw_result = tool_func(**args_dict)

                    # Add to cache after successful execution (before string conversion)
                    if self.tools_handler and self.tools_handler.cache:
                        should_cache = True
                        if (
                            original_tool
                            and hasattr(original_tool, "cache_function")
                            and callable(original_tool.cache_function)
                        ):
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
        self.messages.append(tool_message)

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
            # Return immediately with tool result as final answer
            return AgentFinish(
                thought="Tool result is the final answer",
                output=result,
                text=result,
            )

        # Inject post-tool reasoning prompt to enforce analysis
        reasoning_prompt = self._i18n.slice("post_tool_reasoning")
        reasoning_message: LLMMessage = {
            "role": "user",
            "content": reasoning_prompt,
        }
        self.messages.append(reasoning_message)
        return None

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent asynchronously with given inputs.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        if "system" in self.prompt:
            system_prompt = self._format_prompt(
                cast(str, self.prompt.get("system", "")), inputs
            )
            user_prompt = self._format_prompt(
                cast(str, self.prompt.get("user", "")), inputs
            )
            self.messages.append(format_message_for_llm(system_prompt, role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        else:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(format_message_for_llm(user_prompt))

        await self._ainject_multimodal_files(inputs)

        self._show_start_logs()

        self.ask_for_human_input = bool(inputs.get("ask_for_human_input", False))

        try:
            formatted_answer = await self._ainvoke_loop()
        except AssertionError:
            if self.agent.verbose:
                self._printer.print(
                    content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                    color="red",
                )
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
            raise

        if self.ask_for_human_input:
            formatted_answer = self._handle_human_feedback(formatted_answer)

        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        self._create_external_memory(formatted_answer)
        return {"output": formatted_answer.output}

    async def _ainvoke_loop(self) -> AgentFinish:
        """Execute agent loop asynchronously until completion.

        Checks if the LLM supports native function calling and uses that
        approach if available, otherwise falls back to the ReAct text pattern.

        Returns:
            Final answer from the agent.
        """
        # Check if model supports native function calling
        use_native_tools = (
            hasattr(self.llm, "supports_function_calling")
            and callable(getattr(self.llm, "supports_function_calling", None))
            and self.llm.supports_function_calling()
            and self.original_tools
        )

        if use_native_tools:
            return await self._ainvoke_loop_native_tools()

        # Fall back to ReAct text-based pattern
        return await self._ainvoke_loop_react()

    async def _ainvoke_loop_react(self) -> AgentFinish:
        """Execute agent loop asynchronously using ReAct text-based pattern.

        Returns:
            Final answer from the agent.
        """
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    break

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = await aget_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )

                if self.response_model is not None:
                    try:
                        if isinstance(answer, BaseModel):
                            output_json = answer.model_dump_json()
                            formatted_answer = AgentFinish(
                                thought="",
                                output=answer,
                                text=output_json,
                            )
                        else:
                            self.response_model.model_validate_json(answer)
                            formatted_answer = AgentFinish(
                                thought="",
                                output=answer,
                                text=answer,
                            )
                    except ValidationError:
                        # If validation fails, convert BaseModel to JSON string for parsing
                        answer_str = (
                            answer.model_dump_json()
                            if isinstance(answer, BaseModel)
                            else str(answer)
                        )
                        formatted_answer = process_llm_response(
                            answer_str, self.use_stop_words
                        )  # type: ignore[assignment]
                else:
                    # When no response_model, answer should be a string
                    answer_str = str(answer) if not isinstance(answer, str) else answer
                    formatted_answer = process_llm_response(
                        answer_str, self.use_stop_words
                    )  # type: ignore[assignment]

                if isinstance(formatted_answer, AgentAction):
                    fingerprint_context = {}
                    if (
                        self.agent
                        and hasattr(self.agent, "security_config")
                        and hasattr(self.agent.security_config, "fingerprint")
                    ):
                        fingerprint_context = {
                            "agent_fingerprint": str(
                                self.agent.security_config.fingerprint
                            )
                        }

                    tool_result = await aexecute_tool_and_check_finality(
                        agent_action=formatted_answer,
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
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                self._invoke_step_callback(formatted_answer)  # type: ignore[arg-type]
                self._append_message(formatted_answer.text)  # type: ignore[union-attr]

            except OutputParserError as e:
                formatted_answer = handle_output_parser_exception(  # type: ignore[assignment]
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=self._printer,
                    verbose=self.agent.verbose,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

        if not isinstance(formatted_answer, AgentFinish):
            raise RuntimeError(
                "Agent execution ended without reaching a final answer. "
                f"Got {type(formatted_answer).__name__} instead of AgentFinish."
            )
        self._show_logs(formatted_answer)
        return formatted_answer

    async def _ainvoke_loop_native_tools(self) -> AgentFinish:
        """Execute agent loop asynchronously using native function calling.

        This method uses the LLM's native tool/function calling capability
        instead of the text-based ReAct pattern.

        Returns:
            Final answer from the agent.
        """
        # Convert tools to OpenAI schema format
        if not self.original_tools:
            return await self._ainvoke_loop_native_no_tools()

        openai_tools, available_functions = convert_tools_to_openai_schema(
            self.original_tools
        )

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                enforce_rpm_limit(self.request_within_rpm_limit)

                # Call LLM with native tools
                # Pass available_functions=None so the LLM returns tool_calls
                # without executing them. The executor handles tool execution
                # via _handle_native_tool_calls to properly manage message history.
                answer = await aget_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )
                # Check if the response is a list of tool calls
                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    # Handle tool calls - execute tools and add results to messages
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    # If tool has result_as_answer=True, return immediately
                    if tool_finish is not None:
                        return tool_finish
                    # Continue loop to let LLM analyze results and decide next steps
                    continue

                # Text or other response - handle as potential final answer
                if isinstance(answer, str):
                    # Text response - this is the final answer
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=answer,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(answer)  # Save final answer to messages
                    self._show_logs(formatted_answer)
                    return formatted_answer

                if isinstance(answer, BaseModel):
                    output_json = answer.model_dump_json()
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=output_json,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(output_json)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                # Unexpected response type, treat as final answer
                formatted_answer = AgentFinish(
                    thought="",
                    output=str(answer),
                    text=str(answer),
                )
                self._invoke_step_callback(formatted_answer)
                self._append_message(str(answer))  # Save final answer to messages
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(self._printer, e, verbose=self.agent.verbose)
                raise e
            finally:
                self.iterations += 1

    async def _ainvoke_loop_native_no_tools(self) -> AgentFinish:
        """Execute a simple async LLM call when no tools are available.

        Returns:
            Final answer from the agent.
        """
        enforce_rpm_limit(self.request_within_rpm_limit)

        answer = await aget_llm_response(
            llm=self.llm,
            messages=self.messages,
            callbacks=self.callbacks,
            printer=self._printer,
            from_task=self.task,
            from_agent=self.agent,
            response_model=self.response_model,
            executor_context=self,
            verbose=self.agent.verbose,
        )

        if isinstance(answer, BaseModel):
            output_json = answer.model_dump_json()
            formatted_answer = AgentFinish(
                thought="",
                output=answer,
                text=output_json,
            )
        else:
            answer_str = answer if isinstance(answer, str) else str(answer)
            formatted_answer = AgentFinish(
                thought="",
                output=answer_str,
                text=answer_str,
            )
        self._show_logs(formatted_answer)
        return formatted_answer

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> AgentAction | AgentFinish:
        """Process agent action and tool execution.

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
            self.messages.append({"role": "assistant", "content": tool_result.result})
            return formatted_answer

        return handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            messages=self.messages,
            step_callback=self.step_callback,
            show_logs=self._show_logs,
        )

    def _invoke_step_callback(
        self, formatted_answer: AgentAction | AgentFinish
    ) -> None:
        """Invoke step callback.

        Args:
            formatted_answer: Current agent response.
        """
        if self.step_callback:
            self.step_callback(formatted_answer)

    def _append_message(
        self, text: str, role: Literal["user", "assistant", "system"] = "assistant"
    ) -> None:
        """Add message to conversation history.

        Args:
            text: Message content.
            role: Message role (default: assistant).
        """
        self.messages.append(format_message_for_llm(text, role=role))

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

        future = crewai_event_bus.emit(
            self.agent,
            AgentLogsExecutionEvent(
                agent_role=self.agent.role,
                formatted_answer=formatted_answer,
                verbose=self.agent.verbose
                or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
            ),
        )

        if future is not None:
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to show logs for agent execution event: {e}")

    def _handle_crew_training_output(
        self, result: AgentFinish, human_feedback: str | None = None
    ) -> None:
        """Save training data.

        Args:
            result: Agent's final output.
            human_feedback: Optional feedback from human.
        """
        agent_id = str(self.agent.id)
        train_iteration = (
            getattr(self.crew, "_train_iteration", None) if self.crew else None
        )

        if train_iteration is None or not isinstance(train_iteration, int):
            if self.agent.verbose:
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
                if self.agent.verbose:
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

        Args:
            formatted_answer: Initial agent result.

        Returns:
            Final answer after feedback.
        """
        output_str = (
            formatted_answer.output
            if isinstance(formatted_answer.output, str)
            else formatted_answer.output.model_dump_json()
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
        """Process training feedback.

        Args:
            initial_answer: Initial agent output.
            feedback: Training feedback.

        Returns:
            Improved answer.
        """
        self._handle_crew_training_output(initial_answer, feedback)
        self.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )
        improved_answer = self._invoke_loop()
        self._handle_crew_training_output(improved_answer)
        self.ask_for_human_input = False
        return improved_answer

    def _handle_regular_feedback(
        self, current_answer: AgentFinish, initial_feedback: str
    ) -> AgentFinish:
        """Process regular feedback iteratively.

        Args:
            current_answer: Current agent output.
            initial_feedback: Initial user feedback.

        Returns:
            Final answer after iterations.
        """
        feedback = initial_feedback
        answer = current_answer

        while self.ask_for_human_input:
            # If the user provides a blank response, assume they are happy with the result
            if feedback.strip() == "":
                self.ask_for_human_input = False
            else:
                answer = self._process_feedback_iteration(feedback)
                output_str = (
                    answer.output
                    if isinstance(answer.output, str)
                    else answer.output.model_dump_json()
                )
                feedback = self._ask_human_input(output_str)

        return answer

    def _process_feedback_iteration(self, feedback: str) -> AgentFinish:
        """Process single feedback iteration.

        Args:
            feedback: User feedback.

        Returns:
            Updated agent response.
        """
        self.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )
        return self._invoke_loop()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for BaseClient Protocol.

        This allows the Protocol to be used in Pydantic models without
        requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()
