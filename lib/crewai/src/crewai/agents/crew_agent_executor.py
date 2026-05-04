# mypy: disable-error-code="union-attr,arg-type"
"""Agent executor for crew AI agents.

Handles agent execution flow including LLM interactions, tool execution,
and memory management.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars
import inspect
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)
from pydantic.functional_serializers import PlainSerializer

from crewai.agents.agent_builder.base_agent import _serialize_llm_ref, _validate_llm_ref
from crewai.agents.agent_builder.base_agent_executor import BaseAgentExecutor
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.core.providers.human_input import ExecutorContext, get_provider
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
from crewai.types.callback import SerializableCallable
from crewai.utilities.agent_utils import (
    _llm_stop_words_applied,
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
    parse_tool_call_args,
    process_llm_response,
    track_delegation_if_needed,
)
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.file_store import aget_all_files, get_all_files
from crewai.utilities.i18n import I18N_DEFAULT
from crewai.utilities.printer import PRINTER
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.tool_utils import (
    aexecute_tool_and_check_finality,
    execute_tool_and_check_finality,
)
from crewai.utilities.training_handler import CrewTrainingHandler


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.tools.tool_types import ToolResult
    from crewai.utilities.prompts import StandardPromptResult, SystemPromptResult
    from crewai.utilities.types import LLMMessage


class CrewAgentExecutor(BaseAgentExecutor):
    """Executor for crew agents.

    Manages the execution lifecycle of an agent including prompt formatting,
    LLM interactions, tool execution, and feedback handling.
    """

    executor_type: Literal["crew"] = "crew"
    llm: Annotated[
        BaseLLM | str | None,
        BeforeValidator(_validate_llm_ref),
        PlainSerializer(_serialize_llm_ref, return_type=dict | None, when_used="json"),
    ] = Field(default=None)
    prompt: SystemPromptResult | StandardPromptResult | None = Field(default=None)
    tools: list[CrewStructuredTool] = Field(default_factory=list)
    tools_names: str = Field(default="")
    stop: list[str] = Field(
        default_factory=list, validation_alias=AliasChoices("stop", "stop_words")
    )
    tools_description: str = Field(default="")
    tools_handler: ToolsHandler | None = Field(default=None)
    step_callback: SerializableCallable | None = Field(default=None, exclude=True)
    original_tools: list[BaseTool] = Field(default_factory=list)
    function_calling_llm: Annotated[
        BaseLLM | str | None,
        BeforeValidator(_validate_llm_ref),
        PlainSerializer(_serialize_llm_ref, return_type=dict | None, when_used="json"),
    ] = Field(default=None)
    respect_context_window: bool = Field(default=False)
    request_within_rpm_limit: SerializableCallable | None = Field(
        default=None, exclude=True
    )
    callbacks: list[TokenCalcHandler] = Field(default_factory=list, exclude=True)
    response_model: type[BaseModel] | None = Field(default=None, exclude=True)
    ask_for_human_input: bool = Field(default=False)
    log_error_after: int = Field(default=3)
    before_llm_call_hooks: list[SerializableCallable] = Field(
        default_factory=list, exclude=True
    )
    after_llm_call_hooks: list[SerializableCallable] = Field(
        default_factory=list, exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.before_llm_call_hooks:
            self.before_llm_call_hooks.extend(get_before_llm_call_hooks())
        if not self.after_llm_call_hooks:
            self.after_llm_call_hooks.extend(get_after_llm_call_hooks())

    @property
    def use_stop_words(self) -> bool:
        """Check to determine if stop words are being used.

        Returns:
            bool: True if tool should be used or not.
        """
        from crewai.llms.base_llm import BaseLLM

        return (
            self.llm.supports_stop_words() if isinstance(self.llm, BaseLLM) else False
        )

    def _setup_messages(self, inputs: dict[str, Any]) -> None:
        """Set up messages for the agent execution.

        Args:
            inputs: Input dictionary containing prompt variables.
        """
        provider = get_provider()
        if provider.setup_messages(cast(ExecutorContext, cast(object, self))):
            return

        if self.prompt is not None and "system" in self.prompt:
            system_prompt = self._format_prompt(
                cast(str, self.prompt.get("system", "")), inputs
            )
            user_prompt = self._format_prompt(
                cast(str, self.prompt.get("user", "")), inputs
            )
            self.messages.append(format_message_for_llm(system_prompt, role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        elif self.prompt is not None:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(format_message_for_llm(user_prompt))

        provider.post_setup_messages(cast(ExecutorContext, cast(object, self)))

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with given inputs.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        if self._resuming:
            self._resuming = False
        else:
            self.messages = []
            self.iterations = 0
            self._setup_messages(inputs)
            self._inject_multimodal_files(inputs)

        self._show_start_logs()

        self.ask_for_human_input = bool(inputs.get("ask_for_human_input", False))

        with _llm_stop_words_applied(self.llm, self):
            try:
                formatted_answer = self._invoke_loop()
            except AssertionError:
                if self.agent.verbose:
                    PRINTER.print(
                        content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                        color="red",
                    )
                raise
            except Exception as e:
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
                raise

            if self.ask_for_human_input:
                formatted_answer = self._handle_human_feedback(formatted_answer)

        self._save_to_memory(formatted_answer)
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
        use_native_tools = (
            hasattr(self.llm, "supports_function_calling")
            and callable(getattr(self.llm, "supports_function_calling", None))
            and self.llm.supports_function_calling()
            and self.original_tools
        )

        if use_native_tools:
            return self._invoke_loop_native_tools()

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
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    break

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
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
                        answer_str = (
                            answer.model_dump_json()
                            if isinstance(answer, BaseModel)
                            else str(answer)
                        )
                        formatted_answer = process_llm_response(
                            answer_str, self.use_stop_words
                        )  # type: ignore[assignment]
                else:
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

                    tool_result = execute_tool_and_check_finality(
                        agent_action=formatted_answer,
                        fingerprint_context=fingerprint_context,
                        tools=self.tools,
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

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text)

            except OutputParserError as e:
                formatted_answer = handle_output_parser_exception(  # type: ignore[assignment]
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=PRINTER,
                    verbose=self.agent.verbose,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
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

    def _invoke_loop_native_tools(self) -> AgentFinish:
        """Execute agent loop using native function calling.

        This method uses the LLM's native tool/function calling capability
        instead of the text-based ReAct pattern. The LLM directly returns
        structured tool calls which are executed and results fed back.

        Returns:
            Final answer from the agent.
        """
        if not self.original_tools:
            return self._invoke_loop_native_no_tools()

        openai_tools, available_functions, self._tool_name_mapping = (
            convert_tools_to_openai_schema(self.original_tools)
        )

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )

                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    if tool_finish is not None:
                        return tool_finish
                    continue

                if isinstance(answer, str):
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=answer,
                    )
                    self._invoke_step_callback(formatted_answer)
                    self._append_message(answer)
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

                formatted_answer = AgentFinish(
                    thought="",
                    output=str(answer),
                    text=str(answer),
                )
                self._invoke_step_callback(formatted_answer)
                self._append_message(str(answer))
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
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
            llm=cast("BaseLLM", self.llm),
            messages=self.messages,
            callbacks=self.callbacks,
            printer=PRINTER,
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
        if hasattr(first_item, "function") or (
            isinstance(first_item, dict) and "function" in first_item
        ):
            return True
        if (
            hasattr(first_item, "type")
            and getattr(first_item, "type", None) == "tool_use"
        ):
            return True
        if hasattr(first_item, "name") and hasattr(first_item, "input"):
            return True
        if (
            isinstance(first_item, dict)
            and "name" in first_item
            and "input" in first_item
        ):
            return True
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
        if not tool_calls:
            return None

        parsed_calls = [
            parsed
            for tool_call in tool_calls
            if (parsed := self._parse_native_tool_call(tool_call)) is not None
        ]
        if not parsed_calls:
            return None

        original_tools_by_name: dict[str, Any] = dict(self._tool_name_mapping)

        if len(parsed_calls) > 1:
            has_result_as_answer_in_batch = any(
                bool(
                    original_tools_by_name.get(func_name)
                    and getattr(
                        original_tools_by_name.get(func_name), "result_as_answer", False
                    )
                )
                for _, func_name, _ in parsed_calls
            )
            has_max_usage_count_in_batch = any(
                bool(
                    original_tools_by_name.get(func_name)
                    and getattr(
                        original_tools_by_name.get(func_name),
                        "max_usage_count",
                        None,
                    )
                    is not None
                )
                for _, func_name, _ in parsed_calls
            )

            if has_result_as_answer_in_batch or has_max_usage_count_in_batch:
                logger.debug(
                    "Skipping parallel native execution because batch includes result_as_answer or max_usage_count tool"
                )
            else:
                execution_plan: list[
                    tuple[str, str, str | dict[str, Any], Any | None]
                ] = []
                for call_id, func_name, func_args in parsed_calls:
                    original_tool = original_tools_by_name.get(func_name)
                    execution_plan.append(
                        (call_id, func_name, func_args, original_tool)
                    )

                self._append_assistant_tool_calls_message(
                    [
                        (call_id, func_name, func_args)
                        for call_id, func_name, func_args, _ in execution_plan
                    ]
                )

                max_workers = min(8, len(execution_plan))
                ordered_results: list[dict[str, Any] | None] = [None] * len(
                    execution_plan
                )
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(
                            contextvars.copy_context().run,
                            self._execute_single_native_tool_call,
                            call_id=call_id,
                            func_name=func_name,
                            func_args=func_args,
                            available_functions=available_functions,
                            original_tool=original_tool,
                            should_execute=True,
                        ): idx
                        for idx, (
                            call_id,
                            func_name,
                            func_args,
                            original_tool,
                        ) in enumerate(execution_plan)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        ordered_results[idx] = future.result()

                for execution_result in ordered_results:
                    if not execution_result:
                        continue
                    tool_finish = self._append_tool_result_and_check_finality(
                        execution_result
                    )
                    if tool_finish:
                        return tool_finish

                reasoning_prompt = I18N_DEFAULT.slice("post_tool_reasoning")
                reasoning_message: LLMMessage = {
                    "role": "user",
                    "content": reasoning_prompt,
                }
                self.messages.append(reasoning_message)
                return None

        call_id, func_name, func_args = parsed_calls[0]
        self._append_assistant_tool_calls_message([(call_id, func_name, func_args)])

        execution_result = self._execute_single_native_tool_call(
            call_id=call_id,
            func_name=func_name,
            func_args=func_args,
            available_functions=available_functions,
            original_tool=original_tools_by_name.get(func_name),
            should_execute=True,
        )
        tool_finish = self._append_tool_result_and_check_finality(execution_result)
        if tool_finish:
            return tool_finish

        reasoning_prompt = I18N_DEFAULT.slice("post_tool_reasoning")
        reasoning_message = {
            "role": "user",
            "content": reasoning_prompt,
        }
        self.messages.append(reasoning_message)
        return None

    def _parse_native_tool_call(
        self, tool_call: Any
    ) -> tuple[str, str, str | dict[str, Any]] | None:
        if hasattr(tool_call, "function"):
            call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
            func_name = sanitize_tool_name(tool_call.function.name)
            return call_id, func_name, tool_call.function.arguments
        if hasattr(tool_call, "function_call") and tool_call.function_call:
            call_id = f"call_{id(tool_call)}"
            func_name = sanitize_tool_name(tool_call.function_call.name)
            func_args = (
                dict(tool_call.function_call.args)
                if tool_call.function_call.args
                else {}
            )
            return call_id, func_name, func_args
        if hasattr(tool_call, "name") and hasattr(tool_call, "input"):
            call_id = getattr(tool_call, "id", f"call_{id(tool_call)}")
            func_name = sanitize_tool_name(tool_call.name)
            return call_id, func_name, tool_call.input
        if isinstance(tool_call, dict):
            call_id = (
                tool_call.get("id")
                or tool_call.get("toolUseId")
                or f"call_{id(tool_call)}"
            )
            func_info = tool_call.get("function", {})
            func_name = sanitize_tool_name(
                func_info.get("name", "") or tool_call.get("name", "")
            )
            func_args = func_info.get("arguments") or tool_call.get("input", {})
            return call_id, func_name, func_args
        return None

    def _append_assistant_tool_calls_message(
        self,
        parsed_calls: list[tuple[str, str, str | dict[str, Any]]],
    ) -> None:
        import json

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
                for call_id, func_name, func_args in parsed_calls
            ],
        }
        self.messages.append(assistant_message)

    def _execute_single_native_tool_call(
        self,
        *,
        call_id: str,
        func_name: str,
        func_args: str | dict[str, Any],
        available_functions: dict[str, Callable[..., Any]],
        original_tool: Any | None = None,
        should_execute: bool = True,
    ) -> dict[str, Any]:
        from datetime import datetime
        import json

        from crewai.events.types.tool_usage_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        args_dict, parse_error = parse_tool_call_args(
            func_args, func_name, call_id, original_tool
        )
        if parse_error is not None:
            return parse_error

        if original_tool is None:
            for tool in self.original_tools or []:
                if sanitize_tool_name(tool.name) == func_name:
                    original_tool = tool
                    break

        max_usage_reached = False
        if not should_execute and original_tool:
            max_usage_reached = True
        elif (
            should_execute
            and original_tool
            and (max_count := getattr(original_tool, "max_usage_count", None))
            is not None
            and getattr(original_tool, "current_usage_count", 0) >= max_count
        ):
            max_usage_reached = True

        from_cache = False
        result: str = "Tool not found"
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

        agent_key = getattr(self.agent, "key", "unknown") if self.agent else "unknown"
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

        track_delegation_if_needed(func_name, args_dict or {}, self.task)

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
            tool_input=args_dict or {},
            tool=structured_tool,
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
                PRINTER.print(
                    content=f"Error in before_tool_call hook: {hook_error}",
                    color="red",
                )

        if hook_blocked:
            result = f"Tool execution blocked by hook. Tool: {func_name}"
        elif max_usage_reached and original_tool:
            result = f"Tool '{func_name}' has reached its usage limit of {original_tool.max_usage_count} times and cannot be used anymore."
        elif not from_cache and func_name in available_functions:
            try:
                raw_result = available_functions[func_name](**(args_dict or {}))

                if self.tools_handler and self.tools_handler.cache:
                    should_cache = True
                    if (
                        original_tool
                        and hasattr(original_tool, "cache_function")
                        and callable(original_tool.cache_function)
                    ):
                        should_cache = original_tool.cache_function(
                            args_dict or {}, raw_result
                        )
                    if should_cache:
                        self.tools_handler.cache.add(
                            tool=func_name, input=input_str, output=raw_result
                        )

                result = (
                    str(raw_result) if not isinstance(raw_result, str) else raw_result
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
                error_event_emitted = True

        after_hook_context = ToolCallHookContext(
            tool_name=func_name,
            tool_input=args_dict or {},
            tool=structured_tool,
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
                PRINTER.print(
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

    def _append_tool_result_and_check_finality(
        self, execution_result: dict[str, Any]
    ) -> AgentFinish | None:
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
        self.messages.append(tool_message)

        if self.agent and self.agent.verbose:
            cache_info = " (from cache)" if from_cache else ""
            PRINTER.print(
                content=f"Tool {func_name} executed with result{cache_info}: {result[:200]}...",
                color="green",
            )

        if (
            original_tool
            and hasattr(original_tool, "result_as_answer")
            and original_tool.result_as_answer
        ):
            return AgentFinish(
                thought="Tool result is the final answer",
                output=result,
                text=result,
            )
        return None

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent asynchronously with given inputs.

        Args:
            inputs: Input dictionary containing prompt variables.

        Returns:
            Dictionary with agent output.
        """
        if self._resuming:
            self._resuming = False
        else:
            self.messages = []
            self.iterations = 0
            self._setup_messages(inputs)
            await self._ainject_multimodal_files(inputs)

        self._show_start_logs()

        self.ask_for_human_input = bool(inputs.get("ask_for_human_input", False))

        with _llm_stop_words_applied(self.llm, self):
            try:
                formatted_answer = await self._ainvoke_loop()
            except AssertionError:
                if self.agent.verbose:
                    PRINTER.print(
                        content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                        color="red",
                    )
                raise
            except Exception as e:
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
                raise

            if self.ask_for_human_input:
                formatted_answer = await self._ahandle_human_feedback(formatted_answer)

        self._save_to_memory(formatted_answer)
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
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    break

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = await aget_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
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
                        answer_str = (
                            answer.model_dump_json()
                            if isinstance(answer, BaseModel)
                            else str(answer)
                        )
                        formatted_answer = process_llm_response(
                            answer_str, self.use_stop_words
                        )  # type: ignore[assignment]
                else:
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

                await self._ainvoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text)

            except OutputParserError as e:
                formatted_answer = handle_output_parser_exception(  # type: ignore[assignment]
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=PRINTER,
                    verbose=self.agent.verbose,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
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

        openai_tools, available_functions, self._tool_name_mapping = (
            convert_tools_to_openai_schema(self.original_tools)
        )

        while True:
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        None,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    self._show_logs(formatted_answer)
                    return formatted_answer

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = await aget_llm_response(
                    llm=cast("BaseLLM", self.llm),
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=PRINTER,
                    tools=openai_tools,
                    available_functions=None,
                    from_task=self.task,
                    from_agent=self.agent,
                    response_model=self.response_model,
                    executor_context=self,
                    verbose=self.agent.verbose,
                )
                if (
                    isinstance(answer, list)
                    and answer
                    and self._is_tool_call_list(answer)
                ):
                    tool_finish = self._handle_native_tool_calls(
                        answer, available_functions
                    )
                    if tool_finish is not None:
                        return tool_finish
                    continue

                if isinstance(answer, str):
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=answer,
                    )
                    await self._ainvoke_step_callback(formatted_answer)
                    self._append_message(answer)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                if isinstance(answer, BaseModel):
                    output_json = answer.model_dump_json()
                    formatted_answer = AgentFinish(
                        thought="",
                        output=answer,
                        text=output_json,
                    )
                    await self._ainvoke_step_callback(formatted_answer)
                    self._append_message(output_json)
                    self._show_logs(formatted_answer)
                    return formatted_answer

                formatted_answer = AgentFinish(
                    thought="",
                    output=str(answer),
                    text=str(answer),
                )
                await self._ainvoke_step_callback(formatted_answer)
                self._append_message(str(answer))
                self._show_logs(formatted_answer)
                return formatted_answer

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=PRINTER,
                        messages=self.messages,
                        llm=cast("BaseLLM", self.llm),
                        callbacks=self.callbacks,
                        verbose=self.agent.verbose,
                    )
                    continue
                handle_unknown_error(PRINTER, e, verbose=self.agent.verbose)
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
            llm=cast("BaseLLM", self.llm),
            messages=self.messages,
            callbacks=self.callbacks,
            printer=PRINTER,
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
        add_image_tool = I18N_DEFAULT.tools("add_image")
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
        """Invoke step callback (sync context).

        Args:
            formatted_answer: Current agent response.
        """
        if self.step_callback:
            cb_result = self.step_callback(formatted_answer)
            if inspect.iscoroutine(cb_result):
                asyncio.run(cb_result)

    async def _ainvoke_step_callback(
        self, formatted_answer: AgentAction | AgentFinish
    ) -> None:
        """Invoke step callback (async context).

        Args:
            formatted_answer: Current agent response.
        """
        if self.step_callback:
            cb_result = self.step_callback(formatted_answer)
            if inspect.iscoroutine(cb_result):
                await cb_result

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
                PRINTER.print(
                    content="Invalid or missing train iteration. Cannot save training data.",
                    color="red",
                )
            return

        training_handler = CrewTrainingHandler(TRAINING_DATA_FILE)
        training_data = training_handler.load() or {}

        agent_training_data = training_data.get(agent_id, {})

        if human_feedback is not None:
            agent_training_data[train_iteration] = {
                "initial_output": result.output,
                "human_feedback": human_feedback,
            }
        else:
            if train_iteration in agent_training_data:
                agent_training_data[train_iteration]["improved_output"] = result.output
            else:
                if self.agent.verbose:
                    PRINTER.print(
                        content=(
                            f"No existing training data for agent {agent_id} and iteration "
                            f"{train_iteration}. Cannot save improved output."
                        ),
                        color="red",
                    )
                return

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
        """Process human feedback via the configured provider.

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
        """Process human feedback asynchronously via the configured provider.

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

    def _format_feedback_message(self, feedback: str) -> LLMMessage:
        """Format feedback as a message for the LLM.

        Args:
            feedback: User feedback string.

        Returns:
            Formatted message dict.
        """
        return format_message_for_llm(
            I18N_DEFAULT.slice("feedback_instructions").format(feedback=feedback)
        )
