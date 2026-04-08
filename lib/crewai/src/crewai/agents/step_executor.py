"""StepExecutor: Isolated executor for a single plan step.

Implements the direct-action execution pattern from Plan-and-Act
(arxiv 2503.09572): the Executor receives one step description,
makes a single LLM call, executes any tool call returned, and
returns the result immediately.

There is no inner loop. Recovery from failure (retry, replan) is
the responsibility of PlannerObserver and AgentExecutor — keeping
this class single-purpose and fast.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from crewai.agents.parser import AgentAction, AgentFinish
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.agent_utils import (
    build_tool_calls_assistant_message,
    check_native_tool_support,
    enforce_rpm_limit,
    execute_single_native_tool_call,
    extract_task_section,
    format_message_for_llm,
    is_tool_call_list,
    process_llm_response,
    setup_native_tools,
)
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.planning_types import TodoItem
from crewai.utilities.printer import PRINTER
from crewai.utilities.step_execution_context import StepExecutionContext, StepResult
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool


class StepExecutor:
    """Executes a SINGLE todo item using direct-action execution.

    The StepExecutor owns its own message list per invocation. It never reads
    or writes the AgentExecutor's state. Results flow back via StepResult.

    Execution pattern (per Plan-and-Act, arxiv 2503.09572):
        1. Build messages from todo + context
        2. Call LLM once (with or without native tools)
        3. If tool call → execute it → return tool result
        4. If text answer → return it directly
        No inner loop — recovery is PlannerObserver's responsibility.

    Args:
        llm: The language model to use for execution.
        tools: Structured tools available to the executor.
        agent: The agent instance (for role/goal/verbose/config).
        original_tools: Original BaseTool instances (needed for native tool schema).
        tools_handler: Optional tools handler for caching and delegation tracking.
        task: Optional task context.
        crew: Optional crew context.
        function_calling_llm: Optional separate LLM for function calling.
        request_within_rpm_limit: Optional RPM limit function.
        callbacks: Optional list of callbacks.
        i18n: Optional i18n instance.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[CrewStructuredTool],
        agent: BaseAgent,
        original_tools: list[BaseTool] | None = None,
        tools_handler: ToolsHandler | None = None,
        task: Task | None = None,
        crew: Crew | None = None,
        function_calling_llm: BaseLLM | None = None,
        request_within_rpm_limit: Callable[[], bool] | None = None,
        callbacks: list[Any] | None = None,
        i18n: I18N | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.agent = agent
        self.original_tools = original_tools or []
        self.tools_handler = tools_handler
        self.task = task
        self.crew = crew
        self.function_calling_llm = function_calling_llm
        self.request_within_rpm_limit = request_within_rpm_limit
        self.callbacks = callbacks or []
        self._i18n: I18N = i18n or get_i18n()

        # Native tool support — set up once
        self._use_native_tools = check_native_tool_support(
            self.llm, self.original_tools
        )
        self._openai_tools: list[dict[str, Any]] = []
        self._available_functions: dict[str, Callable[..., Any]] = {}
        if self._use_native_tools and self.original_tools:
            (
                self._openai_tools,
                self._available_functions,
                _,
            ) = setup_native_tools(self.original_tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        todo: TodoItem,
        context: StepExecutionContext,
        max_step_iterations: int = 15,
        step_timeout: int | None = None,
    ) -> StepResult:
        """Execute a single todo item using a multi-turn action loop.

        Enforces the RPM limit, builds a fresh message list, then iterates
        LLM call → tool execution → observation until the LLM signals it is
        done (text answer) or max_step_iterations is reached.  Never touches
        external AgentExecutor state.

        Args:
            todo: The todo item to execute.
            context: Immutable context with task info and dependency results.
            max_step_iterations: Maximum LLM iterations in the multi-turn loop.
            step_timeout: Maximum wall-clock seconds for this step. None = no limit.

        Returns:
            StepResult with the outcome.
        """
        start_time = time.monotonic()
        tool_calls_made: list[str] = []

        try:
            enforce_rpm_limit(self.request_within_rpm_limit)
            messages = self._build_isolated_messages(todo, context)

            if self._use_native_tools:
                result_text = self._execute_native(
                    messages,
                    tool_calls_made,
                    max_step_iterations=max_step_iterations,
                    step_timeout=step_timeout,
                    start_time=start_time,
                )
            else:
                result_text = self._execute_text_parsed(
                    messages,
                    tool_calls_made,
                    max_step_iterations=max_step_iterations,
                    step_timeout=step_timeout,
                    start_time=start_time,
                )
            self._validate_expected_tool_usage(todo, tool_calls_made)

            elapsed = time.monotonic() - start_time
            return StepResult(
                success=True,
                result=result_text,
                tool_calls_made=tool_calls_made,
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return StepResult(
                success=False,
                result="",
                error=str(e),
                tool_calls_made=tool_calls_made,
                execution_time=elapsed,
            )

    # ------------------------------------------------------------------
    # Internal: Message building
    # ------------------------------------------------------------------

    def _build_isolated_messages(
        self, todo: TodoItem, context: StepExecutionContext
    ) -> list[LLMMessage]:
        """Build a fresh message list for this step's execution.

        System prompt tells the LLM it is an Executor focused on one step.
        User prompt provides the step description, dependencies, and tools.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(todo, context)

        return [
            format_message_for_llm(system_prompt, role="system"),
            format_message_for_llm(user_prompt, role="user"),
        ]

    def _build_system_prompt(self) -> str:
        """Build the Executor's system prompt."""
        role = self.agent.role if self.agent else "Assistant"
        goal = self.agent.goal if self.agent else "Complete tasks efficiently"
        backstory = getattr(self.agent, "backstory", "") or ""

        tools_section = ""
        if self.tools and not self._use_native_tools:
            tool_names = ", ".join(sanitize_tool_name(t.name) for t in self.tools)
            tools_section = self._i18n.retrieve(
                "planning", "step_executor_tools_section"
            ).format(tool_names=tool_names)
        elif self.tools:
            tool_names = ", ".join(sanitize_tool_name(t.name) for t in self.tools)
            tools_section = f"\n\nAvailable tools: {tool_names}"

        return self._i18n.retrieve("planning", "step_executor_system_prompt").format(
            role=role,
            backstory=backstory,
            goal=goal,
            tools_section=tools_section,
        )

    def _build_user_prompt(self, todo: TodoItem, context: StepExecutionContext) -> str:
        """Build the user prompt for this specific step."""
        parts: list[str] = []

        # Include overall task context so the executor knows the full goal and
        # required output format/location — critical for knowing WHAT to produce.
        # We extract only the task body (not tool instructions or verification
        # sections) to avoid duplicating directives already in the system prompt.
        if context.task_description:
            task_section = extract_task_section(context.task_description)
            if task_section:
                parts.append(
                    self._i18n.retrieve(
                        "planning", "step_executor_task_context"
                    ).format(
                        task_context=task_section,
                    )
                )

        parts.append(
            self._i18n.retrieve("planning", "step_executor_user_prompt").format(
                step_description=todo.description,
            )
        )

        if todo.tool_to_use:
            parts.append(
                self._i18n.retrieve("planning", "step_executor_suggested_tool").format(
                    tool_to_use=todo.tool_to_use,
                )
            )

        # Include dependency results (final results only, no traces)
        if context.dependency_results:
            parts.append(
                self._i18n.retrieve("planning", "step_executor_context_header")
            )
            for step_num, result in sorted(context.dependency_results.items()):
                parts.append(
                    self._i18n.retrieve(
                        "planning", "step_executor_context_entry"
                    ).format(step_number=step_num, result=result)
                )

        parts.append(self._i18n.retrieve("planning", "step_executor_complete_step"))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal: Multi-turn execution loop
    # ------------------------------------------------------------------

    def _execute_text_parsed(
        self,
        messages: list[LLMMessage],
        tool_calls_made: list[str],
        max_step_iterations: int = 15,
        step_timeout: int | None = None,
        start_time: float | None = None,
    ) -> str:
        """Execute step using text-parsed tool calling with a multi-turn loop.

        Iterates LLM call → tool execution → observation until the LLM
        produces a Final Answer or max_step_iterations is reached.
        This allows the agent to: run a command, see the output, adjust its
        approach, and run another command — all within a single plan step.
        """
        use_stop_words = self.llm.supports_stop_words() if self.llm else False
        last_tool_result = ""

        for _ in range(max_step_iterations):
            # Check step timeout
            if step_timeout and start_time:
                elapsed = time.monotonic() - start_time
                if elapsed >= step_timeout:
                    return last_tool_result or f"Step timed out after {elapsed:.0f}s"
            answer = self.llm.call(
                messages,
                callbacks=self.callbacks,
                from_task=self.task,
                from_agent=self.agent,
            )

            if not answer:
                raise ValueError("Empty response from LLM")

            answer_str = str(answer)
            formatted = process_llm_response(answer_str, use_stop_words)

            if isinstance(formatted, AgentFinish):
                return str(formatted.output)

            if isinstance(formatted, AgentAction):
                tool_calls_made.append(formatted.tool)
                tool_result = self._execute_text_tool_with_events(formatted)
                last_tool_result = tool_result
                # Append the assistant's reasoning + action, then the observation.
                # _build_observation_message handles vision sentinels so the LLM
                # receives an image content block instead of raw base64 text.
                messages.append({"role": "assistant", "content": answer_str})
                messages.append(self._build_observation_message(tool_result))
                continue

            # Raw text response with no Final Answer marker — treat as done
            return answer_str

        # Max iterations reached — return the last tool result we accumulated
        return last_tool_result

    def _execute_text_tool_with_events(self, formatted: AgentAction) -> str:
        """Execute text-parsed tool calls with tool usage events."""
        args_dict = self._parse_tool_args(formatted.tool_input)
        agent_key = getattr(self.agent, "key", "unknown") if self.agent else "unknown"
        started_at = datetime.now()
        crewai_event_bus.emit(
            self,
            event=ToolUsageStartedEvent(
                tool_name=formatted.tool,
                tool_args=args_dict,
                from_agent=self.agent,
                from_task=self.task,
                agent_key=agent_key,
            ),
        )

        try:
            fingerprint_context = {}
            if (
                self.agent
                and hasattr(self.agent, "security_config")
                and hasattr(self.agent.security_config, "fingerprint")
            ):
                fingerprint_context = {
                    "agent_fingerprint": str(self.agent.security_config.fingerprint)
                }

            tool_result = execute_tool_and_check_finality(
                agent_action=formatted,
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
            crewai_event_bus.emit(
                self,
                event=ToolUsageErrorEvent(
                    tool_name=formatted.tool,
                    tool_args=args_dict,
                    from_agent=self.agent,
                    from_task=self.task,
                    agent_key=agent_key,
                    error=e,
                ),
            )
            raise

        crewai_event_bus.emit(
            self,
            event=ToolUsageFinishedEvent(
                output=str(tool_result.result),
                tool_name=formatted.tool,
                tool_args=args_dict,
                from_agent=self.agent,
                from_task=self.task,
                agent_key=agent_key,
                started_at=started_at,
                finished_at=datetime.now(),
            ),
        )
        return str(tool_result.result)

    def _parse_tool_args(self, tool_input: Any) -> dict[str, Any]:
        """Parse tool args from the parser output into a dict payload for events."""
        if isinstance(tool_input, dict):
            return tool_input
        if isinstance(tool_input, str):
            stripped_input = tool_input.strip()
            if not stripped_input:
                return {}
            try:
                parsed = json.loads(stripped_input)
                if isinstance(parsed, dict):
                    return parsed
                return {"input": parsed}
            except json.JSONDecodeError:
                return {"input": stripped_input}
        return {"input": str(tool_input)}

    # ------------------------------------------------------------------
    # Internal: Vision support
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vision_sentinel(raw: str) -> tuple[str, str] | None:
        """Parse a VISION_IMAGE sentinel into (media_type, base64_data), or None."""
        prefix = "VISION_IMAGE:"
        if not raw.startswith(prefix):
            return None
        rest = raw[len(prefix) :]
        sep = rest.find(":")
        if sep <= 0:
            return None
        return rest[:sep], rest[sep + 1 :]

    @staticmethod
    def _build_observation_message(tool_result: str) -> LLMMessage:
        """Build an observation message, converting vision sentinels to image blocks.

        When a tool returns a VISION_IMAGE sentinel (e.g. from read_image),
        we build a multimodal content block so the LLM can actually *see*
        the image rather than receiving a wall of base64 text.

        Uses the standard image_url / data-URI format so each LLM provider's
        SDK (OpenAI, LiteLLM, etc.) handles the provider-specific conversion.

        Format: ``VISION_IMAGE:<media_type>:<base64_data>``
        """
        parsed = StepExecutor._parse_vision_sentinel(tool_result)
        if parsed:
            media_type, b64_data = parsed
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Observation: Here is the image:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}",
                        },
                    },
                ],
            }
        return {"role": "user", "content": f"Observation: {tool_result}"}

    def _validate_expected_tool_usage(
        self,
        todo: TodoItem,
        tool_calls_made: list[str],
    ) -> None:
        """Fail step execution when a required tool is configured but not called."""
        expected_tool = getattr(todo, "tool_to_use", None)
        if not expected_tool:
            return
        expected_tool_name = sanitize_tool_name(expected_tool)
        available_tool_names = {
            sanitize_tool_name(tool.name)
            for tool in self.tools
            if getattr(tool, "name", "")
        } | set(self._available_functions.keys())
        if expected_tool_name not in available_tool_names:
            return
        called_names = {sanitize_tool_name(name) for name in tool_calls_made}
        if expected_tool_name not in called_names:
            raise ValueError(
                f"Expected tool '{expected_tool_name}' was not called "
                f"for step {todo.step_number}."
            )

    def _execute_native(
        self,
        messages: list[LLMMessage],
        tool_calls_made: list[str],
        max_step_iterations: int = 15,
        step_timeout: int | None = None,
        start_time: float | None = None,
    ) -> str:
        """Execute step using native function calling with a multi-turn loop.

        Iterates LLM call → tool execution → appended results until the LLM
        returns a text answer (no more tool calls) or max_step_iterations is
        reached.  This lets the agent run a shell command, observe the output,
        correct mistakes, and issue follow-up commands — all within one step.
        """
        accumulated_results: list[str] = []

        for _ in range(max_step_iterations):
            # Check step timeout
            if step_timeout and start_time:
                elapsed = time.monotonic() - start_time
                if elapsed >= step_timeout:
                    return (
                        "\n\n".join(accumulated_results)
                        if accumulated_results
                        else f"Step timed out after {elapsed:.0f}s"
                    )
            answer = self.llm.call(
                messages,
                tools=self._openai_tools,
                callbacks=self.callbacks,
                from_task=self.task,
                from_agent=self.agent,
            )

            if not answer:
                raise ValueError("Empty response from LLM")

            if isinstance(answer, BaseModel):
                return answer.model_dump_json()

            if isinstance(answer, list) and answer and is_tool_call_list(answer):
                # _execute_native_tool_calls appends assistant + tool messages
                # to `messages` as a side-effect, so the next LLM call will
                # see the full conversation history including tool outputs.
                result = self._execute_native_tool_calls(
                    answer, messages, tool_calls_made
                )
                accumulated_results.append(result)
                continue

            # Text answer → LLM decided the step is done
            return str(answer)

        # Max iterations reached — return everything we accumulated
        return "\n".join(filter(None, accumulated_results))

    def _execute_native_tool_calls(
        self,
        tool_calls: list[Any],
        messages: list[LLMMessage],
        tool_calls_made: list[str],
    ) -> str:
        """Execute a batch of native tool calls and return their results.

        Returns the result of the first tool marked result_as_answer if any,
        otherwise returns all tool results concatenated.
        """
        assistant_message, _reports = build_tool_calls_assistant_message(tool_calls)
        if assistant_message:
            messages.append(assistant_message)

        tool_results: list[str] = []
        for tool_call in tool_calls:
            call_result = execute_single_native_tool_call(
                tool_call,
                available_functions=self._available_functions,
                original_tools=self.original_tools,
                structured_tools=self.tools,
                tools_handler=self.tools_handler,
                agent=self.agent,
                task=self.task,
                crew=self.crew,
                event_source=self,
                printer=PRINTER,
                verbose=bool(self.agent and self.agent.verbose),
            )

            if call_result.func_name:
                tool_calls_made.append(call_result.func_name)

            if call_result.result_as_answer:
                return str(call_result.result)

            if call_result.tool_message:
                raw_content = call_result.tool_message.get("content", "")
                if isinstance(raw_content, str):
                    parsed = self._parse_vision_sentinel(raw_content)
                    if parsed:
                        media_type, b64_data = parsed
                        # Replace the sentinel with a standard image_url content block.
                        # Each provider's _format_messages handles conversion to
                        # its native format (e.g. Anthropic image blocks).
                        modified: LLMMessage = cast(
                            LLMMessage, dict(call_result.tool_message)
                        )
                        modified["content"] = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64_data}",
                                },
                            }
                        ]
                        messages.append(modified)
                        tool_results.append("[image]")
                    else:
                        messages.append(call_result.tool_message)
                        if raw_content:
                            tool_results.append(raw_content)
                else:
                    messages.append(call_result.tool_message)
                    if raw_content:
                        tool_results.append(str(raw_content))

        return "\n".join(tool_results) if tool_results else ""
