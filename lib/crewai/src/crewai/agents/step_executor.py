"""StepExecutor: Isolated executor for a single plan step.

Implements a bounded ReAct loop scoped to ONE todo item. The tool execution
machinery (native function calling, text-parsed tools, caching, hooks) lives
here — moved from AgentExecutor so the outer loop stays clean.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
)
from crewai.utilities.agent_utils import (
    build_tool_calls_assistant_message,
    check_native_tool_support,
    enforce_rpm_limit,
    execute_single_native_tool_call,
    format_message_for_llm,
    is_tool_call_list,
    process_llm_response,
    setup_native_tools,
)
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.planning_types import TodoItem
from crewai.utilities.printer import Printer
from crewai.utilities.step_execution_context import StepExecutionContext, StepResult
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool


# Maximum number of tool-call iterations within a single step
_MAX_STEP_ITERATIONS: int = 10


class StepExecutor:
    """Executes a SINGLE todo item in isolation using a bounded ReAct loop.

    The StepExecutor owns its own message list per invocation. It never reads
    or writes the AgentExecutor's state. Results flow back via StepResult.

    The internal loop:
        1. Build messages from todo + context
        2. Call LLM (with or without native tools)
        3. If tool call → execute tool, append result, loop back to 2
        4. If final answer → return StepResult
        5. If max iterations → force final answer

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
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[CrewStructuredTool],
        agent: Agent,
        original_tools: list[BaseTool] | None = None,
        tools_handler: ToolsHandler | None = None,
        task: Task | None = None,
        crew: Crew | None = None,
        function_calling_llm: BaseLLM | Any | None = None,
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
        self._printer: Printer = Printer()

        # Native tool support — set up once
        self._use_native_tools = check_native_tool_support(self.llm, self.original_tools)
        self._openai_tools: list[dict[str, Any]] = []
        self._available_functions: dict[str, Callable[..., Any]] = {}
        if self._use_native_tools and self.original_tools:
            self._openai_tools, self._available_functions = setup_native_tools(
                self.original_tools
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, todo: TodoItem, context: StepExecutionContext) -> StepResult:
        """Execute a single todo item in isolation.

        Builds a fresh message list, runs a bounded ReAct loop, and returns
        the result. Never touches external state.

        Args:
            todo: The todo item to execute.
            context: Immutable context with task info and dependency results.

        Returns:
            StepResult with the outcome.
        """
        start_time = time.monotonic()
        tool_calls_made: list[str] = []

        try:
            messages = self._build_isolated_messages(todo, context)
            result_text = self._run_react_loop(todo, messages, tool_calls_made)

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

        messages: list[LLMMessage] = [
            format_message_for_llm(system_prompt, role="system"),
            format_message_for_llm(user_prompt, role="user"),
        ]
        return messages

    def _build_system_prompt(self) -> str:
        """Build the Executor's system prompt.

        Emphasizes: complete THIS step only. Do not plan ahead.
        Includes CoT reasoning instruction (per PLAN-AND-ACT Section 3.4).
        """
        role = self.agent.role if self.agent else "Assistant"
        goal = self.agent.goal if self.agent else "Complete tasks efficiently"
        backstory = getattr(self.agent, "backstory", "") or ""

        tools_section = ""
        if self.tools and not self._use_native_tools:
            tool_names = ", ".join(sanitize_tool_name(t.name) for t in self.tools)
            tools_section = self._i18n.retrieve(
                "planning", "step_executor_tools_section"
            ).format(tool_names=tool_names)

        return self._i18n.retrieve("planning", "step_executor_system_prompt").format(
            role=role,
            backstory=backstory,
            goal=goal,
            tools_section=tools_section,
        )

    def _build_user_prompt(self, todo: TodoItem, context: StepExecutionContext) -> str:
        """Build the user prompt for this specific step."""
        parts: list[str] = []

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
    # Internal: Bounded ReAct loop
    # ------------------------------------------------------------------

    def _run_react_loop(
        self,
        todo: TodoItem,
        messages: list[LLMMessage],
        tool_calls_made: list[str],
    ) -> str:
        """Run a bounded ReAct loop for a single step.

        Returns the final answer text.
        """
        for _iteration in range(_MAX_STEP_ITERATIONS):
            enforce_rpm_limit(self.request_within_rpm_limit)

            if self._use_native_tools:
                result = self._native_tool_iteration(messages, tool_calls_made)
            else:
                result = self._text_parsed_iteration(messages, tool_calls_made)

            if result is not None:
                # Got a final answer
                return result

            # No final answer yet — loop continues with updated messages

        # Max iterations reached — force a final answer
        return self._force_final_answer(messages)

    def _text_parsed_iteration(
        self,
        messages: list[LLMMessage],
        tool_calls_made: list[str],
    ) -> str | None:
        """Single iteration using text-parsed tool calling.

        Returns final answer string if done, None to continue looping.
        """
        try:
            answer = self.llm.call(
                messages,
                callbacks=self.callbacks,
                from_task=self.task,
                from_agent=self.agent,
            )
        except Exception:
            raise

        if not answer:
            raise ValueError("Empty response from LLM")

        answer_str = str(answer)
        use_stop_words = self.llm.supports_stop_words() if self.llm else False
        formatted = process_llm_response(answer_str, use_stop_words)

        if isinstance(formatted, AgentFinish):
            return str(formatted.output)

        if isinstance(formatted, AgentAction):
            # Execute the tool
            tool_calls_made.append(formatted.tool)

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

            # Append observation to messages
            observation = f"Observation: {tool_result.result}"
            messages.append(
                format_message_for_llm(
                    formatted.text + f"\n{observation}",
                    role="assistant",
                )
            )

            if tool_result.result_as_answer:
                return str(tool_result.result)

            # Add reasoning prompt for next iteration
            reasoning_prompt = self._i18n.slice("post_tool_reasoning")
            messages.append(format_message_for_llm(reasoning_prompt, role="user"))

            return None  # Continue looping

        return answer_str  # Fallback: treat as final answer

    def _native_tool_iteration(
        self,
        messages: list[LLMMessage],
        tool_calls_made: list[str],
    ) -> str | None:
        """Single iteration using native function calling.

        Returns final answer string if done, None to continue looping.
        """
        try:
            answer = self.llm.call(
                messages,
                tools=self._openai_tools,
                callbacks=self.callbacks,
                from_task=self.task,
                from_agent=self.agent,
            )
        except Exception:
            raise

        if not answer:
            raise ValueError("Empty response from LLM")

        # Check if the response is a list of tool calls
        if isinstance(answer, list) and answer and is_tool_call_list(answer):
            return self._execute_native_tool_calls(answer, messages, tool_calls_made)

        # Text response — this is the final answer
        if isinstance(answer, str):
            return answer

        # BaseModel response
        if isinstance(answer, BaseModel):
            return answer.model_dump_json()

        return str(answer)

    def _execute_native_tool_calls(
        self,
        tool_calls: list[Any],
        messages: list[LLMMessage],
        tool_calls_made: list[str],
    ) -> str | None:
        """Execute a batch of native tool calls and append results to messages.

        Returns final answer string if a tool has result_as_answer, else None.
        """
        # Build and append assistant message with tool call reports
        assistant_message, _reports = build_tool_calls_assistant_message(tool_calls)
        if assistant_message:
            messages.append(assistant_message)

        # Execute each tool call via shared pipeline
        final_answer: str | None = None
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
                printer=self._printer,
                verbose=bool(self.agent and self.agent.verbose),
            )

            if call_result.func_name:
                tool_calls_made.append(call_result.func_name)

            if call_result.tool_message:
                messages.append(call_result.tool_message)

            if call_result.result_as_answer:
                final_answer = call_result.result

        if final_answer is not None:
            return final_answer

        return None  # Continue looping

    def _force_final_answer(self, messages: list[LLMMessage]) -> str:
        """Force the LLM to provide a final answer when max iterations reached."""
        force_prompt = self._i18n.retrieve(
            "planning", "step_executor_force_final_answer"
        )
        if not self._use_native_tools:
            force_prompt += self._i18n.retrieve(
                "planning", "step_executor_force_final_answer_suffix"
            )

        messages.append(format_message_for_llm(force_prompt, role="user"))

        try:
            answer = self.llm.call(
                messages,
                callbacks=self.callbacks,
                from_task=self.task,
                from_agent=self.agent,
            )
            if answer:
                answer_str = str(answer)
                # Try to extract just the final answer portion
                if "Final Answer:" in answer_str:
                    return answer_str.split("Final Answer:")[-1].strip()
                return answer_str
        except Exception:  # noqa: S110
            pass

        return self._i18n.retrieve("planning", "step_could_not_complete")
