"""StepExecutor: Isolated executor for a single plan step.

Implements a bounded ReAct loop scoped to ONE todo item. The tool execution
machinery (native function calling, text-parsed tools, caching, hooks) lives
here — moved from AgentExecutor so the outer Plan-and-Execute loop stays clean.

Based on PLAN-AND-ACT (Section 3.2): The Executor translates high-level plan
steps into concrete environment actions.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import json
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.utilities.agent_utils import (
    convert_tools_to_openai_schema,
    enforce_rpm_limit,
    extract_tool_call_info,
    format_message_for_llm,
    process_llm_response,
    track_delegation_if_needed,
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
        self._use_native_tools = self._check_native_tool_support()
        self._openai_tools: list[dict[str, Any]] = []
        self._available_functions: dict[str, Callable[..., Any]] = {}
        if self._use_native_tools:
            self._setup_native_tools()

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
            tools_section = f"\n\nAvailable tools: {tool_names}"
            tools_section += "\n\nTo use a tool, respond with:\nThought: <your reasoning>\nAction: <tool_name>\nAction Input: <input>"
            tools_section += "\n\nWhen you have the final answer, respond with:\nThought: <your reasoning>\nFinal Answer: <your answer>"

        return f"""You are {role}. {backstory}

Your goal: {goal}

You are executing a specific step in a multi-step plan. Focus ONLY on completing
the current step. Do not plan ahead or worry about future steps.

Before acting, briefly reason about what you need to do and which approach
or tool would be most helpful for this specific step.{tools_section}"""

    def _build_user_prompt(self, todo: TodoItem, context: StepExecutionContext) -> str:
        """Build the user prompt for this specific step."""
        parts: list[str] = []

        parts.append(f"## Current Step\n{todo.description}")

        if todo.tool_to_use:
            parts.append(f"\nSuggested tool: {todo.tool_to_use}")

        # Include dependency results (final results only, no traces)
        if context.dependency_results:
            parts.append("\n## Context from previous steps:")
            for step_num, result in sorted(context.dependency_results.items()):
                parts.append(f"Step {step_num} result: {result}")

        parts.append("\nComplete this step and provide your result.")

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
        for iteration in range(_MAX_STEP_ITERATIONS):
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
        if isinstance(answer, list) and answer and self._is_tool_call_list(answer):
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
        # Build assistant message with tool calls
        tool_calls_to_report: list[dict[str, Any]] = []
        for tool_call in tool_calls:
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
            # Preserve raw parts for Gemini compatibility
            if all(type(tc).__qualname__ == "Part" for tc in tool_calls):
                assistant_message["raw_tool_call_parts"] = list(tool_calls)
            messages.append(assistant_message)

        # Execute each tool call
        final_answer: str | None = None
        for tool_call in tool_calls:
            info = extract_tool_call_info(tool_call)
            if not info:
                continue

            call_id, func_name, func_args = info
            tool_calls_made.append(func_name)

            # Parse arguments
            if isinstance(func_args, str):
                try:
                    args_dict = json.loads(func_args)
                except json.JSONDecodeError:
                    args_dict = {}
            else:
                args_dict = func_args

            agent_key = (
                getattr(self.agent, "key", "unknown") if self.agent else "unknown"
            )

            # Find original tool for cache_function and result_as_answer
            original_tool = None
            for tool in self.original_tools:
                if sanitize_tool_name(tool.name) == func_name:
                    original_tool = tool
                    break

            # Check max usage count
            max_usage_reached = False
            if (
                original_tool
                and original_tool.max_usage_count is not None
                and original_tool.current_usage_count >= original_tool.max_usage_count
            ):
                max_usage_reached = True

            # Check cache
            from_cache = False
            input_str = json.dumps(args_dict) if args_dict else ""
            result = "Tool not found"

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

            # Emit tool started event
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

            # Find structured tool for hooks
            structured_tool: CrewStructuredTool | None = None
            for structured in self.tools or []:
                if sanitize_tool_name(structured.name) == func_name:
                    structured_tool = structured
                    break

            # Before hooks
            hook_blocked = False
            before_hook_context = ToolCallHookContext(
                tool_name=func_name,
                tool_input=args_dict,
                tool=structured_tool,  # type: ignore[arg-type]
                agent=self.agent,
                task=self.task,
                crew=self.crew,
            )
            try:
                for hook in get_before_tool_call_hooks():
                    if hook(before_hook_context) is False:
                        hook_blocked = True
                        break
            except Exception:
                pass

            if hook_blocked:
                result = f"Tool execution blocked by hook. Tool: {func_name}"
            elif not from_cache and not max_usage_reached:
                if func_name in self._available_functions:
                    try:
                        tool_func = self._available_functions[func_name]
                        raw_result = tool_func(**args_dict)

                        # Cache result
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
                result = (
                    f"Tool '{func_name}' has reached its usage limit of "
                    f"{original_tool.max_usage_count} times and cannot be used anymore."
                )

            # After hooks
            after_hook_context = ToolCallHookContext(
                tool_name=func_name,
                tool_input=args_dict,
                tool=structured_tool,  # type: ignore[arg-type]
                agent=self.agent,
                task=self.task,
                crew=self.crew,
                tool_result=result,
            )
            try:
                for after_hook in get_after_tool_call_hooks():
                    hook_result = after_hook(after_hook_context)
                    if hook_result is not None:
                        result = hook_result
                        after_hook_context.tool_result = result
            except Exception:
                pass

            # Emit tool finished event
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
            messages.append(tool_message)

            if self.agent and self.agent.verbose:
                cache_info = " (from cache)" if from_cache else ""
                self._printer.print(
                    content=f"Tool {func_name} executed with result{cache_info}: {result[:200]}...",
                    color="green",
                )

            # Check result_as_answer
            if (
                original_tool
                and hasattr(original_tool, "result_as_answer")
                and original_tool.result_as_answer
            ):
                final_answer = result

        if final_answer is not None:
            return final_answer

        return None  # Continue looping

    def _force_final_answer(self, messages: list[LLMMessage]) -> str:
        """Force the LLM to provide a final answer when max iterations reached."""
        force_prompt = (
            "You have used the maximum number of tool calls for this step. "
            "Based on the information gathered so far, provide your final answer now."
        )
        if not self._use_native_tools:
            force_prompt += "\n\nFinal Answer: "

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
        except Exception:
            pass

        return "Step could not be completed within the iteration limit."

    # ------------------------------------------------------------------
    # Internal: Native tool support
    # ------------------------------------------------------------------

    def _check_native_tool_support(self) -> bool:
        """Check if LLM supports native function calling."""
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
        """Check if a response is a list of tool calls."""
        if not response:
            return False
        first_item = response[0]
        # OpenAI-style
        if hasattr(first_item, "function") or (
            isinstance(first_item, dict) and "function" in first_item
        ):
            return True
        # Anthropic-style (ToolUseBlock)
        if (
            hasattr(first_item, "type")
            and getattr(first_item, "type", None) == "tool_use"
        ):
            return True
        if hasattr(first_item, "name") and hasattr(first_item, "input"):
            return True
        # Bedrock-style
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
