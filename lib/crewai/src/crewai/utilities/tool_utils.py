from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.security.fingerprint import Fingerprint
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.tools.tool_usage import ToolUsage, ToolUsageError
from crewai.utilities.i18n import I18N
from crewai.utilities.logger import Logger
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task


async def aexecute_tool_and_check_finality(
    agent_action: AgentAction,
    tools: list[CrewStructuredTool],
    i18n: I18N,
    agent_key: str | None = None,
    agent_role: str | None = None,
    tools_handler: ToolsHandler | None = None,
    task: Task | None = None,
    agent: Agent | BaseAgent | None = None,
    function_calling_llm: BaseLLM | LLM | None = None,
    fingerprint_context: dict[str, str] | None = None,
    crew: Crew | None = None,
) -> ToolResult:
    """Execute a tool asynchronously and check if the result should be a final answer.

    This is the async version of execute_tool_and_check_finality. It integrates tool
    hooks for before and after tool execution, allowing programmatic interception
    and modification of tool calls.

    Args:
        agent_action: The action containing the tool to execute.
        tools: List of available tools.
        i18n: Internationalization settings.
        agent_key: Optional key for event emission.
        agent_role: Optional role for event emission.
        tools_handler: Optional tools handler for tool execution.
        task: Optional task for tool execution.
        agent: Optional agent instance for tool execution.
        function_calling_llm: Optional LLM for function calling.
        fingerprint_context: Optional context for fingerprinting.
        crew: Optional crew instance for hook context.

    Returns:
        ToolResult containing the execution result and whether it should be
        treated as a final answer.
    """
    logger = Logger(verbose=crew.verbose if crew else False)
    tool_name_to_tool_map = {sanitize_tool_name(tool.name): tool for tool in tools}

    if agent_key and agent_role and agent:
        fingerprint_context = fingerprint_context or {}
        if agent:
            if hasattr(agent, "set_fingerprint") and callable(agent.set_fingerprint):
                if isinstance(fingerprint_context, dict):
                    try:
                        fingerprint_obj = Fingerprint.from_dict(fingerprint_context)
                        agent.set_fingerprint(fingerprint=fingerprint_obj)
                    except Exception as e:
                        raise ValueError(f"Failed to set fingerprint: {e}") from e

    tool_usage = ToolUsage(
        tools_handler=tools_handler,
        tools=tools,
        function_calling_llm=function_calling_llm,  # type: ignore[arg-type]
        task=task,
        agent=agent,
        action=agent_action,
    )

    tool_calling = tool_usage.parse_tool_calling(agent_action.text)

    if isinstance(tool_calling, ToolUsageError):
        return ToolResult(tool_calling.message, False)

    sanitized_tool_name = sanitize_tool_name(tool_calling.tool_name)
    tool = tool_name_to_tool_map.get(sanitized_tool_name)
    if tool:
        tool_input = tool_calling.arguments if tool_calling.arguments else {}
        hook_context = ToolCallHookContext(
            tool_name=tool_calling.tool_name,
            tool_input=tool_input,
            tool=tool,
            agent=agent,
            task=task,
            crew=crew,
        )

        before_hooks = get_before_tool_call_hooks()
        try:
            for hook in before_hooks:
                result = hook(hook_context)
                if result is False:
                    blocked_message = (
                        f"Tool execution blocked by hook. "
                        f"Tool: {tool_calling.tool_name}"
                    )
                    return ToolResult(blocked_message, False)
        except Exception as e:
            logger.log("error", f"Error in before_tool_call hook: {e}")

        tool_result = await tool_usage.ause(tool_calling, agent_action.text)

        after_hook_context = ToolCallHookContext(
            tool_name=tool_calling.tool_name,
            tool_input=tool_input,
            tool=tool,
            agent=agent,
            task=task,
            crew=crew,
            tool_result=tool_result,
        )

        after_hooks = get_after_tool_call_hooks()
        modified_result: str = tool_result
        try:
            for after_hook in after_hooks:
                hook_result = after_hook(after_hook_context)
                if hook_result is not None:
                    modified_result = hook_result
                    after_hook_context.tool_result = modified_result
        except Exception as e:
            logger.log("error", f"Error in after_tool_call hook: {e}")

        return ToolResult(modified_result, tool.result_as_answer)

    tool_result = i18n.errors("wrong_tool_name").format(
        tool=sanitized_tool_name,
        tools=", ".join(tool_name_to_tool_map.keys()),
    )
    return ToolResult(result=tool_result, result_as_answer=False)


def execute_tool_and_check_finality(
    agent_action: AgentAction,
    tools: list[CrewStructuredTool],
    i18n: I18N,
    agent_key: str | None = None,
    agent_role: str | None = None,
    tools_handler: ToolsHandler | None = None,
    task: Task | None = None,
    agent: Agent | BaseAgent | None = None,
    function_calling_llm: BaseLLM | LLM | None = None,
    fingerprint_context: dict[str, str] | None = None,
    crew: Crew | None = None,
) -> ToolResult:
    """Execute a tool and check if the result should be treated as a final answer.

    This function integrates tool hooks for before and after tool execution,
    allowing programmatic interception and modification of tool calls.

    Args:
        agent_action: The action containing the tool to execute
        tools: List of available tools
        i18n: Internationalization settings
        agent_key: Optional key for event emission
        agent_role: Optional role for event emission
        tools_handler: Optional tools handler for tool execution
        task: Optional task for tool execution
        agent: Optional agent instance for tool execution
        function_calling_llm: Optional LLM for function calling
        fingerprint_context: Optional context for fingerprinting
        crew: Optional crew instance for hook context

    Returns:
        ToolResult containing the execution result and whether it should be treated as a final answer
    """
    logger = Logger(verbose=crew.verbose if crew else False)
    tool_name_to_tool_map = {sanitize_tool_name(tool.name): tool for tool in tools}

    if agent_key and agent_role and agent:
        fingerprint_context = fingerprint_context or {}
        if agent:
            if hasattr(agent, "set_fingerprint") and callable(agent.set_fingerprint):
                if isinstance(fingerprint_context, dict):
                    try:
                        fingerprint_obj = Fingerprint.from_dict(fingerprint_context)
                        agent.set_fingerprint(fingerprint=fingerprint_obj)
                    except Exception as e:
                        raise ValueError(f"Failed to set fingerprint: {e}") from e

    tool_usage = ToolUsage(
        tools_handler=tools_handler,
        tools=tools,
        function_calling_llm=function_calling_llm,  # type: ignore[arg-type]
        task=task,
        agent=agent,
        action=agent_action,
    )

    tool_calling = tool_usage.parse_tool_calling(agent_action.text)

    if isinstance(tool_calling, ToolUsageError):
        return ToolResult(tool_calling.message, False)

    sanitized_tool_name = sanitize_tool_name(tool_calling.tool_name)
    tool = tool_name_to_tool_map.get(sanitized_tool_name)
    if tool:
        tool_input = tool_calling.arguments if tool_calling.arguments else {}
        hook_context = ToolCallHookContext(
            tool_name=tool_calling.tool_name,
            tool_input=tool_input,
            tool=tool,
            agent=agent,
            task=task,
            crew=crew,
        )

        before_hooks = get_before_tool_call_hooks()
        try:
            for hook in before_hooks:
                result = hook(hook_context)
                if result is False:
                    blocked_message = (
                        f"Tool execution blocked by hook. "
                        f"Tool: {tool_calling.tool_name}"
                    )
                    return ToolResult(blocked_message, False)
        except Exception as e:
            logger.log("error", f"Error in before_tool_call hook: {e}")

        tool_result = tool_usage.use(tool_calling, agent_action.text)

        after_hook_context = ToolCallHookContext(
            tool_name=tool_calling.tool_name,
            tool_input=tool_input,
            tool=tool,
            agent=agent,
            task=task,
            crew=crew,
            tool_result=tool_result,
        )

        # Execute after_tool_call hooks
        after_hooks = get_after_tool_call_hooks()
        modified_result: str = tool_result
        try:
            for after_hook in after_hooks:
                hook_result = after_hook(after_hook_context)
                if hook_result is not None:
                    modified_result = hook_result
                    after_hook_context.tool_result = modified_result
        except Exception as e:
            logger.log("error", f"Error in after_tool_call hook: {e}")

        return ToolResult(modified_result, tool.result_as_answer)

    tool_result = i18n.errors("wrong_tool_name").format(
        tool=sanitized_tool_name,
        tools=", ".join(tool_name_to_tool_map.keys()),
    )
    return ToolResult(result=tool_result, result_as_answer=False)
