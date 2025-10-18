from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.security.fingerprint import Fingerprint
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.tools.tool_usage import ToolUsage, ToolUsageError
from crewai.utilities.i18n import I18N

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task


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
) -> ToolResult:
    """Execute a tool and check if the result should be treated as a final answer.

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

    Returns:
        ToolResult containing the execution result and whether it should be treated as a final answer
    """
    tool_name_to_tool_map = {tool.name: tool for tool in tools}

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

    # Create tool usage instance
    tool_usage = ToolUsage(
        tools_handler=tools_handler,
        tools=tools,
        function_calling_llm=function_calling_llm,
        task=task,
        agent=agent,
        action=agent_action,
    )

    # Parse tool calling
    tool_calling = tool_usage.parse_tool_calling(agent_action.text)

    if isinstance(tool_calling, ToolUsageError):
        return ToolResult(tool_calling.message, False)

    # Check if tool name matches
    if tool_calling.tool_name.casefold().strip() in [
        name.casefold().strip() for name in tool_name_to_tool_map
    ] or tool_calling.tool_name.casefold().replace("_", " ") in [
        name.casefold().strip() for name in tool_name_to_tool_map
    ]:
        tool_result = tool_usage.use(tool_calling, agent_action.text)
        tool = tool_name_to_tool_map.get(tool_calling.tool_name)
        if tool:
            return ToolResult(tool_result, tool.result_as_answer)

    # Handle invalid tool name
    tool_result = i18n.errors("wrong_tool_name").format(
        tool=tool_calling.tool_name,
        tools=", ".join([tool.name.casefold() for tool in tools]),
    )
    return ToolResult(result=tool_result, result_as_answer=False)
