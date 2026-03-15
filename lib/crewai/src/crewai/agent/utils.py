"""Utility functions for agent task execution.

This module contains shared logic extracted from the Agent's execute_task
and aexecute_task methods to reduce code duplication.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.knowledge_events import (
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.i18n import I18N


def handle_reasoning(agent: Agent, task: Task) -> None:
    """Handle the reasoning process for an agent before task execution.

    Args:
        agent: The agent performing the task.
        task: The task to execute.
    """
    if not agent.reasoning:
        return

    try:
        from crewai.utilities.reasoning_handler import (
            AgentReasoning,
            AgentReasoningOutput,
        )

        reasoning_handler = AgentReasoning(task=task, agent=agent)
        reasoning_output: AgentReasoningOutput = (
            reasoning_handler.handle_agent_reasoning()
        )
        task.description += f"\n\nReasoning Plan:\n{reasoning_output.plan.plan}"
    except Exception as e:
        agent._logger.log("error", f"Error during reasoning process: {e!s}")


def build_task_prompt_with_schema(task: Task, task_prompt: str, i18n: I18N) -> str:
    """Build task prompt with JSON/Pydantic schema instructions if applicable.

    Args:
        task: The task being executed.
        task_prompt: The initial task prompt.
        i18n: Internationalization instance.

    Returns:
        The task prompt potentially augmented with schema instructions.
    """
    if (task.output_json or task.output_pydantic) and not task.response_model:
        if task.output_json:
            schema_dict = generate_model_description(task.output_json)
            schema = json.dumps(schema_dict["json_schema"]["schema"], indent=2)
            task_prompt += "\n" + i18n.slice("formatted_task_instructions").format(
                output_format=schema
            )
        elif task.output_pydantic:
            schema_dict = generate_model_description(task.output_pydantic)
            schema = json.dumps(schema_dict["json_schema"]["schema"], indent=2)
            task_prompt += "\n" + i18n.slice("formatted_task_instructions").format(
                output_format=schema
            )
    return task_prompt


def format_task_with_context(task_prompt: str, context: str | None, i18n: I18N) -> str:
    """Format task prompt with context if provided.

    Args:
        task_prompt: The task prompt.
        context: Optional context string.
        i18n: Internationalization instance.

    Returns:
        The task prompt formatted with context if provided.
    """
    if context:
        return i18n.slice("task_with_context").format(task=task_prompt, context=context)
    return task_prompt


def get_knowledge_config(agent: Agent) -> dict[str, Any]:
    """Get knowledge configuration from agent.

    Args:
        agent: The agent instance.

    Returns:
        Dictionary of knowledge configuration.
    """
    return agent.knowledge_config.model_dump() if agent.knowledge_config else {}


def handle_knowledge_retrieval(
    agent: Agent,
    task: Task,
    task_prompt: str,
    knowledge_config: dict[str, Any],
    query_func: Any,
    crew_query_func: Any,
) -> str:
    """Handle knowledge retrieval for task execution.

    This function handles both agent-specific and crew-specific knowledge queries.

    Args:
        agent: The agent performing the task.
        task: The task being executed.
        task_prompt: The current task prompt.
        knowledge_config: Knowledge configuration dictionary.
        query_func: Function to query agent knowledge (sync or async).
        crew_query_func: Function to query crew knowledge (sync or async).

    Returns:
        The task prompt potentially augmented with knowledge context.
    """
    if not (agent.knowledge or (agent.crew and agent.crew.knowledge)):
        return task_prompt

    crewai_event_bus.emit(
        agent,
        event=KnowledgeRetrievalStartedEvent(
            from_task=task,
            from_agent=agent,
        ),
    )
    try:
        agent.knowledge_search_query = agent._get_knowledge_search_query(
            task_prompt, task
        )
        if agent.knowledge_search_query:
            if agent.knowledge:
                agent_knowledge_snippets = query_func(
                    [agent.knowledge_search_query], **knowledge_config
                )
                if agent_knowledge_snippets:
                    agent.agent_knowledge_context = extract_knowledge_context(
                        agent_knowledge_snippets
                    )
                    if agent.agent_knowledge_context:
                        task_prompt += agent.agent_knowledge_context

            knowledge_snippets = crew_query_func(
                [agent.knowledge_search_query], **knowledge_config
            )
            if knowledge_snippets:
                agent.crew_knowledge_context = extract_knowledge_context(
                    knowledge_snippets
                )
                if agent.crew_knowledge_context:
                    task_prompt += agent.crew_knowledge_context

            crewai_event_bus.emit(
                agent,
                event=KnowledgeRetrievalCompletedEvent(
                    query=agent.knowledge_search_query,
                    from_task=task,
                    from_agent=agent,
                    retrieved_knowledge=_combine_knowledge_context(agent),
                ),
            )
    except Exception as e:
        crewai_event_bus.emit(
            agent,
            event=KnowledgeSearchQueryFailedEvent(
                query=agent.knowledge_search_query or "",
                error=str(e),
                from_task=task,
                from_agent=agent,
            ),
        )
    return task_prompt


def _combine_knowledge_context(agent: Agent) -> str:
    """Combine agent and crew knowledge contexts into a single string.

    Args:
        agent: The agent with knowledge contexts.

    Returns:
        Combined knowledge context string.
    """
    agent_ctx = agent.agent_knowledge_context or ""
    crew_ctx = agent.crew_knowledge_context or ""
    separator = "\n" if agent_ctx and crew_ctx else ""
    return agent_ctx + separator + crew_ctx


def apply_training_data(agent: Agent, task_prompt: str) -> str:
    """Apply training data to the task prompt.

    Args:
        agent: The agent performing the task.
        task_prompt: The task prompt.

    Returns:
        The task prompt with training data applied.
    """
    if agent.crew and agent.crew._train:
        return agent._training_handler(task_prompt=task_prompt)
    return agent._use_trained_data(task_prompt=task_prompt)


def process_tool_results(agent: Agent, result: Any) -> Any:
    """Process tool results, returning result_as_answer if applicable.

    Args:
        agent: The agent with tool results.
        result: The current result.

    Returns:
        The final result, potentially overridden by tool result_as_answer.
    """
    for tool_result in agent.tools_results:
        if tool_result.get("result_as_answer", False):
            result = tool_result["result"]
    return result


def save_last_messages(agent: Agent) -> None:
    """Save the last messages from agent executor.

    Sanitizes messages to be compatible with TaskOutput's LLMMessage type,
    which accepts 'user', 'assistant', 'system', and 'tool' roles.
    Preserves tool_call_id/name for tool messages and tool_calls for assistant messages.

    Args:
        agent: The agent instance.
    """
    if not agent.agent_executor or not hasattr(agent.agent_executor, "messages"):
        agent._last_messages = []
        return

    sanitized_messages: list[LLMMessage] = []
    for msg in agent.agent_executor.messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        content = msg.get("content")
        if content is None:
            content = ""
        sanitized_msg: LLMMessage = {"role": role, "content": content}
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                sanitized_msg["tool_call_id"] = tool_call_id
            name = msg.get("name")
            if name:
                sanitized_msg["name"] = name
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                sanitized_msg["tool_calls"] = tool_calls
        sanitized_messages.append(sanitized_msg)

    agent._last_messages = sanitized_messages


def prepare_tools(
    agent: Agent, tools: list[BaseTool] | None, task: Task
) -> list[BaseTool]:
    """Prepare tools for task execution and create agent executor.

    Args:
        agent: The agent instance.
        tools: Optional list of tools.
        task: The task being executed.

    Returns:
        The list of tools to use.
    """
    final_tools = tools or agent.tools or []
    agent.create_agent_executor(tools=final_tools, task=task)
    return final_tools


def validate_max_execution_time(max_execution_time: int | None) -> None:
    """Validate max_execution_time parameter.

    Args:
        max_execution_time: The maximum execution time to validate.

    Raises:
        ValueError: If max_execution_time is not a positive integer.
    """
    if max_execution_time is not None:
        if not isinstance(max_execution_time, int) or max_execution_time <= 0:
            raise ValueError(
                "Max Execution time must be a positive integer greater than zero"
            )


async def ahandle_knowledge_retrieval(
    agent: Agent,
    task: Task,
    task_prompt: str,
    knowledge_config: dict[str, Any],
) -> str:
    """Handle async knowledge retrieval for task execution.

    Args:
        agent: The agent performing the task.
        task: The task being executed.
        task_prompt: The current task prompt.
        knowledge_config: Knowledge configuration dictionary.

    Returns:
        The task prompt potentially augmented with knowledge context.
    """
    if not (agent.knowledge or (agent.crew and agent.crew.knowledge)):
        return task_prompt

    crewai_event_bus.emit(
        agent,
        event=KnowledgeRetrievalStartedEvent(
            from_task=task,
            from_agent=agent,
        ),
    )
    try:
        agent.knowledge_search_query = agent._get_knowledge_search_query(
            task_prompt, task
        )
        if agent.knowledge_search_query:
            if agent.knowledge:
                agent_knowledge_snippets = await agent.knowledge.aquery(
                    [agent.knowledge_search_query], **knowledge_config
                )
                if agent_knowledge_snippets:
                    agent.agent_knowledge_context = extract_knowledge_context(
                        agent_knowledge_snippets
                    )
                    if agent.agent_knowledge_context:
                        task_prompt += agent.agent_knowledge_context

            knowledge_snippets = await agent.crew.aquery_knowledge(
                [agent.knowledge_search_query], **knowledge_config
            )
            if knowledge_snippets:
                agent.crew_knowledge_context = extract_knowledge_context(
                    knowledge_snippets
                )
                if agent.crew_knowledge_context:
                    task_prompt += agent.crew_knowledge_context

            crewai_event_bus.emit(
                agent,
                event=KnowledgeRetrievalCompletedEvent(
                    query=agent.knowledge_search_query,
                    from_task=task,
                    from_agent=agent,
                    retrieved_knowledge=_combine_knowledge_context(agent),
                ),
            )
    except Exception as e:
        crewai_event_bus.emit(
            agent,
            event=KnowledgeSearchQueryFailedEvent(
                query=agent.knowledge_search_query or "",
                error=str(e),
                from_task=task,
                from_agent=agent,
            ),
        )
    return task_prompt
