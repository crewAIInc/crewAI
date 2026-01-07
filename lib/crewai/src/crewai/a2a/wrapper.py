"""A2A agent wrapping logic for metaclass integration.

Wraps agent classes with A2A delegation capabilities.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any

from a2a.types import Role, TaskState
from pydantic import BaseModel, ValidationError

from crewai.a2a.config import A2AConfig
from crewai.a2a.extensions.base import ExtensionRegistry
from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.templates import (
    AVAILABLE_AGENTS_TEMPLATE,
    CONVERSATION_TURN_INFO_TEMPLATE,
    PREVIOUS_A2A_CONVERSATION_TEMPLATE,
    REMOTE_AGENT_COMPLETED_NOTICE,
    UNAVAILABLE_AGENTS_NOTICE_TEMPLATE,
)
from crewai.a2a.types import AgentResponseProtocol
from crewai.a2a.utils import (
    aexecute_a2a_delegation,
    afetch_agent_card,
    execute_a2a_delegation,
    fetch_agent_card,
    get_a2a_agents_and_response_model,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConversationCompletedEvent,
    A2AMessageSentEvent,
)


if TYPE_CHECKING:
    from a2a.types import AgentCard, Message

    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


def wrap_agent_with_a2a_instance(
    agent: Agent, extension_registry: ExtensionRegistry | None = None
) -> None:
    """Wrap an agent instance's execute_task and aexecute_task methods with A2A support.

    This function modifies the agent instance by wrapping its execute_task
    and aexecute_task methods to add A2A delegation capabilities. Should only
    be called when the agent has a2a configuration set.

    Args:
        agent: The agent instance to wrap.
        extension_registry: Optional registry of A2A extensions.
    """
    if extension_registry is None:
        extension_registry = ExtensionRegistry()

    extension_registry.inject_all_tools(agent)

    original_execute_task = agent.execute_task.__func__  # type: ignore[attr-defined]
    original_aexecute_task = agent.aexecute_task.__func__  # type: ignore[attr-defined]

    @wraps(original_execute_task)
    def execute_task_with_a2a(
        self: Agent,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute task with A2A delegation support (sync)."""
        if not self.a2a:
            return original_execute_task(self, task, context, tools)  # type: ignore[no-any-return]

        a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)

        return _execute_task_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_fn=original_execute_task,
            task=task,
            agent_response_model=agent_response_model,
            context=context,
            tools=tools,
            extension_registry=extension_registry,
        )

    @wraps(original_aexecute_task)
    async def aexecute_task_with_a2a(
        self: Agent,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute task with A2A delegation support (async)."""
        if not self.a2a:
            return await original_aexecute_task(self, task, context, tools)  # type: ignore[no-any-return]

        a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)

        return await _aexecute_task_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_fn=original_aexecute_task,
            task=task,
            agent_response_model=agent_response_model,
            context=context,
            tools=tools,
            extension_registry=extension_registry,
        )

    object.__setattr__(agent, "execute_task", MethodType(execute_task_with_a2a, agent))
    object.__setattr__(
        agent, "aexecute_task", MethodType(aexecute_task_with_a2a, agent)
    )


def _fetch_card_from_config(
    config: A2AConfig,
) -> tuple[A2AConfig, AgentCard | Exception]:
    """Fetch agent card from A2A config.

    Args:
        config: A2A configuration

    Returns:
        Tuple of (config, card or exception)
    """
    try:
        card = fetch_agent_card(
            endpoint=config.endpoint,
            auth=config.auth,
            timeout=config.timeout,
        )
        return config, card
    except Exception as e:
        return config, e


def _fetch_agent_cards_concurrently(
    a2a_agents: list[A2AConfig],
) -> tuple[dict[str, AgentCard], dict[str, str]]:
    """Fetch agent cards concurrently for multiple A2A agents.

    Args:
        a2a_agents: List of A2A agent configurations

    Returns:
        Tuple of (agent_cards dict, failed_agents dict mapping endpoint to error message)
    """
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(a2a_agents)) as executor:
        futures = {
            executor.submit(_fetch_card_from_config, config): config
            for config in a2a_agents
        }
        for future in as_completed(futures):
            config, result = future.result()
            if isinstance(result, Exception):
                if config.fail_fast:
                    raise RuntimeError(
                        f"Failed to fetch agent card from {config.endpoint}. "
                        f"Ensure the A2A agent is running and accessible. Error: {result}"
                    ) from result
                failed_agents[config.endpoint] = str(result)
            else:
                agent_cards[config.endpoint] = result

    return agent_cards, failed_agents


def _execute_task_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig],
    original_fn: Callable[..., str],
    task: Task,
    agent_response_model: type[BaseModel],
    context: str | None,
    tools: list[BaseTool] | None,
    extension_registry: ExtensionRegistry,
) -> str:
    """Wrap execute_task with A2A delegation logic.

    Args:
        self: The agent instance
        a2a_agents: Dictionary of A2A agent configurations
        original_fn: The original execute_task method
        task: The task to execute
        context: Optional context for task execution
        tools: Optional tools available to the agent
        agent_response_model: Optional agent response model
        extension_registry: Registry of A2A extensions

    Returns:
        Task execution result (either from LLM or A2A agent)
    """
    original_description: str = task.description
    original_output_pydantic = task.output_pydantic
    original_response_model = task.response_model

    agent_cards, failed_agents = _fetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        unavailable_agents_text = ""
        for endpoint, error in failed_agents.items():
            unavailable_agents_text += f"  - {endpoint}: {error}\n"

        notice = UNAVAILABLE_AGENTS_NOTICE_TEMPLATE.substitute(
            unavailable_agents=unavailable_agents_text
        )
        task.description = f"{original_description}{notice}"

        try:
            return original_fn(self, task, context, tools)
        finally:
            task.description = original_description

    task.description, _ = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
        extension_registry=extension_registry,
    )
    task.response_model = agent_response_model

    try:
        raw_result = original_fn(self, task, context, tools)
        agent_response = _parse_agent_response(
            raw_result=raw_result, agent_response_model=agent_response_model
        )

        if extension_registry and isinstance(agent_response, BaseModel):
            agent_response = extension_registry.process_response_with_all(
                agent_response, {}
            )

        if isinstance(agent_response, BaseModel) and isinstance(
            agent_response, AgentResponseProtocol
        ):
            if agent_response.is_a2a:
                return _delegate_to_a2a(
                    self,
                    agent_response=agent_response,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_cards=agent_cards,
                    original_task_description=original_description,
                    extension_registry=extension_registry,
                )
            return str(agent_response.message)

        return raw_result
    finally:
        task.description = original_description
        task.output_pydantic = original_output_pydantic
        task.response_model = original_response_model


def _augment_prompt_with_a2a(
    a2a_agents: list[A2AConfig],
    task_description: str,
    agent_cards: dict[str, AgentCard],
    conversation_history: list[Message] | None = None,
    turn_num: int = 0,
    max_turns: int | None = None,
    failed_agents: dict[str, str] | None = None,
    extension_registry: ExtensionRegistry | None = None,
    remote_task_completed: bool = False,
) -> tuple[str, bool]:
    """Add A2A delegation instructions to prompt.

    Args:
        a2a_agents: Dictionary of A2A agent configurations
        task_description: Original task description
        agent_cards: dictionary mapping agent IDs to AgentCards
        conversation_history: Previous A2A Messages from conversation
        turn_num: Current turn number (0-indexed)
        max_turns: Maximum allowed turns (from config)
        failed_agents: Dictionary mapping failed agent endpoints to error messages
        extension_registry: Optional registry of A2A extensions

    Returns:
        Tuple of (augmented prompt, disable_structured_output flag)
    """

    if not agent_cards:
        return task_description, False

    agents_text = ""

    for config in a2a_agents:
        if config.endpoint in agent_cards:
            card = agent_cards[config.endpoint]
            agents_text += f"\n{card.model_dump_json(indent=2, exclude_none=True, include={'description', 'url', 'skills'})}\n"

    failed_agents = failed_agents or {}
    if failed_agents:
        agents_text += "\n<!-- Unavailable Agents -->\n"
        for endpoint, error in failed_agents.items():
            agents_text += f"\n<!-- Agent: {endpoint}\n     Status: Unavailable\n     Error: {error} -->\n"

    agents_text = AVAILABLE_AGENTS_TEMPLATE.substitute(available_a2a_agents=agents_text)

    history_text = ""

    if conversation_history:
        for msg in conversation_history:
            history_text += f"\n{msg.model_dump_json(indent=2, exclude_none=True, exclude={'message_id'})}\n"

    history_text = PREVIOUS_A2A_CONVERSATION_TEMPLATE.substitute(
        previous_a2a_conversation=history_text
    )

    extension_states = {}
    disable_structured_output = False
    if extension_registry and conversation_history:
        extension_states = extension_registry.extract_all_states(conversation_history)
        for state in extension_states.values():
            if state.is_ready():
                disable_structured_output = True
                break
    turn_info = ""

    if max_turns is not None and conversation_history:
        turn_count = turn_num + 1
        warning = ""
        if turn_count >= max_turns:
            warning = (
                "CRITICAL: This is the FINAL turn. You MUST conclude the conversation now.\n"
                "Set is_a2a=false and provide your final response to complete the task."
            )
        elif turn_count == max_turns - 1:
            warning = "WARNING: Next turn will be the last. Consider wrapping up the conversation."

        turn_info = CONVERSATION_TURN_INFO_TEMPLATE.substitute(
            turn_count=turn_count,
            max_turns=max_turns,
            warning=warning,
        )

    completion_notice = ""
    if remote_task_completed and conversation_history:
        completion_notice = REMOTE_AGENT_COMPLETED_NOTICE

    augmented_prompt = f"""{task_description}

IMPORTANT: You have the ability to delegate this task to remote A2A agents.
{agents_text}
{history_text}{turn_info}{completion_notice}

"""

    if extension_registry:
        augmented_prompt = extension_registry.augment_prompt_with_all(
            augmented_prompt, extension_states
        )

    return augmented_prompt, disable_structured_output


def _parse_agent_response(
    raw_result: str | dict[str, Any], agent_response_model: type[BaseModel]
) -> BaseModel | str | dict[str, Any]:
    """Parse LLM output as AgentResponse or return raw agent response."""
    if agent_response_model:
        try:
            if isinstance(raw_result, str):
                return agent_response_model.model_validate_json(raw_result)
            if isinstance(raw_result, dict):
                return agent_response_model.model_validate(raw_result)
        except ValidationError:
            return raw_result
    return raw_result


def _handle_max_turns_exceeded(
    conversation_history: list[Message],
    max_turns: int,
) -> str:
    """Handle the case when max turns is exceeded.

    Shared logic for both sync and async delegation.

    Returns:
        Final message if found in history.

    Raises:
        Exception: If no final message found and max turns exceeded.
    """
    if conversation_history:
        for msg in reversed(conversation_history):
            if msg.role == Role.agent:
                text_parts = [
                    part.root.text for part in msg.parts if part.root.kind == "text"
                ]
                final_message = (
                    " ".join(text_parts) if text_parts else "Conversation completed"
                )
                crewai_event_bus.emit(
                    None,
                    A2AConversationCompletedEvent(
                        status="completed",
                        final_result=final_message,
                        error=None,
                        total_turns=max_turns,
                    ),
                )
                return final_message

    crewai_event_bus.emit(
        None,
        A2AConversationCompletedEvent(
            status="failed",
            final_result=None,
            error=f"Conversation exceeded maximum turns ({max_turns})",
            total_turns=max_turns,
        ),
    )
    raise Exception(f"A2A conversation exceeded maximum turns ({max_turns})")


def _process_response_result(
    raw_result: str,
    disable_structured_output: bool,
    turn_num: int,
    agent_role: str,
    agent_response_model: type[BaseModel],
) -> tuple[str | None, str | None]:
    """Process LLM response and determine next action.

    Shared logic for both sync and async handlers.

    Returns:
        Tuple of (final_result, next_request).
    """
    if disable_structured_output:
        final_turn_number = turn_num + 1
        result_text = str(raw_result)
        crewai_event_bus.emit(
            None,
            A2AMessageSentEvent(
                message=result_text,
                turn_number=final_turn_number,
                is_multiturn=True,
                agent_role=agent_role,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AConversationCompletedEvent(
                status="completed",
                final_result=result_text,
                error=None,
                total_turns=final_turn_number,
            ),
        )
        return result_text, None

    llm_response = _parse_agent_response(
        raw_result=raw_result, agent_response_model=agent_response_model
    )

    if isinstance(llm_response, BaseModel) and isinstance(
        llm_response, AgentResponseProtocol
    ):
        if not llm_response.is_a2a:
            final_turn_number = turn_num + 1
            crewai_event_bus.emit(
                None,
                A2AMessageSentEvent(
                    message=str(llm_response.message),
                    turn_number=final_turn_number,
                    is_multiturn=True,
                    agent_role=agent_role,
                ),
            )
            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="completed",
                    final_result=str(llm_response.message),
                    error=None,
                    total_turns=final_turn_number,
                ),
            )
            return str(llm_response.message), None
        return None, str(llm_response.message)

    return str(raw_result), None


def _prepare_agent_cards_dict(
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: dict[str, AgentCard] | None,
) -> dict[str, AgentCard]:
    """Prepare agent cards dictionary from result and existing cards.

    Shared logic for both sync and async response handlers.
    """
    agent_cards_dict = agent_cards or {}
    if "agent_card" in a2a_result and agent_id not in agent_cards_dict:
        agent_cards_dict[agent_id] = a2a_result["agent_card"]
    return agent_cards_dict


def _prepare_delegation_context(
    self: Agent,
    agent_response: AgentResponseProtocol,
    task: Task,
    original_task_description: str | None,
) -> tuple[
    list[A2AConfig],
    type[BaseModel],
    str,
    str,
    A2AConfig,
    str | None,
    str | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[str],
    str,
    int,
]:
    """Prepare delegation context from agent response and task.

    Shared logic for both sync and async delegation.

    Returns:
        Tuple containing all the context values needed for delegation.
    """
    a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)
    agent_ids = tuple(config.endpoint for config in a2a_agents)
    current_request = str(agent_response.message)

    if hasattr(agent_response, "a2a_ids") and agent_response.a2a_ids:
        agent_id = agent_response.a2a_ids[0]
    else:
        agent_id = agent_ids[0] if agent_ids else ""

    if agent_id and agent_id not in agent_ids:
        raise ValueError(
            f"Unknown A2A agent ID(s): {agent_response.a2a_ids} not in {agent_ids}"
        )

    agent_config = next(filter(lambda x: x.endpoint == agent_id, a2a_agents))
    task_config = task.config or {}
    context_id = task_config.get("context_id")
    task_id_config = task_config.get("task_id")
    metadata = task_config.get("metadata")
    extensions = task_config.get("extensions")
    reference_task_ids = task_config.get("reference_task_ids", [])

    if original_task_description is None:
        original_task_description = task.description

    max_turns = agent_config.max_turns

    return (
        a2a_agents,
        agent_response_model,
        current_request,
        agent_id,
        agent_config,
        context_id,
        task_id_config,
        metadata,
        extensions,
        reference_task_ids,
        original_task_description,
        max_turns,
    )


def _handle_task_completion(
    a2a_result: TaskStateResult,
    task: Task,
    task_id_config: str | None,
    reference_task_ids: list[str],
    agent_config: A2AConfig,
    turn_num: int,
) -> tuple[str | None, str | None, list[str]]:
    """Handle task completion state including reference task updates.

    Shared logic for both sync and async delegation.

    Returns:
        Tuple of (result_if_trusted, updated_task_id, updated_reference_task_ids).
    """
    if a2a_result["status"] == TaskState.completed:
        if task_id_config is not None and task_id_config not in reference_task_ids:
            reference_task_ids.append(task_id_config)
            if task.config is None:
                task.config = {}
            task.config["reference_task_ids"] = reference_task_ids
        task_id_config = None

        if agent_config.trust_remote_completion_status:
            result_text = a2a_result.get("result", "")
            final_turn_number = turn_num + 1
            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="completed",
                    final_result=result_text,
                    error=None,
                    total_turns=final_turn_number,
                ),
            )
            return str(result_text), task_id_config, reference_task_ids

    return None, task_id_config, reference_task_ids


def _handle_agent_response_and_continue(
    self: Agent,
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: dict[str, AgentCard] | None,
    a2a_agents: list[A2AConfig],
    original_task_description: str,
    conversation_history: list[Message],
    turn_num: int,
    max_turns: int,
    task: Task,
    original_fn: Callable[..., str],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_response_model: type[BaseModel],
    remote_task_completed: bool = False,
) -> tuple[str | None, str | None]:
    """Handle A2A result and get CrewAI agent's response.

    Args:
        self: The agent instance
        a2a_result: Result from A2A delegation
        agent_id: ID of the A2A agent
        agent_cards: Pre-fetched agent cards
        a2a_agents: List of A2A configurations
        original_task_description: Original task description
        conversation_history: Conversation history
        turn_num: Current turn number
        max_turns: Maximum turns allowed
        task: The task being executed
        original_fn: Original execute_task method
        context: Optional context
        tools: Optional tools
        agent_response_model: Response model for parsing

    Returns:
        Tuple of (final_result, current_request) where:
        - final_result is not None if conversation should end
        - current_request is the next message to send if continuing
    """
    agent_cards_dict = _prepare_agent_cards_dict(a2a_result, agent_id, agent_cards)

    task.description, disable_structured_output = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_task_description,
        conversation_history=conversation_history,
        turn_num=turn_num,
        max_turns=max_turns,
        agent_cards=agent_cards_dict,
        remote_task_completed=remote_task_completed,
    )

    original_response_model = task.response_model
    if disable_structured_output:
        task.response_model = None

    raw_result = original_fn(self, task, context, tools)

    if disable_structured_output:
        task.response_model = original_response_model

    return _process_response_result(
        raw_result=raw_result,
        disable_structured_output=disable_structured_output,
        turn_num=turn_num,
        agent_role=self.role,
        agent_response_model=agent_response_model,
    )


def _delegate_to_a2a(
    self: Agent,
    agent_response: AgentResponseProtocol,
    task: Task,
    original_fn: Callable[..., str],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_cards: dict[str, AgentCard] | None = None,
    original_task_description: str | None = None,
    extension_registry: ExtensionRegistry | None = None,
) -> str:
    """Delegate to A2A agent with multi-turn conversation support.

    Args:
        self: The agent instance
        agent_response: The AgentResponse indicating delegation
        task: The task being executed (for extracting A2A fields)
        original_fn: The original execute_task method for follow-ups
        context: Optional context for task execution
        tools: Optional tools available to the agent
        agent_cards: Pre-fetched agent cards from _execute_task_with_a2a
        original_task_description: The original task description before A2A augmentation
        extension_registry: Optional registry of A2A extensions

    Returns:
        Result from A2A agent

    Raises:
        ImportError: If a2a-sdk is not installed
    """
    (
        a2a_agents,
        agent_response_model,
        current_request,
        agent_id,
        agent_config,
        context_id,
        task_id_config,
        metadata,
        extensions,
        reference_task_ids,
        original_task_description,
        max_turns,
    ) = _prepare_delegation_context(
        self, agent_response, task, original_task_description
    )

    conversation_history: list[Message] = []

    try:
        for turn_num in range(max_turns):
            console_formatter = getattr(crewai_event_bus, "_console", None)
            agent_branch = None
            if console_formatter:
                agent_branch = getattr(
                    console_formatter, "current_agent_branch", None
                ) or getattr(console_formatter, "current_task_branch", None)

            a2a_result = execute_a2a_delegation(
                endpoint=agent_config.endpoint,
                auth=agent_config.auth,
                timeout=agent_config.timeout,
                task_description=current_request,
                context_id=context_id,
                task_id=task_id_config,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=extensions,
                conversation_history=conversation_history,
                agent_id=agent_id,
                agent_role=Role.user,
                agent_branch=agent_branch,
                response_model=agent_config.response_model,
                turn_number=turn_num + 1,
                updates=agent_config.updates,
            )

            conversation_history = a2a_result.get("history", [])

            if conversation_history:
                latest_message = conversation_history[-1]
                if latest_message.task_id is not None:
                    task_id_config = latest_message.task_id
                if latest_message.context_id is not None:
                    context_id = latest_message.context_id

            if a2a_result["status"] in [TaskState.completed, TaskState.input_required]:
                trusted_result, task_id_config, reference_task_ids = (
                    _handle_task_completion(
                        a2a_result,
                        task,
                        task_id_config,
                        reference_task_ids,
                        agent_config,
                        turn_num,
                    )
                )
                if trusted_result is not None:
                    return trusted_result

                final_result, next_request = _handle_agent_response_and_continue(
                    self=self,
                    a2a_result=a2a_result,
                    agent_id=agent_id,
                    agent_cards=agent_cards,
                    a2a_agents=a2a_agents,
                    original_task_description=original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=max_turns,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_response_model=agent_response_model,
                    remote_task_completed=(a2a_result["status"] == TaskState.completed),
                )

                if final_result is not None:
                    return final_result

                if next_request is not None:
                    current_request = next_request

                continue

            error_msg = a2a_result.get("error", "Unknown error")

            final_result, next_request = _handle_agent_response_and_continue(
                self=self,
                a2a_result=a2a_result,
                agent_id=agent_id,
                agent_cards=agent_cards,
                a2a_agents=a2a_agents,
                original_task_description=original_task_description,
                conversation_history=conversation_history,
                turn_num=turn_num,
                max_turns=max_turns,
                task=task,
                original_fn=original_fn,
                context=context,
                tools=tools,
                agent_response_model=agent_response_model,
                remote_task_completed=False,
            )

            if final_result is not None:
                return final_result

            if next_request is not None:
                current_request = next_request
                continue

            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="failed",
                    final_result=None,
                    error=error_msg,
                    total_turns=turn_num + 1,
                ),
            )
            return f"A2A delegation failed: {error_msg}"

        return _handle_max_turns_exceeded(conversation_history, max_turns)

    finally:
        task.description = original_task_description


async def _afetch_card_from_config(
    config: A2AConfig,
) -> tuple[A2AConfig, AgentCard | Exception]:
    """Fetch agent card from A2A config asynchronously."""
    try:
        card = await afetch_agent_card(
            endpoint=config.endpoint,
            auth=config.auth,
            timeout=config.timeout,
        )
        return config, card
    except Exception as e:
        return config, e


async def _afetch_agent_cards_concurrently(
    a2a_agents: list[A2AConfig],
) -> tuple[dict[str, AgentCard], dict[str, str]]:
    """Fetch agent cards concurrently for multiple A2A agents using asyncio."""
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    tasks = [_afetch_card_from_config(config) for config in a2a_agents]
    results = await asyncio.gather(*tasks)

    for config, result in results:
        if isinstance(result, Exception):
            if config.fail_fast:
                raise RuntimeError(
                    f"Failed to fetch agent card from {config.endpoint}. "
                    f"Ensure the A2A agent is running and accessible. Error: {result}"
                ) from result
            failed_agents[config.endpoint] = str(result)
        else:
            agent_cards[config.endpoint] = result

    return agent_cards, failed_agents


async def _aexecute_task_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig],
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    task: Task,
    agent_response_model: type[BaseModel],
    context: str | None,
    tools: list[BaseTool] | None,
    extension_registry: ExtensionRegistry,
) -> str:
    """Async version of _execute_task_with_a2a."""
    original_description: str = task.description
    original_output_pydantic = task.output_pydantic
    original_response_model = task.response_model

    agent_cards, failed_agents = await _afetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        unavailable_agents_text = ""
        for endpoint, error in failed_agents.items():
            unavailable_agents_text += f"  - {endpoint}: {error}\n"

        notice = UNAVAILABLE_AGENTS_NOTICE_TEMPLATE.substitute(
            unavailable_agents=unavailable_agents_text
        )
        task.description = f"{original_description}{notice}"

        try:
            return await original_fn(self, task, context, tools)
        finally:
            task.description = original_description

    task.description, _ = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
        extension_registry=extension_registry,
    )
    task.response_model = agent_response_model

    try:
        raw_result = await original_fn(self, task, context, tools)
        agent_response = _parse_agent_response(
            raw_result=raw_result, agent_response_model=agent_response_model
        )

        if extension_registry and isinstance(agent_response, BaseModel):
            agent_response = extension_registry.process_response_with_all(
                agent_response, {}
            )

        if isinstance(agent_response, BaseModel) and isinstance(
            agent_response, AgentResponseProtocol
        ):
            if agent_response.is_a2a:
                return await _adelegate_to_a2a(
                    self,
                    agent_response=agent_response,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_cards=agent_cards,
                    original_task_description=original_description,
                    extension_registry=extension_registry,
                )
            return str(agent_response.message)

        return raw_result
    finally:
        task.description = original_description
        task.output_pydantic = original_output_pydantic
        task.response_model = original_response_model


async def _ahandle_agent_response_and_continue(
    self: Agent,
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: dict[str, AgentCard] | None,
    a2a_agents: list[A2AConfig],
    original_task_description: str,
    conversation_history: list[Message],
    turn_num: int,
    max_turns: int,
    task: Task,
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_response_model: type[BaseModel],
    remote_task_completed: bool = False,
) -> tuple[str | None, str | None]:
    """Async version of _handle_agent_response_and_continue."""
    agent_cards_dict = _prepare_agent_cards_dict(a2a_result, agent_id, agent_cards)

    task.description, disable_structured_output = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_task_description,
        conversation_history=conversation_history,
        turn_num=turn_num,
        max_turns=max_turns,
        agent_cards=agent_cards_dict,
        remote_task_completed=remote_task_completed,
    )

    original_response_model = task.response_model
    if disable_structured_output:
        task.response_model = None

    raw_result = await original_fn(self, task, context, tools)

    if disable_structured_output:
        task.response_model = original_response_model

    return _process_response_result(
        raw_result=raw_result,
        disable_structured_output=disable_structured_output,
        turn_num=turn_num,
        agent_role=self.role,
        agent_response_model=agent_response_model,
    )


async def _adelegate_to_a2a(
    self: Agent,
    agent_response: AgentResponseProtocol,
    task: Task,
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_cards: dict[str, AgentCard] | None = None,
    original_task_description: str | None = None,
    extension_registry: ExtensionRegistry | None = None,
) -> str:
    """Async version of _delegate_to_a2a."""
    (
        a2a_agents,
        agent_response_model,
        current_request,
        agent_id,
        agent_config,
        context_id,
        task_id_config,
        metadata,
        extensions,
        reference_task_ids,
        original_task_description,
        max_turns,
    ) = _prepare_delegation_context(
        self, agent_response, task, original_task_description
    )

    conversation_history: list[Message] = []

    try:
        for turn_num in range(max_turns):
            console_formatter = getattr(crewai_event_bus, "_console", None)
            agent_branch = None
            if console_formatter:
                agent_branch = getattr(
                    console_formatter, "current_agent_branch", None
                ) or getattr(console_formatter, "current_task_branch", None)

            a2a_result = await aexecute_a2a_delegation(
                endpoint=agent_config.endpoint,
                auth=agent_config.auth,
                timeout=agent_config.timeout,
                task_description=current_request,
                context_id=context_id,
                task_id=task_id_config,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=extensions,
                conversation_history=conversation_history,
                agent_id=agent_id,
                agent_role=Role.user,
                agent_branch=agent_branch,
                response_model=agent_config.response_model,
                turn_number=turn_num + 1,
                updates=agent_config.updates,
            )

            conversation_history = a2a_result.get("history", [])

            if conversation_history:
                latest_message = conversation_history[-1]
                if latest_message.task_id is not None:
                    task_id_config = latest_message.task_id
                if latest_message.context_id is not None:
                    context_id = latest_message.context_id

            if a2a_result["status"] in [TaskState.completed, TaskState.input_required]:
                trusted_result, task_id_config, reference_task_ids = (
                    _handle_task_completion(
                        a2a_result,
                        task,
                        task_id_config,
                        reference_task_ids,
                        agent_config,
                        turn_num,
                    )
                )
                if trusted_result is not None:
                    return trusted_result

                final_result, next_request = await _ahandle_agent_response_and_continue(
                    self=self,
                    a2a_result=a2a_result,
                    agent_id=agent_id,
                    agent_cards=agent_cards,
                    a2a_agents=a2a_agents,
                    original_task_description=original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=max_turns,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_response_model=agent_response_model,
                    remote_task_completed=(a2a_result["status"] == TaskState.completed),
                )

                if final_result is not None:
                    return final_result

                if next_request is not None:
                    current_request = next_request

                continue

            error_msg = a2a_result.get("error", "Unknown error")

            final_result, next_request = await _ahandle_agent_response_and_continue(
                self=self,
                a2a_result=a2a_result,
                agent_id=agent_id,
                agent_cards=agent_cards,
                a2a_agents=a2a_agents,
                original_task_description=original_task_description,
                conversation_history=conversation_history,
                turn_num=turn_num,
                max_turns=max_turns,
                task=task,
                original_fn=original_fn,
                context=context,
                tools=tools,
                agent_response_model=agent_response_model,
            )

            if final_result is not None:
                return final_result

            if next_request is not None:
                current_request = next_request
                continue

            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="failed",
                    final_result=None,
                    error=error_msg,
                    total_turns=turn_num + 1,
                ),
            )
            return f"A2A delegation failed: {error_msg}"

        return _handle_max_turns_exceeded(conversation_history, max_turns)

    finally:
        task.description = original_task_description
