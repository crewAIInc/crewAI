"""A2A agent wrapping logic for metaclass integration.

Wraps agent classes with A2A delegation capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

from a2a.types import Role
from pydantic import BaseModel, ValidationError

from crewai.a2a.config import A2AConfig
from crewai.a2a.templates import (
    AVAILABLE_AGENTS_TEMPLATE,
    CONVERSATION_TURN_INFO_TEMPLATE,
    PREVIOUS_A2A_CONVERSATION_TEMPLATE,
    UNAVAILABLE_AGENTS_NOTICE_TEMPLATE,
)
from crewai.a2a.types import AgentResponseProtocol
from crewai.a2a.utils import (
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


def wrap_agent_with_a2a_instance(agent: Agent) -> None:
    """Wrap an agent instance's execute_task method with A2A support.

    This function modifies the agent instance by wrapping its execute_task
    method to add A2A delegation capabilities. Should only be called when
    the agent has a2a configuration set.

    Args:
        agent: The agent instance to wrap
    """
    original_execute_task = agent.execute_task.__func__  # type: ignore[attr-defined]

    @wraps(original_execute_task)
    def execute_task_with_a2a(
        self: Agent,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute task with A2A delegation support.

        Args:
            self: The agent instance
            task: The task to execute
            context: Optional context for task execution
            tools: Optional tools available to the agent

        Returns:
            Task execution result
        """
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
        )

    object.__setattr__(agent, "execute_task", MethodType(execute_task_with_a2a, agent))


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

    task.description = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
    )
    task.response_model = agent_response_model

    try:
        raw_result = original_fn(self, task, context, tools)
        agent_response = _parse_agent_response(
            raw_result=raw_result, agent_response_model=agent_response_model
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
) -> str:
    """Add A2A delegation instructions to prompt.

    Args:
        a2a_agents: Dictionary of A2A agent configurations
        task_description: Original task description
        agent_cards: dictionary mapping agent IDs to AgentCards
        conversation_history: Previous A2A Messages from conversation
        turn_num: Current turn number (0-indexed)
        max_turns: Maximum allowed turns (from config)
        failed_agents: Dictionary mapping failed agent endpoints to error messages

    Returns:
        Augmented task description with A2A instructions
    """

    if not agent_cards:
        return task_description

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

    return f"""{task_description}

IMPORTANT: You have the ability to delegate this task to remote A2A agents.

{agents_text}
{history_text}{turn_info}


"""


def _parse_agent_response(
    raw_result: str | dict[str, Any], agent_response_model: type[BaseModel]
) -> BaseModel | str:
    """Parse LLM output as AgentResponse or return raw agent response.

    Args:
        raw_result: Raw output from LLM
        agent_response_model: The agent response model

    Returns:
        Parsed AgentResponse or string
    """
    if agent_response_model:
        try:
            if isinstance(raw_result, str):
                return agent_response_model.model_validate_json(raw_result)
            if isinstance(raw_result, dict):
                return agent_response_model.model_validate(raw_result)
        except ValidationError:
            return cast(str, raw_result)
    return cast(str, raw_result)


def _handle_agent_response_and_continue(
    self: Agent,
    a2a_result: dict[str, Any],
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
    agent_cards_dict = agent_cards or {}
    if "agent_card" in a2a_result and agent_id not in agent_cards_dict:
        agent_cards_dict[agent_id] = a2a_result["agent_card"]

    task.description = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_task_description,
        conversation_history=conversation_history,
        turn_num=turn_num,
        max_turns=max_turns,
        agent_cards=agent_cards_dict,
    )

    raw_result = original_fn(self, task, context, tools)
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
                    agent_role=self.role,
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


def _delegate_to_a2a(
    self: Agent,
    agent_response: AgentResponseProtocol,
    task: Task,
    original_fn: Callable[..., str],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_cards: dict[str, AgentCard] | None = None,
    original_task_description: str | None = None,
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

    Returns:
        Result from A2A agent

    Raises:
        ImportError: If a2a-sdk is not installed
    """
    a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)
    agent_ids = tuple(config.endpoint for config in a2a_agents)
    current_request = str(agent_response.message)
    agent_id = agent_response.a2a_ids[0]

    if agent_id not in agent_ids:
        raise ValueError(
            f"Unknown A2A agent ID(s): {agent_response.a2a_ids} not in {agent_ids}"
        )

    agent_config = next(filter(lambda x: x.endpoint == agent_id, a2a_agents))
    task_config = task.config or {}
    context_id = task_config.get("context_id")
    task_id_config = task_config.get("task_id")
    reference_task_ids = task_config.get("reference_task_ids")
    metadata = task_config.get("metadata")
    extensions = task_config.get("extensions")

    if original_task_description is None:
        original_task_description = task.description

    conversation_history: list[Message] = []
    max_turns = agent_config.max_turns

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
            )

            conversation_history = a2a_result.get("history", [])

            if a2a_result["status"] in ["completed", "input_required"]:
                if (
                    a2a_result["status"] == "completed"
                    and agent_config.trust_remote_completion_status
                ):
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
                    return result_text  # type: ignore[no-any-return]

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
                )

                if final_result is not None:
                    return final_result

                if next_request is not None:
                    current_request = next_request

                continue

            error_msg = a2a_result.get("error", "Unknown error")
            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="failed",
                    final_result=None,
                    error=error_msg,
                    total_turns=turn_num + 1,
                ),
            )
            raise Exception(f"A2A delegation failed: {error_msg}")

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

    finally:
        task.description = original_task_description
