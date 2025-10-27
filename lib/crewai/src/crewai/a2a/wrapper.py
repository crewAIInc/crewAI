"""A2A agent wrapping logic for metaclass integration.

Wraps agent classes with A2A delegation capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from a2a.types import Role
from pydantic import BaseModel, ValidationError

from crewai.a2a.config import A2AConfig
from crewai.a2a.templates import (
    AVAILABLE_AGENTS_TEMPLATE,
    PREVIOUS_A2A_CONVERSATION_TEMPLATE,
)
from crewai.a2a.utils import (
    create_agent_response_model,
    execute_a2a_delegation,
    fetch_agent_card,
    get_a2a_agents_and_response_model,
)


if TYPE_CHECKING:
    from a2a.types import AgentCard, Message

    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


@runtime_checkable
class AgentResponseProtocol(Protocol):
    """Protocol for the dynamically created AgentResponse model."""

    a2a_ids: tuple[str, ...]
    message: str
    is_a2a: bool


def wrap_agent_with_a2a(
    namespace: dict[str, Any], bases: tuple[type, ...]
) -> dict[str, Callable[..., Any]]:
    """Create A2A-wrapped methods for an agent class.

    Args:
        namespace: The class namespace dictionary
        bases: Base classes of the agent

    Returns:
        Dictionary of methods to add to the class namespace
    """
    original_execute_task = _find_execute_task(namespace, bases)

    if not original_execute_task:
        return {}

    def execute_task(
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
            return original_execute_task(self, task, context, tools)

        a2a_configs: list[A2AConfig] = [self.a2a] if isinstance(self.a2a, A2AConfig) else self.a2a

        agent_ids = tuple(config.endpoint for config in a2a_configs)
        agent_response_model: type[BaseModel] = create_agent_response_model(agent_ids)

        return _execute_task_with_a2a(
            self=self,
            a2a_agents=a2a_configs,
            original_fn=original_execute_task,
            task=task,
            agent_response_model=agent_response_model,
            context=context,
            tools=tools,
        )

    return {
        "execute_task": execute_task,
        "_execute_task_with_a2a": _execute_task_with_a2a,
        "_augment_prompt_with_a2a": _augment_prompt_with_a2a,
        "_parse_agent_response": _parse_agent_response,
        "_delegate_to_a2a": _delegate_to_a2a,
    }


def _find_execute_task(
    namespace: dict[str, Any], bases: tuple[type, ...]
) -> Callable[..., str] | None:
    """Find the execute_task method in namespace or base classes.

    Args:
        namespace: The class namespace dictionary
        bases: Base classes to search

    Returns:
        The execute_task method or None if not found
    """
    original_execute_task = namespace.get("execute_task")

    if not original_execute_task:
        for base in bases:
            if hasattr(base, "execute_task"):
                original_execute_task = base.execute_task
                break

    return original_execute_task


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

    agent_cards: dict[str, AgentCard] = {}
    for config in a2a_agents:
        try:
            agent_cards[config.endpoint] = fetch_agent_card(
                endpoint=config.endpoint,
                auth=config.auth,
                timeout=config.timeout,
            )
        except Exception as e:  # noqa: PERF203
            print(e)

    task.description = _augment_prompt_with_a2a(self, a2a_agents=a2a_agents, task_description=original_description, agent_cards=agent_cards)
    task.response_model = agent_response_model

    try:
        raw_result = original_fn(self, task, context, tools)
        agent_response = _parse_agent_response(self, raw_result=raw_result, agent_response_model=agent_response_model)
        print(agent_response)
        if isinstance(agent_response, BaseModel) and isinstance(agent_response, AgentResponseProtocol):
            if agent_response.is_a2a:
                print("delegating to a2a")
                return _delegate_to_a2a(
                    self, agent_response=agent_response, task=task, original_fn=original_fn, context=context, tools=tools, agent_cards=agent_cards, original_task_description=original_description
                )
            return str(agent_response.message)
        print("return raw result, no delegation")
        return raw_result
    finally:
        task.description = original_description
        task.output_pydantic = original_output_pydantic
        task.response_model = original_response_model


def _augment_prompt_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig],
    task_description: str,
    agent_cards: dict[str, AgentCard],
    conversation_history: list[Message] | None = None,
    turn_num: int = 0,
    max_turns: int | None = None,
) -> str:
    """Add A2A delegation instructions to prompt.

    Args:
        self: The agent instance
        a2a_agents: Dictionary of A2A agent configurations
        task_description: Original task description
        agent_cards: dictionary mapping agent IDs to AgentCards
        conversation_history: Previous A2A Messages from conversation
        turn_num: Current turn number (0-indexed)
        max_turns: Maximum allowed turns (from config)

    Returns:
        Augmented task description with A2A instructions
    """

    if not agent_cards:
        agent_cards = {}

    agents_text = ""

    for config in a2a_agents:
        card = agent_cards[config.endpoint]
        agents_text += f"\n{card.model_dump_json(indent=2, exclude_none=True)}\n"


    agents_text = AVAILABLE_AGENTS_TEMPLATE.substitute(available_a2a_agents=agents_text)

    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            history_text += f"\n{msg.model_dump_json(indent=2, exclude_none=True)}\n"

    history_text = PREVIOUS_A2A_CONVERSATION_TEMPLATE.substitute(previous_a2a_conversation=history_text)
    turn_info = ""

    if max_turns is not None and conversation_history:
        # Use the passed turn_num (0-indexed) and convert to 1-indexed for display
        turn_count = turn_num + 1
        turn_info = f"\n\nConversation Progress: Turn {turn_count} of {max_turns}\n"
        if turn_count >= max_turns:
            turn_info += "⚠️ CRITICAL: This is the FINAL turn. You MUST conclude the conversation now.\n"
            turn_info += "Set is_a2a=false and provide your final response to complete the task.\n"
        elif turn_count == max_turns - 1:
            turn_info += "⚠️ WARNING: Next turn will be the last. Consider wrapping up the conversation.\n"

    return f"""{task_description}

IMPORTANT: You have the ability to delegate this task to remote A2A agents.

{agents_text}
{history_text}{turn_info}


"""


def _parse_agent_response(
    self: Agent,
    raw_result: str | dict[str, Any],
    agent_response_model: type[BaseModel]
) -> BaseModel | str:
    """Parse LLM output as AgentResponse or return raw agent response.

    Args:
        self: The agent instance
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


def _delegate_to_a2a(
    self: Agent,
    agent_response: BaseModel,
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
    agent_ids = (config.endpoint for config in a2a_agents)
    current_request = str(agent_response.message)
    agent_id = agent_response.a2a_ids[0]

    if agent_id not in agent_ids:
        raise ValueError(f"Unknown A2A agent ID(s): {agent_response.a2a_ids} not in {agent_ids}")

    agent_config = next(filter(lambda x: x.endpoint == agent_id, a2a_agents))
    task_config = task.config or {}
    context_id = task_config.get("context_id")
    task_id_config = task_config.get("task_id")
    reference_task_ids = task_config.get("reference_task_ids")
    metadata = task_config.get("metadata")
    extensions = task_config.get("extensions")
    # Use the passed original_task_description if available, otherwise fall back to task.description
    if original_task_description is None:
        original_task_description = task.description
    conversation_history: list[Message] = []
    max_turns = agent_config.max_turns

    # Use the pre-fetched agent card if available
    agent_card_obj: AgentCard | None = None
    if agent_cards and agent_id in agent_cards:
        agent_card_obj = agent_cards[agent_id]

    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.a2a_events import A2AConversationCompletedEvent

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
                context=None,
                context_id=context_id,
                task_id=task_id_config,
                reference_task_ids=reference_task_ids,
                metadata=metadata,
                extensions=extensions,
                conversation_history=conversation_history,
                agent_id=agent_id,
                agent_role=cast(Role, self.role),
                agent_branch=agent_branch,
                response_model=self.a2a.response_model,
                turn_number=turn_num + 1,
            )
            conversation_history = a2a_result["history"]

            if agent_card_obj is None and "agent_card" in a2a_result:
                agent_card_obj = a2a_result["agent_card"]

            if a2a_result["status"] == "completed":
                result_text = str(a2a_result["result"])

                # Use passed agent_cards or create a new dict with the current card
                agent_cards_dict = agent_cards if agent_cards else ({agent_id: agent_card_obj} if agent_card_obj else {})
                task.description = _augment_prompt_with_a2a(
                    self,
                    a2a_agents=a2a_agents,
                    task_description=original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=max_turns,
                    agent_cards=agent_cards_dict,
                )
                raw_result = original_fn(self, task, context, tools)
                llm_response = _parse_agent_response(self, raw_result=raw_result, agent_response_model=agent_response_model)

                if isinstance(llm_response, BaseModel) and isinstance(llm_response, AgentResponseProtocol):
                    current_request = str(llm_response.message)
                    if not llm_response.is_a2a:
                        from crewai.events.types.a2a_events import A2AMessageSentEvent

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
                        return str(llm_response.message)
                continue

            if a2a_result["status"] == "input_required":
                # Use passed agent_cards or create a new dict with the current card
                agent_cards_dict = agent_cards if agent_cards else ({agent_id: agent_card_obj} if agent_card_obj else {})
                task.description = _augment_prompt_with_a2a(
                    self,
                    a2a_agents=a2a_agents,
                    task_description=original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=max_turns,
                    agent_cards=agent_cards_dict,
                )
                raw_result = original_fn(self, task, context, tools)
                llm_response = _parse_agent_response(self, raw_result=raw_result, agent_response_model=agent_response_model)
                if isinstance(llm_response, BaseModel) and isinstance(llm_response, AgentResponseProtocol):
                    if not llm_response.is_a2a:
                        final_turn_number = turn_num + 1
                        from crewai.events.types.a2a_events import A2AMessageSentEvent

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
                        return str(llm_response.message)
                    current_request = str(llm_response.message)
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
