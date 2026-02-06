"""A2A agent wrapping logic for metaclass integration.

Wraps agent classes with A2A delegation capabilities.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import json
from types import MethodType
from typing import TYPE_CHECKING, Any, NamedTuple

from a2a.types import Role, TaskState
from pydantic import BaseModel, ValidationError

from crewai.a2a.config import A2AClientConfig, A2AConfig
from crewai.a2a.extensions.base import (
    A2AExtension,
    ConversationState,
    ExtensionRegistry,
)
from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.templates import (
    AVAILABLE_AGENTS_TEMPLATE,
    CONVERSATION_TURN_INFO_TEMPLATE,
    PREVIOUS_A2A_CONVERSATION_TEMPLATE,
    REMOTE_AGENT_RESPONSE_NOTICE,
    UNAVAILABLE_AGENTS_NOTICE_TEMPLATE,
)
from crewai.a2a.types import AgentResponseProtocol
from crewai.a2a.utils.agent_card import (
    afetch_agent_card,
    fetch_agent_card,
    inject_a2a_server_methods,
)
from crewai.a2a.utils.delegation import (
    aexecute_a2a_delegation,
    execute_a2a_delegation,
)
from crewai.a2a.utils.response_model import get_a2a_agents_and_response_model
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConversationCompletedEvent,
    A2AMessageSentEvent,
)
from crewai.lite_agent_output import LiteAgentOutput
from crewai.task import Task


if TYPE_CHECKING:
    from a2a.types import AgentCard, Message

    from crewai.agent.core import Agent
    from crewai.tools.base_tool import BaseTool


class DelegationContext(NamedTuple):
    """Context prepared for A2A delegation.

    Groups all the values needed to execute a delegation to a remote A2A agent.
    """

    a2a_agents: list[A2AConfig | A2AClientConfig]
    agent_response_model: type[BaseModel] | None
    current_request: str
    agent_id: str
    agent_config: A2AConfig | A2AClientConfig
    context_id: str | None
    task_id: str | None
    metadata: dict[str, Any] | None
    extensions: dict[str, Any] | None
    reference_task_ids: list[str]
    original_task_description: str
    max_turns: int


class DelegationState(NamedTuple):
    """Mutable state for A2A delegation loop.

    Groups values that may change during delegation turns.
    """

    current_request: str
    context_id: str | None
    task_id: str | None
    reference_task_ids: list[str]
    conversation_history: list[Message]
    agent_card: AgentCard | None
    agent_card_dict: dict[str, Any] | None
    agent_name: str | None


def wrap_agent_with_a2a_instance(
    agent: Agent, extension_registry: ExtensionRegistry | None = None
) -> None:
    """Wrap an agent instance's task execution and kickoff methods with A2A support.

    This function modifies the agent instance by wrapping its execute_task,
    aexecute_task, kickoff, and kickoff_async methods to add A2A delegation
    capabilities. Should only be called when the agent has a2a configuration set.

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

    original_kickoff = agent.kickoff.__func__  # type: ignore[attr-defined]
    original_kickoff_async = agent.kickoff_async.__func__  # type: ignore[attr-defined]

    @wraps(original_kickoff)
    def kickoff_with_a2a(
        self: Agent,
        messages: str | list[Any],
        response_format: type[Any] | None = None,
        input_files: dict[str, Any] | None = None,
    ) -> Any:
        """Execute agent kickoff with A2A delegation support."""
        if not self.a2a:
            return original_kickoff(self, messages, response_format, input_files)

        a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)

        if not a2a_agents:
            return original_kickoff(self, messages, response_format, input_files)

        return _kickoff_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_kickoff=original_kickoff,
            messages=messages,
            response_format=response_format,
            input_files=input_files,
            agent_response_model=agent_response_model,
            extension_registry=extension_registry,
        )

    @wraps(original_kickoff_async)
    async def kickoff_async_with_a2a(
        self: Agent,
        messages: str | list[Any],
        response_format: type[Any] | None = None,
        input_files: dict[str, Any] | None = None,
    ) -> Any:
        """Execute agent kickoff with A2A delegation support."""
        if not self.a2a:
            return await original_kickoff_async(
                self, messages, response_format, input_files
            )

        a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)

        if not a2a_agents:
            return await original_kickoff_async(
                self, messages, response_format, input_files
            )

        return await _akickoff_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_kickoff_async=original_kickoff_async,
            messages=messages,
            response_format=response_format,
            input_files=input_files,
            agent_response_model=agent_response_model,
            extension_registry=extension_registry,
        )

    object.__setattr__(agent, "kickoff", MethodType(kickoff_with_a2a, agent))
    object.__setattr__(
        agent, "kickoff_async", MethodType(kickoff_async_with_a2a, agent)
    )

    inject_a2a_server_methods(agent)


def _fetch_card_from_config(
    config: A2AConfig | A2AClientConfig,
) -> tuple[A2AConfig | A2AClientConfig, AgentCard | Exception]:
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
    a2a_agents: list[A2AConfig | A2AClientConfig],
) -> tuple[dict[str, AgentCard], dict[str, str]]:
    """Fetch agent cards concurrently for multiple A2A agents.

    Args:
        a2a_agents: List of A2A agent configurations

    Returns:
        Tuple of (agent_cards dict, failed_agents dict mapping endpoint to error message)
    """
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    if not a2a_agents:
        return agent_cards, failed_agents

    max_workers = min(len(a2a_agents), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_fn: Callable[..., str],
    task: Task,
    agent_response_model: type[BaseModel] | None,
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

    task.description, _, extension_states = _augment_prompt_with_a2a(
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
                agent_response, extension_states
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
                    _extension_registry=extension_registry,
                )
            task.output_pydantic = None
            return agent_response.message

        return raw_result
    finally:
        task.description = original_description
        if task.output_pydantic is not None:
            task.output_pydantic = original_output_pydantic
        task.response_model = original_response_model


def _kickoff_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_kickoff: Callable[..., LiteAgentOutput],
    messages: str | list[Any],
    response_format: type[Any] | None,
    input_files: dict[str, Any] | None,
    agent_response_model: type[BaseModel] | None,
    extension_registry: ExtensionRegistry,
) -> LiteAgentOutput:
    """Execute kickoff with A2A delegation support (sync).

    Args:
        self: The agent instance.
        a2a_agents: List of A2A agent configurations.
        original_kickoff: The original kickoff method.
        messages: Messages to send to the agent.
        response_format: Optional response format.
        input_files: Optional input files.
        agent_response_model: Optional agent response model.
        extension_registry: Registry of A2A extensions.

    Returns:
        LiteAgentOutput from kickoff or A2A delegation.
    """
    if isinstance(messages, str):
        description = messages
    else:
        content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        description = content if isinstance(content, str) else ""

    if not description:
        return original_kickoff(self, messages, response_format, input_files)

    fake_task = Task(
        description=description,
        agent=self,
        expected_output="Result from A2A delegation",
        input_files=input_files or {},
    )

    agent_cards, failed_agents = _fetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        return original_kickoff(self, messages, response_format, input_files)

    fake_task.description, _, extension_states = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
        extension_registry=extension_registry,
    )
    fake_task.response_model = agent_response_model

    try:
        result: LiteAgentOutput = original_kickoff(
            self, messages, agent_response_model or response_format, input_files
        )
        agent_response = _parse_agent_response(
            raw_result=result.raw, agent_response_model=agent_response_model
        )

        if extension_registry and isinstance(agent_response, BaseModel):
            agent_response = extension_registry.process_response_with_all(
                agent_response, extension_states
            )

        if isinstance(agent_response, BaseModel) and isinstance(
            agent_response, AgentResponseProtocol
        ):
            if agent_response.is_a2a:

                def _kickoff_adapter(
                    self_: Agent,
                    _task: Task,
                    _context: str | None,
                    _tools: list[Any] | None,
                ) -> str:
                    fmt = (
                        _task.response_model or agent_response_model or response_format
                    )
                    output: LiteAgentOutput = original_kickoff(
                        self_, messages, fmt, input_files
                    )
                    return output.raw

                result_str = _delegate_to_a2a(
                    self,
                    agent_response=agent_response,
                    task=fake_task,
                    original_fn=_kickoff_adapter,
                    context=None,
                    tools=None,
                    agent_cards=agent_cards,
                    original_task_description=description,
                    _extension_registry=extension_registry,
                )
                return LiteAgentOutput(
                    raw=result_str,
                    pydantic=None,
                    agent_role=self.role,
                    usage_metrics=None,
                    messages=[],
                )
            return LiteAgentOutput(
                raw=agent_response.message,
                pydantic=None,
                agent_role=self.role,
                usage_metrics=result.usage_metrics,
                messages=result.messages,
            )

        return result
    finally:
        fake_task.description = description


async def _akickoff_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_kickoff_async: Callable[..., Coroutine[Any, Any, LiteAgentOutput]],
    messages: str | list[Any],
    response_format: type[Any] | None,
    input_files: dict[str, Any] | None,
    agent_response_model: type[BaseModel] | None,
    extension_registry: ExtensionRegistry,
) -> LiteAgentOutput:
    """Execute kickoff with A2A delegation support (async).

    Args:
        self: The agent instance.
        a2a_agents: List of A2A agent configurations.
        original_kickoff_async: The original kickoff_async method.
        messages: Messages to send to the agent.
        response_format: Optional response format.
        input_files: Optional input files.
        agent_response_model: Optional agent response model.
        extension_registry: Registry of A2A extensions.

    Returns:
        LiteAgentOutput from kickoff or A2A delegation.
    """
    if isinstance(messages, str):
        description = messages
    else:
        content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        description = content if isinstance(content, str) else ""

    if not description:
        return await original_kickoff_async(
            self, messages, response_format, input_files
        )

    fake_task = Task(
        description=description,
        agent=self,
        expected_output="Result from A2A delegation",
        input_files=input_files or {},
    )

    agent_cards, failed_agents = await _afetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        return await original_kickoff_async(
            self, messages, response_format, input_files
        )

    fake_task.description, _, extension_states = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
        extension_registry=extension_registry,
    )
    fake_task.response_model = agent_response_model

    try:
        result: LiteAgentOutput = await original_kickoff_async(
            self, messages, agent_response_model or response_format, input_files
        )
        agent_response = _parse_agent_response(
            raw_result=result.raw, agent_response_model=agent_response_model
        )

        if extension_registry and isinstance(agent_response, BaseModel):
            agent_response = extension_registry.process_response_with_all(
                agent_response, extension_states
            )

        if isinstance(agent_response, BaseModel) and isinstance(
            agent_response, AgentResponseProtocol
        ):
            if agent_response.is_a2a:

                async def _kickoff_adapter(
                    self_: Agent,
                    _task: Task,
                    _context: str | None,
                    _tools: list[Any] | None,
                ) -> str:
                    fmt = (
                        _task.response_model or agent_response_model or response_format
                    )
                    output: LiteAgentOutput = await original_kickoff_async(
                        self_, messages, fmt, input_files
                    )
                    return output.raw

                result_str = await _adelegate_to_a2a(
                    self,
                    agent_response=agent_response,
                    task=fake_task,
                    original_fn=_kickoff_adapter,
                    context=None,
                    tools=None,
                    agent_cards=agent_cards,
                    original_task_description=description,
                    _extension_registry=extension_registry,
                )
                return LiteAgentOutput(
                    raw=result_str,
                    pydantic=None,
                    agent_role=self.role,
                    usage_metrics=None,
                    messages=[],
                )
            return LiteAgentOutput(
                raw=agent_response.message,
                pydantic=None,
                agent_role=self.role,
                usage_metrics=result.usage_metrics,
                messages=result.messages,
            )

        return result
    finally:
        fake_task.description = description


def _augment_prompt_with_a2a(
    a2a_agents: list[A2AConfig | A2AClientConfig],
    task_description: str,
    agent_cards: Mapping[str, AgentCard | dict[str, Any]],
    conversation_history: list[Message] | None = None,
    turn_num: int = 0,
    max_turns: int | None = None,
    failed_agents: dict[str, str] | None = None,
    extension_registry: ExtensionRegistry | None = None,
    remote_status_notice: str = "",
) -> tuple[str, bool, dict[type[A2AExtension], ConversationState]]:
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
        remote_status_notice: Optional notice about remote agent status to append

    Returns:
        Tuple of (augmented prompt, disable_structured_output flag, extension_states dict)
    """

    if not agent_cards:
        return task_description, False, {}

    agents_text = ""

    for config in a2a_agents:
        if config.endpoint in agent_cards:
            card = agent_cards[config.endpoint]
            if isinstance(card, dict):
                filtered = {
                    k: v
                    for k, v in card.items()
                    if k in {"description", "url", "skills"} and v is not None
                }
                agents_text += f"\n{json.dumps(filtered, indent=2)}\n"
            else:
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

    augmented_prompt = f"""{task_description}

IMPORTANT: You have the ability to delegate this task to remote A2A agents.
{agents_text}
{history_text}{turn_info}{remote_status_notice}

"""

    if extension_registry:
        augmented_prompt = extension_registry.augment_prompt_with_all(
            augmented_prompt, extension_states
        )

    return augmented_prompt, disable_structured_output, extension_states


def _parse_agent_response(
    raw_result: str | dict[str, Any], agent_response_model: type[BaseModel] | None
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
    from_task: Any | None = None,
    from_agent: Any | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    agent_card: dict[str, Any] | None = None,
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
                        from_task=from_task,
                        from_agent=from_agent,
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        agent_card=agent_card,
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
            from_task=from_task,
            from_agent=from_agent,
            endpoint=endpoint,
            a2a_agent_name=a2a_agent_name,
            agent_card=agent_card,
        ),
    )
    raise Exception(f"A2A conversation exceeded maximum turns ({max_turns})")


def _emit_delegation_failed(
    error_msg: str,
    turn_num: int,
    from_task: Any | None,
    from_agent: Any | None,
    endpoint: str | None,
    a2a_agent_name: str | None,
    agent_card: dict[str, Any] | None,
) -> str:
    """Emit failure event and return formatted error message."""
    crewai_event_bus.emit(
        None,
        A2AConversationCompletedEvent(
            status="failed",
            final_result=None,
            error=error_msg,
            total_turns=turn_num + 1,
            from_task=from_task,
            from_agent=from_agent,
            endpoint=endpoint,
            a2a_agent_name=a2a_agent_name,
            agent_card=agent_card,
        ),
    )
    return f"A2A delegation failed: {error_msg}"


def _process_response_result(
    raw_result: str,
    disable_structured_output: bool,
    turn_num: int,
    agent_role: str,
    agent_response_model: type[BaseModel] | None,
    extension_registry: ExtensionRegistry | None = None,
    extension_states: dict[type[A2AExtension], ConversationState] | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    agent_card: dict[str, Any] | None = None,
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
                from_task=from_task,
                from_agent=from_agent,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AConversationCompletedEvent(
                status="completed",
                final_result=result_text,
                error=None,
                total_turns=final_turn_number,
                from_task=from_task,
                from_agent=from_agent,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                agent_card=agent_card,
            ),
        )
        return result_text, None

    llm_response = _parse_agent_response(
        raw_result=raw_result, agent_response_model=agent_response_model
    )

    if extension_registry and isinstance(llm_response, BaseModel):
        llm_response = extension_registry.process_response_with_all(
            llm_response, extension_states or {}
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
                    from_task=from_task,
                    from_agent=from_agent,
                    endpoint=endpoint,
                    a2a_agent_name=a2a_agent_name,
                ),
            )
            crewai_event_bus.emit(
                None,
                A2AConversationCompletedEvent(
                    status="completed",
                    final_result=str(llm_response.message),
                    error=None,
                    total_turns=final_turn_number,
                    from_task=from_task,
                    from_agent=from_agent,
                    endpoint=endpoint,
                    a2a_agent_name=a2a_agent_name,
                    agent_card=agent_card,
                ),
            )
            return llm_response.message, None
        return None, llm_response.message

    return str(raw_result), None


def _prepare_agent_cards_dict(
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: Mapping[str, AgentCard | dict[str, Any]] | None,
) -> dict[str, AgentCard | dict[str, Any]]:
    """Prepare agent cards dictionary from result and existing cards.

    Shared logic for both sync and async response handlers.
    """
    agent_cards_dict: dict[str, AgentCard | dict[str, Any]] = (
        dict(agent_cards) if agent_cards else {}
    )
    if "agent_card" in a2a_result and agent_id not in agent_cards_dict:
        agent_cards_dict[agent_id] = a2a_result["agent_card"]
    return agent_cards_dict


def _init_delegation_state(
    ctx: DelegationContext,
    agent_cards: dict[str, AgentCard] | None,
) -> DelegationState:
    """Initialize delegation state from context and agent cards.

    Args:
        ctx: Delegation context with config and settings.
        agent_cards: Pre-fetched agent cards.

    Returns:
        Initial delegation state for the conversation loop.
    """
    current_agent_card = agent_cards.get(ctx.agent_id) if agent_cards else None
    return DelegationState(
        current_request=ctx.current_request,
        context_id=ctx.context_id,
        task_id=ctx.task_id,
        reference_task_ids=list(ctx.reference_task_ids),
        conversation_history=[],
        agent_card=current_agent_card,
        agent_card_dict=current_agent_card.model_dump() if current_agent_card else None,
        agent_name=current_agent_card.name if current_agent_card else None,
    )


def _get_turn_context(
    agent_config: A2AConfig | A2AClientConfig,
) -> tuple[Any | None, list[str] | None]:
    """Get context for a delegation turn.

    Returns:
        Tuple of (agent_branch, accepted_output_modes).
    """
    console_formatter = getattr(crewai_event_bus, "_console", None)
    agent_branch = None
    if console_formatter:
        agent_branch = getattr(
            console_formatter, "current_agent_branch", None
        ) or getattr(console_formatter, "current_task_branch", None)

    accepted_output_modes = None
    if isinstance(agent_config, A2AClientConfig):
        accepted_output_modes = agent_config.accepted_output_modes

    return agent_branch, accepted_output_modes


def _prepare_delegation_context(
    self: Agent,
    agent_response: AgentResponseProtocol,
    task: Task,
    original_task_description: str | None,
) -> DelegationContext:
    """Prepare delegation context from agent response and task.

    Shared logic for both sync and async delegation.

    Returns:
        DelegationContext with all values needed for delegation.
    """
    a2a_agents, agent_response_model = get_a2a_agents_and_response_model(self.a2a)
    agent_ids = tuple(config.endpoint for config in a2a_agents)
    current_request = str(agent_response.message)

    if not a2a_agents:
        raise ValueError("No A2A agents configured for delegation")

    if isinstance(agent_response, AgentResponseProtocol) and agent_response.a2a_ids:
        agent_id = agent_response.a2a_ids[0]
    else:
        agent_id = agent_ids[0]

    if agent_id not in agent_ids:
        raise ValueError(f"Unknown A2A agent ID: {agent_id} not in {agent_ids}")

    agent_config = next(filter(lambda x: x.endpoint == agent_id, a2a_agents), None)
    if agent_config is None:
        raise ValueError(f"Agent configuration not found for endpoint: {agent_id}")
    task_config = task.config or {}

    if original_task_description is None:
        original_task_description = task.description

    return DelegationContext(
        a2a_agents=a2a_agents,
        agent_response_model=agent_response_model,
        current_request=current_request,
        agent_id=agent_id,
        agent_config=agent_config,
        context_id=task_config.get("context_id"),
        task_id=task_config.get("task_id"),
        metadata=task_config.get("metadata"),
        extensions=task_config.get("extensions"),
        reference_task_ids=task_config.get("reference_task_ids", []),
        original_task_description=original_task_description,
        max_turns=agent_config.max_turns,
    )


def _handle_task_completion(
    a2a_result: TaskStateResult,
    task: Task,
    task_id_config: str | None,
    reference_task_ids: list[str],
    agent_config: A2AConfig | A2AClientConfig,
    turn_num: int,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    agent_card: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, list[str], str]:
    """Handle task completion state including reference task updates.

    When a remote task completes, this function:
    1. Adds the completed task_id to reference_task_ids (if not already present)
    2. Clears task_id_config to signal that a new task ID should be generated for next turn
    3. Updates task.config with the reference list for subsequent A2A calls

    The reference_task_ids list tracks all completed tasks in this conversation chain,
    allowing the remote agent to maintain context across multi-turn interactions.

    Shared logic for both sync and async delegation.

    Args:
        a2a_result: Result from A2A delegation containing task status.
        task: CrewAI Task object to update with reference IDs.
        task_id_config: Current task ID (will be added to references if task completed).
        reference_task_ids: Mutable list of completed task IDs (updated in place).
        agent_config: A2A configuration with trust settings.
        turn_num: Current turn number.
        from_task: Optional CrewAI Task for event metadata.
        from_agent: Optional CrewAI Agent for event metadata.
        endpoint: A2A endpoint URL.
        a2a_agent_name: Name of remote A2A agent.
        agent_card: Agent card dict for event metadata.

    Returns:
        Tuple of (result_if_trusted, updated_task_id, updated_reference_task_ids, remote_notice).
        - result_if_trusted: Final result if trust_remote_completion_status=True, else None
        - updated_task_id: None (cleared to generate new ID for next turn)
        - updated_reference_task_ids: The mutated list with completed task added
        - remote_notice: Template notice about remote agent response
    """
    remote_notice = ""
    if a2a_result["status"] == TaskState.completed:
        remote_notice = REMOTE_AGENT_RESPONSE_NOTICE

        if task_id_config is not None and task_id_config not in reference_task_ids:
            reference_task_ids.append(task_id_config)

            if task.config is None:
                task.config = {}
            task.config["reference_task_ids"] = list(reference_task_ids)

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
                    from_task=from_task,
                    from_agent=from_agent,
                    endpoint=endpoint,
                    a2a_agent_name=a2a_agent_name,
                    agent_card=agent_card,
                ),
            )
            return str(result_text), task_id_config, reference_task_ids, remote_notice

    return None, task_id_config, reference_task_ids, remote_notice


def _handle_agent_response_and_continue(
    self: Agent,
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: dict[str, AgentCard] | None,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_task_description: str,
    conversation_history: list[Message],
    turn_num: int,
    max_turns: int,
    task: Task,
    original_fn: Callable[..., str],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_response_model: type[BaseModel] | None,
    extension_registry: ExtensionRegistry | None = None,
    remote_status_notice: str = "",
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    agent_card: dict[str, Any] | None = None,
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

    (
        task.description,
        disable_structured_output,
        extension_states,
    ) = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_task_description,
        conversation_history=conversation_history,
        turn_num=turn_num,
        max_turns=max_turns,
        agent_cards=agent_cards_dict,
        remote_status_notice=remote_status_notice,
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
        extension_registry=extension_registry,
        extension_states=extension_states,
        from_task=task,
        from_agent=self,
        endpoint=endpoint,
        a2a_agent_name=a2a_agent_name,
        agent_card=agent_card,
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
    _extension_registry: ExtensionRegistry | None = None,
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
        _extension_registry: Optional registry of A2A extensions (unused, reserved for future use)

    Returns:
        Result from A2A agent

    Raises:
        ImportError: If a2a-sdk is not installed
    """
    ctx = _prepare_delegation_context(
        self, agent_response, task, original_task_description
    )
    state = _init_delegation_state(ctx, agent_cards)
    current_request = state.current_request
    context_id = state.context_id
    task_id = state.task_id
    reference_task_ids = state.reference_task_ids
    conversation_history = state.conversation_history

    try:
        for turn_num in range(ctx.max_turns):
            agent_branch, accepted_output_modes = _get_turn_context(ctx.agent_config)

            a2a_result = execute_a2a_delegation(
                endpoint=ctx.agent_config.endpoint,
                auth=ctx.agent_config.auth,
                timeout=ctx.agent_config.timeout,
                task_description=current_request,
                context_id=context_id,
                task_id=task_id,
                reference_task_ids=reference_task_ids,
                metadata=ctx.metadata,
                extensions=ctx.extensions,
                conversation_history=conversation_history,
                agent_id=ctx.agent_id,
                agent_role=Role.user,
                agent_branch=agent_branch,
                response_model=ctx.agent_config.response_model,
                turn_number=turn_num + 1,
                updates=ctx.agent_config.updates,
                transport=ctx.agent_config.transport,
                from_task=task,
                from_agent=self,
                client_extensions=getattr(ctx.agent_config, "extensions", None),
                accepted_output_modes=accepted_output_modes,
                input_files=task.input_files,
            )

            conversation_history = a2a_result.get("history", [])

            if conversation_history:
                latest_message = conversation_history[-1]
                if latest_message.task_id is not None:
                    task_id = latest_message.task_id
                if latest_message.context_id is not None:
                    context_id = latest_message.context_id

            if a2a_result["status"] in [TaskState.completed, TaskState.input_required]:
                trusted_result, task_id, reference_task_ids, remote_notice = (
                    _handle_task_completion(
                        a2a_result,
                        task,
                        task_id,
                        reference_task_ids,
                        ctx.agent_config,
                        turn_num,
                        from_task=task,
                        from_agent=self,
                        endpoint=ctx.agent_config.endpoint,
                        a2a_agent_name=state.agent_name,
                        agent_card=state.agent_card_dict,
                    )
                )
                if trusted_result is not None:
                    return trusted_result

                final_result, next_request = _handle_agent_response_and_continue(
                    self=self,
                    a2a_result=a2a_result,
                    agent_id=ctx.agent_id,
                    agent_cards=agent_cards,
                    a2a_agents=ctx.a2a_agents,
                    original_task_description=ctx.original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=ctx.max_turns,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_response_model=ctx.agent_response_model,
                    extension_registry=_extension_registry,
                    remote_status_notice=remote_notice,
                    endpoint=ctx.agent_config.endpoint,
                    a2a_agent_name=state.agent_name,
                    agent_card=state.agent_card_dict,
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
                agent_id=ctx.agent_id,
                agent_cards=agent_cards,
                a2a_agents=ctx.a2a_agents,
                original_task_description=ctx.original_task_description,
                conversation_history=conversation_history,
                turn_num=turn_num,
                max_turns=ctx.max_turns,
                task=task,
                original_fn=original_fn,
                context=context,
                tools=tools,
                agent_response_model=ctx.agent_response_model,
                extension_registry=_extension_registry,
                endpoint=ctx.agent_config.endpoint,
                a2a_agent_name=state.agent_name,
                agent_card=state.agent_card_dict,
            )

            if final_result is not None:
                return final_result

            if next_request is not None:
                current_request = next_request
                continue

            return _emit_delegation_failed(
                error_msg,
                turn_num,
                task,
                self,
                ctx.agent_config.endpoint,
                state.agent_name,
                state.agent_card_dict,
            )

        return _handle_max_turns_exceeded(
            conversation_history,
            ctx.max_turns,
            from_task=task,
            from_agent=self,
            endpoint=ctx.agent_config.endpoint,
            a2a_agent_name=state.agent_name,
            agent_card=state.agent_card_dict,
        )

    finally:
        task.description = ctx.original_task_description


async def _afetch_card_from_config(
    config: A2AConfig | A2AClientConfig,
) -> tuple[A2AConfig | A2AClientConfig, AgentCard | Exception]:
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
    a2a_agents: list[A2AConfig | A2AClientConfig],
) -> tuple[dict[str, AgentCard], dict[str, str]]:
    """Fetch agent cards concurrently for multiple A2A agents using asyncio."""
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    if not a2a_agents:
        return agent_cards, failed_agents

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
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    task: Task,
    agent_response_model: type[BaseModel] | None,
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

    task.description, _, extension_states = _augment_prompt_with_a2a(
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
                agent_response, extension_states
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
                    _extension_registry=extension_registry,
                )
            task.output_pydantic = None
            return agent_response.message

        return raw_result
    finally:
        task.description = original_description
        if task.output_pydantic is not None:
            task.output_pydantic = original_output_pydantic
        task.response_model = original_response_model


async def _ahandle_agent_response_and_continue(
    self: Agent,
    a2a_result: TaskStateResult,
    agent_id: str,
    agent_cards: dict[str, AgentCard] | None,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_task_description: str,
    conversation_history: list[Message],
    turn_num: int,
    max_turns: int,
    task: Task,
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    context: str | None,
    tools: list[BaseTool] | None,
    agent_response_model: type[BaseModel] | None,
    extension_registry: ExtensionRegistry | None = None,
    remote_status_notice: str = "",
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    agent_card: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Async version of _handle_agent_response_and_continue."""
    agent_cards_dict = _prepare_agent_cards_dict(a2a_result, agent_id, agent_cards)

    (
        task.description,
        disable_structured_output,
        extension_states,
    ) = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_task_description,
        conversation_history=conversation_history,
        turn_num=turn_num,
        max_turns=max_turns,
        agent_cards=agent_cards_dict,
        remote_status_notice=remote_status_notice,
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
        extension_registry=extension_registry,
        extension_states=extension_states,
        from_task=task,
        from_agent=self,
        endpoint=endpoint,
        a2a_agent_name=a2a_agent_name,
        agent_card=agent_card,
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
    _extension_registry: ExtensionRegistry | None = None,
) -> str:
    """Async version of _delegate_to_a2a."""
    ctx = _prepare_delegation_context(
        self, agent_response, task, original_task_description
    )
    state = _init_delegation_state(ctx, agent_cards)
    current_request = state.current_request
    context_id = state.context_id
    task_id = state.task_id
    reference_task_ids = state.reference_task_ids
    conversation_history = state.conversation_history

    try:
        for turn_num in range(ctx.max_turns):
            agent_branch, accepted_output_modes = _get_turn_context(ctx.agent_config)

            a2a_result = await aexecute_a2a_delegation(
                endpoint=ctx.agent_config.endpoint,
                auth=ctx.agent_config.auth,
                timeout=ctx.agent_config.timeout,
                task_description=current_request,
                context_id=context_id,
                task_id=task_id,
                reference_task_ids=reference_task_ids,
                metadata=ctx.metadata,
                extensions=ctx.extensions,
                conversation_history=conversation_history,
                agent_id=ctx.agent_id,
                agent_role=Role.user,
                agent_branch=agent_branch,
                response_model=ctx.agent_config.response_model,
                turn_number=turn_num + 1,
                transport=ctx.agent_config.transport,
                updates=ctx.agent_config.updates,
                from_task=task,
                from_agent=self,
                client_extensions=getattr(ctx.agent_config, "extensions", None),
                accepted_output_modes=accepted_output_modes,
                input_files=task.input_files,
            )

            conversation_history = a2a_result.get("history", [])

            if conversation_history:
                latest_message = conversation_history[-1]
                if latest_message.task_id is not None:
                    task_id = latest_message.task_id
                if latest_message.context_id is not None:
                    context_id = latest_message.context_id

            if a2a_result["status"] in [TaskState.completed, TaskState.input_required]:
                trusted_result, task_id, reference_task_ids, remote_notice = (
                    _handle_task_completion(
                        a2a_result,
                        task,
                        task_id,
                        reference_task_ids,
                        ctx.agent_config,
                        turn_num,
                        from_task=task,
                        from_agent=self,
                        endpoint=ctx.agent_config.endpoint,
                        a2a_agent_name=state.agent_name,
                        agent_card=state.agent_card_dict,
                    )
                )
                if trusted_result is not None:
                    return trusted_result

                final_result, next_request = await _ahandle_agent_response_and_continue(
                    self=self,
                    a2a_result=a2a_result,
                    agent_id=ctx.agent_id,
                    agent_cards=agent_cards,
                    a2a_agents=ctx.a2a_agents,
                    original_task_description=ctx.original_task_description,
                    conversation_history=conversation_history,
                    turn_num=turn_num,
                    max_turns=ctx.max_turns,
                    task=task,
                    original_fn=original_fn,
                    context=context,
                    tools=tools,
                    agent_response_model=ctx.agent_response_model,
                    extension_registry=_extension_registry,
                    remote_status_notice=remote_notice,
                    endpoint=ctx.agent_config.endpoint,
                    a2a_agent_name=state.agent_name,
                    agent_card=state.agent_card_dict,
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
                agent_id=ctx.agent_id,
                agent_cards=agent_cards,
                a2a_agents=ctx.a2a_agents,
                original_task_description=ctx.original_task_description,
                conversation_history=conversation_history,
                turn_num=turn_num,
                max_turns=ctx.max_turns,
                task=task,
                original_fn=original_fn,
                context=context,
                tools=tools,
                agent_response_model=ctx.agent_response_model,
                extension_registry=_extension_registry,
                endpoint=ctx.agent_config.endpoint,
                a2a_agent_name=state.agent_name,
                agent_card=state.agent_card_dict,
            )

            if final_result is not None:
                return final_result

            if next_request is not None:
                current_request = next_request
                continue

            return _emit_delegation_failed(
                error_msg,
                turn_num,
                task,
                self,
                ctx.agent_config.endpoint,
                state.agent_name,
                state.agent_card_dict,
            )

        return _handle_max_turns_exceeded(
            conversation_history,
            ctx.max_turns,
            from_task=task,
            from_agent=self,
            endpoint=ctx.agent_config.endpoint,
            a2a_agent_name=state.agent_name,
            agent_card=state.agent_card_dict,
        )

    finally:
        task.description = ctx.original_task_description
