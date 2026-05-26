"""Tool-based A2A delegation.

Each remote A2A agent is exposed to the local LLM as a BaseTool. The local
agent's normal tool-call loop drives multi-turn delegation: each tool call is
one turn against the remote agent. Per-endpoint conversation state lives in
``A2ADelegationState`` and is shared across the tools built for a single task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from a2a.types import Role, TaskState
from pydantic import BaseModel, Field, PrivateAttr

from crewai.a2a.config import A2AClientConfig, A2AConfig
from crewai.a2a.extensions.base import (
    A2AExtension,
    ConversationState,
    ExtensionRegistry,
)
from crewai.a2a.task_helpers import TaskStateResult
from crewai.a2a.utils.delegation import aexecute_a2a_delegation, execute_a2a_delegation
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import A2AConversationCompletedEvent
from crewai.tools.base_tool import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from a2a.types import AgentCard, Message

    from crewai.task import Task


_DELEGATE_PREFIX = "delegate_to_"


@dataclass
class _EndpointState:
    """Mutable per-endpoint conversation state across tool calls."""

    conversation_history: list[Message] = field(default_factory=list)
    context_id: str | None = None
    task_id: str | None = None
    reference_task_ids: list[str] = field(default_factory=list)
    turn_count: int = 0


@dataclass
class A2ADelegationState:
    """State shared across all A2A delegation tools for a single task execution."""

    agent: Any
    task: Task
    extension_registry: ExtensionRegistry | None = None
    _per_endpoint: dict[str, _EndpointState] = field(default_factory=dict)

    def _state_for(self, endpoint: str) -> _EndpointState:
        return self._per_endpoint.setdefault(endpoint, _EndpointState())

    def _initial_ids_from_task(self, state: _EndpointState) -> None:
        if state.turn_count > 0:
            return
        task_config = self.task.config or {}
        if state.context_id is None:
            state.context_id = task_config.get("context_id")
        if state.task_id is None:
            state.task_id = task_config.get("task_id")
        if not state.reference_task_ids:
            state.reference_task_ids = list(task_config.get("reference_task_ids", []))

    def delegate(
        self,
        config: A2AConfig | A2AClientConfig,
        agent_card: AgentCard | None,
        message: str,
    ) -> str:
        """Run one delegation turn against ``config.endpoint``.

        Returns the remote agent's response text, suitable for handing back to
        the local LLM as a tool result.
        """
        return _run_delegation(self, config, agent_card, message, sync=True)

    async def adelegate(
        self,
        config: A2AConfig | A2AClientConfig,
        agent_card: AgentCard | None,
        message: str,
    ) -> str:
        """Async variant of :meth:`delegate`."""
        return await _run_delegation_async(self, config, agent_card, message)


class _A2ADelegationArgs(BaseModel):
    """Argument schema for A2A delegation tools."""

    message: str = Field(
        ...,
        description=(
            "The question or task to send to the remote agent. Be specific and "
            "self-contained: the remote agent does not see your other tools or "
            "your prior reasoning."
        ),
    )


class A2ADelegationTool(BaseTool):
    """BaseTool that delegates one turn of conversation to a remote A2A agent.

    Each instance is bound to a specific A2A endpoint via ``_config``. Calling
    ``_run`` or ``_arun`` advances that endpoint's conversation by one turn and
    returns the remote agent's response text.
    """

    args_schema: type[BaseModel] = _A2ADelegationArgs

    _config: A2AConfig | A2AClientConfig = PrivateAttr()
    _agent_card: AgentCard | None = PrivateAttr(default=None)
    _state: A2ADelegationState = PrivateAttr()

    def _run(self, message: str) -> str:
        return self._state.delegate(self._config, self._agent_card, message)

    async def _arun(self, message: str) -> str:
        return await self._state.adelegate(self._config, self._agent_card, message)


def build_a2a_tools(
    a2a_agents: list[A2AConfig | A2AClientConfig],
    agent_cards: dict[str, AgentCard],
    state: A2ADelegationState,
) -> list[BaseTool]:
    """Build one ``A2ADelegationTool`` per available A2A agent.

    Tool names collide-disambiguate with a numeric suffix; agents whose cards
    failed to fetch are skipped.
    """
    tools: list[BaseTool] = []
    used_names: set[str] = set()
    for config in a2a_agents:
        card = agent_cards.get(config.endpoint)
        if card is None:
            continue
        name = _build_tool_name(card.name or "remote_agent", used_names)
        used_names.add(name)
        tool = A2ADelegationTool(
            name=name,
            description=_build_tool_description(card),
            max_usage_count=config.max_turns,
        )
        tool._config = config
        tool._agent_card = card
        tool._state = state
        tools.append(tool)
    return tools


def _build_tool_name(card_name: str, used: set[str]) -> str:
    base = sanitize_tool_name(f"{_DELEGATE_PREFIX}{card_name}")
    if base not in used:
        return base
    for i in range(2, 1000):
        candidate = sanitize_tool_name(f"{base}_{i}")
        if candidate not in used:
            return candidate
    raise ValueError(f"Could not generate unique tool name for {card_name!r}")


def _build_tool_description(card: AgentCard) -> str:
    lines: list[str] = [f"Delegate a task to the remote A2A agent {card.name!r}."]
    if card.description:
        lines.append(card.description.strip())
    if card.skills:
        skill_names = ", ".join(s.name for s in card.skills if s.name)
        if skill_names:
            lines.append(f"Capabilities: {skill_names}.")
    lines.append(
        "Use this tool only when the question matches the agent's capabilities. "
        "After receiving a response, prefer answering directly unless you need "
        "another round-trip."
    )
    return "\n".join(lines)


def _run_delegation(
    state: A2ADelegationState,
    config: A2AConfig | A2AClientConfig,
    agent_card: AgentCard | None,
    message: str,
    *,
    sync: bool,
) -> str:
    endpoint_state = state._state_for(config.endpoint)
    state._initial_ids_from_task(endpoint_state)

    extension_states = _extract_extension_states(state, endpoint_state)
    metadata = _merged_metadata(state, endpoint_state, extension_states)
    agent_branch, accepted_output_modes = _turn_context(config)

    a2a_result = execute_a2a_delegation(
        endpoint=config.endpoint,
        auth=config.auth,
        timeout=config.timeout,
        task_description=message,
        context_id=endpoint_state.context_id,
        task_id=endpoint_state.task_id,
        reference_task_ids=endpoint_state.reference_task_ids,
        metadata=metadata or None,
        extensions=(state.task.config or {}).get("extensions"),
        conversation_history=endpoint_state.conversation_history,
        agent_id=config.endpoint,
        agent_role=Role.user,
        agent_branch=agent_branch,
        response_model=config.response_model,
        turn_number=endpoint_state.turn_count + 1,
        updates=config.updates,
        transport=config.transport,
        from_task=state.task,
        from_agent=state.agent,
        client_extensions=getattr(config, "extensions", None),
        accepted_output_modes=accepted_output_modes,
        input_files=state.task.input_files,
    )
    return _finalize_turn(
        state, endpoint_state, config, agent_card, a2a_result, extension_states
    )


async def _run_delegation_async(
    state: A2ADelegationState,
    config: A2AConfig | A2AClientConfig,
    agent_card: AgentCard | None,
    message: str,
) -> str:
    endpoint_state = state._state_for(config.endpoint)
    state._initial_ids_from_task(endpoint_state)

    extension_states = _extract_extension_states(state, endpoint_state)
    metadata = _merged_metadata(state, endpoint_state, extension_states)
    agent_branch, accepted_output_modes = _turn_context(config)

    a2a_result = await aexecute_a2a_delegation(
        endpoint=config.endpoint,
        auth=config.auth,
        timeout=config.timeout,
        task_description=message,
        context_id=endpoint_state.context_id,
        task_id=endpoint_state.task_id,
        reference_task_ids=endpoint_state.reference_task_ids,
        metadata=metadata or None,
        extensions=(state.task.config or {}).get("extensions"),
        conversation_history=endpoint_state.conversation_history,
        agent_id=config.endpoint,
        agent_role=Role.user,
        agent_branch=agent_branch,
        response_model=config.response_model,
        turn_number=endpoint_state.turn_count + 1,
        updates=config.updates,
        transport=config.transport,
        from_task=state.task,
        from_agent=state.agent,
        client_extensions=getattr(config, "extensions", None),
        accepted_output_modes=accepted_output_modes,
        input_files=state.task.input_files,
    )
    return _finalize_turn(
        state, endpoint_state, config, agent_card, a2a_result, extension_states
    )


def _extract_extension_states(
    state: A2ADelegationState,
    endpoint_state: _EndpointState,
) -> dict[type[A2AExtension], ConversationState]:
    if state.extension_registry and endpoint_state.conversation_history:
        return state.extension_registry.extract_all_states(
            endpoint_state.conversation_history
        )
    return {}


def _merged_metadata(
    state: A2ADelegationState,
    endpoint_state: _EndpointState,
    extension_states: dict[type[A2AExtension], ConversationState],
) -> dict[str, Any]:
    task_config = state.task.config or {}
    metadata: dict[str, Any] = dict(task_config.get("metadata") or {})
    if state.extension_registry and extension_states:
        metadata.update(state.extension_registry.prepare_all_metadata(extension_states))
    return metadata


def _turn_context(
    config: A2AConfig | A2AClientConfig,
) -> tuple[Any | None, list[str] | None]:
    console_formatter = getattr(crewai_event_bus, "_console", None)
    agent_branch = None
    if console_formatter:
        agent_branch = getattr(
            console_formatter, "current_agent_branch", None
        ) or getattr(console_formatter, "current_task_branch", None)

    accepted_output_modes = None
    if isinstance(config, A2AClientConfig):
        accepted_output_modes = config.accepted_output_modes
    return agent_branch, accepted_output_modes


def _finalize_turn(
    state: A2ADelegationState,
    endpoint_state: _EndpointState,
    config: A2AConfig | A2AClientConfig,
    agent_card: AgentCard | None,
    a2a_result: TaskStateResult,
    extension_states: dict[type[A2AExtension], ConversationState],
) -> str:
    endpoint_state.conversation_history = list(a2a_result.get("history", []))
    if endpoint_state.conversation_history:
        latest = endpoint_state.conversation_history[-1]
        if latest.task_id is not None:
            endpoint_state.task_id = latest.task_id
        if latest.context_id is not None:
            endpoint_state.context_id = latest.context_id

    endpoint_state.turn_count += 1
    status = a2a_result.get("status")

    if status == TaskState.completed:
        if (
            endpoint_state.task_id is not None
            and endpoint_state.task_id not in endpoint_state.reference_task_ids
        ):
            endpoint_state.reference_task_ids.append(endpoint_state.task_id)
            if state.task.config is None:
                state.task.config = {}
            state.task.config["reference_task_ids"] = list(
                endpoint_state.reference_task_ids
            )
        endpoint_state.task_id = None

        result_text = str(a2a_result.get("result", ""))
        crewai_event_bus.emit(
            None,
            A2AConversationCompletedEvent(
                status="completed",
                final_result=result_text,
                error=None,
                total_turns=endpoint_state.turn_count,
                from_task=state.task,
                from_agent=state.agent,
                endpoint=config.endpoint,
                a2a_agent_name=agent_card.name if agent_card else None,
                agent_card=agent_card.model_dump() if agent_card else None,
            ),
        )
        return _apply_response_extensions(state, result_text, extension_states)

    if status == TaskState.input_required:
        result_text = str(a2a_result.get("result", ""))
        return _apply_response_extensions(state, result_text, extension_states)

    error_msg = a2a_result.get("error", "Unknown error")
    crewai_event_bus.emit(
        None,
        A2AConversationCompletedEvent(
            status="failed",
            final_result=None,
            error=error_msg,
            total_turns=endpoint_state.turn_count,
            from_task=state.task,
            from_agent=state.agent,
            endpoint=config.endpoint,
            a2a_agent_name=agent_card.name if agent_card else None,
            agent_card=agent_card.model_dump() if agent_card else None,
        ),
    )
    return f"Remote agent error: {error_msg}"


def _apply_response_extensions(
    state: A2ADelegationState,
    response_text: str,
    extension_states: dict[type[A2AExtension], ConversationState],
) -> str:
    if not state.extension_registry:
        return response_text
    processed = state.extension_registry.process_response_with_all(
        response_text, extension_states
    )
    return processed if isinstance(processed, str) else str(processed)
