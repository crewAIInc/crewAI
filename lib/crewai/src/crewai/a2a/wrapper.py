"""A2A agent wrapping logic for metaclass integration.

Wraps agent classes with A2A delegation capabilities. Each remote A2A agent
is exposed to the local LLM as a BaseTool; the local agent's tool-call loop
drives multi-turn delegation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import contextvars
from functools import wraps
import json
from types import MethodType
from typing import TYPE_CHECKING, Any

from crewai.a2a.config import A2AClientConfig, A2AConfig
from crewai.a2a.extensions.base import ExtensionRegistry
from crewai.a2a.templates import (
    AVAILABLE_AGENTS_TEMPLATE,
    UNAVAILABLE_AGENTS_NOTICE_TEMPLATE,
)
from crewai.a2a.tools import A2ADelegationState, build_a2a_tools
from crewai.a2a.utils.agent_card import (
    afetch_agent_card,
    fetch_agent_card,
    inject_a2a_server_methods,
)
from crewai.a2a.utils.response_model import extract_a2a_client_configs
from crewai.lite_agent_output import LiteAgentOutput
from crewai.task import Task


if TYPE_CHECKING:
    from a2a.types import AgentCard

    from crewai.agent.core import Agent
    from crewai.tools.base_tool import BaseTool


def wrap_agent_with_a2a_instance(
    agent: Agent, extension_registry: ExtensionRegistry | None = None
) -> None:
    """Wrap an agent instance's task execution and kickoff methods with A2A support.

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

        a2a_agents = extract_a2a_client_configs(self.a2a)

        return _execute_task_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_fn=original_execute_task,
            task=task,
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

        a2a_agents = extract_a2a_client_configs(self.a2a)

        return await _aexecute_task_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_fn=original_aexecute_task,
            task=task,
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

        a2a_agents = extract_a2a_client_configs(self.a2a)
        if not a2a_agents:
            return original_kickoff(self, messages, response_format, input_files)

        return _kickoff_with_a2a(
            self=self,
            a2a_agents=a2a_agents,
            original_kickoff=original_kickoff,
            messages=messages,
            response_format=response_format,
            input_files=input_files,
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

        a2a_agents = extract_a2a_client_configs(self.a2a)
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
    """Fetch an agent card synchronously, capturing any exception."""
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
    """Fetch agent cards concurrently for multiple A2A agents."""
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    if not a2a_agents:
        return agent_cards, failed_agents

    max_workers = min(len(a2a_agents), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                contextvars.copy_context().run, _fetch_card_from_config, config
            ): config
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


async def _afetch_card_from_config(
    config: A2AConfig | A2AClientConfig,
) -> tuple[A2AConfig | A2AClientConfig, AgentCard | Exception]:
    """Async variant of :func:`_fetch_card_from_config`."""
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
    """Async variant of :func:`_fetch_agent_cards_concurrently`."""
    agent_cards: dict[str, AgentCard] = {}
    failed_agents: dict[str, str] = {}

    if not a2a_agents:
        return agent_cards, failed_agents

    fetch_tasks = [_afetch_card_from_config(config) for config in a2a_agents]
    results = await asyncio.gather(*fetch_tasks)

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


def _build_unavailable_notice(failed_agents: dict[str, str]) -> str:
    text = ""
    for endpoint, error in failed_agents.items():
        text += f"  - {endpoint}: {error}\n"
    return UNAVAILABLE_AGENTS_NOTICE_TEMPLATE.substitute(unavailable_agents=text)


def _augment_prompt_with_a2a(
    a2a_agents: list[A2AConfig | A2AClientConfig],
    task_description: str,
    agent_cards: Mapping[str, AgentCard | dict[str, Any]],
    failed_agents: dict[str, str] | None = None,
) -> str:
    """Add A2A delegation context (the available agent cards) to a prompt.

    Tool-call mechanics are documented inside the template; this only renders
    the cards themselves so the LLM can see each remote agent's capabilities.
    """
    if not agent_cards:
        return task_description

    agents_text = ""
    for config in a2a_agents:
        if config.endpoint in agent_cards:
            card = agent_cards[config.endpoint]
            if isinstance(card, dict):
                filtered = {
                    k: v
                    for k, v in card.items()
                    if k in {"name", "description", "url", "skills"} and v is not None
                }
                agents_text += f"\n{json.dumps(filtered, indent=2)}\n"
            else:
                agents_text += (
                    "\n"
                    + card.model_dump_json(
                        indent=2,
                        exclude_none=True,
                        include={"name", "description", "url", "skills"},
                    )
                    + "\n"
                )

    failed_agents = failed_agents or {}
    if failed_agents:
        agents_text += "\n<!-- Unavailable Agents -->\n"
        for endpoint, error in failed_agents.items():
            agents_text += (
                f"\n<!-- Agent: {endpoint}\n"
                f"     Status: Unavailable\n"
                f"     Error: {error} -->\n"
            )

    available = AVAILABLE_AGENTS_TEMPLATE.substitute(available_a2a_agents=agents_text)
    return f"{task_description}\n{available}\n"


def _execute_task_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_fn: Callable[..., str],
    task: Task,
    context: str | None,
    tools: list[BaseTool] | None,
    extension_registry: ExtensionRegistry,
) -> str:
    """Wrap execute_task with A2A delegation logic (sync)."""
    original_description: str = task.description
    agent_cards, failed_agents = _fetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        task.description = (
            f"{original_description}{_build_unavailable_notice(failed_agents)}"
        )
        try:
            return original_fn(self, task, context, tools)
        finally:
            task.description = original_description

    state = A2ADelegationState(
        agent=self, task=task, extension_registry=extension_registry
    )
    a2a_tools = build_a2a_tools(a2a_agents, agent_cards, state)

    augmented = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
    )
    if extension_registry:
        augmented = extension_registry.augment_prompt_with_all(augmented, {})

    task.description = augmented
    combined_tools: list[BaseTool] = [*(tools or []), *a2a_tools]
    try:
        return original_fn(self, task, context, combined_tools)
    finally:
        task.description = original_description


async def _aexecute_task_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_fn: Callable[..., Coroutine[Any, Any, str]],
    task: Task,
    context: str | None,
    tools: list[BaseTool] | None,
    extension_registry: ExtensionRegistry,
) -> str:
    """Async variant of :func:`_execute_task_with_a2a`."""
    original_description: str = task.description
    agent_cards, failed_agents = await _afetch_agent_cards_concurrently(a2a_agents)

    if not agent_cards and a2a_agents and failed_agents:
        task.description = (
            f"{original_description}{_build_unavailable_notice(failed_agents)}"
        )
        try:
            return await original_fn(self, task, context, tools)
        finally:
            task.description = original_description

    state = A2ADelegationState(
        agent=self, task=task, extension_registry=extension_registry
    )
    a2a_tools = build_a2a_tools(a2a_agents, agent_cards, state)

    augmented = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=original_description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
    )
    if extension_registry:
        augmented = extension_registry.augment_prompt_with_all(augmented, {})

    task.description = augmented
    combined_tools: list[BaseTool] = [*(tools or []), *a2a_tools]
    try:
        return await original_fn(self, task, context, combined_tools)
    finally:
        task.description = original_description


@contextlib.contextmanager
def _temporarily_extend_tools(agent: Agent, extra: list[BaseTool]) -> Iterator[None]:
    """Append ``extra`` to ``agent.tools`` for the lifetime of the context."""
    if not extra:
        yield
        return
    original_tools = agent.tools
    if original_tools is None:
        agent.tools = list(extra)
    else:
        agent.tools = [*original_tools, *extra]
    try:
        yield
    finally:
        agent.tools = original_tools


def _kickoff_description(messages: str | list[Any]) -> str:
    if isinstance(messages, str):
        return messages
    content = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        None,
    )
    return content if isinstance(content, str) else ""


def _kickoff_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_kickoff: Callable[..., LiteAgentOutput],
    messages: str | list[Any],
    response_format: type[Any] | None,
    input_files: dict[str, Any] | None,
    extension_registry: ExtensionRegistry,
) -> LiteAgentOutput:
    """Execute kickoff with A2A delegation support (sync)."""
    description = _kickoff_description(messages)
    if not description:
        return original_kickoff(self, messages, response_format, input_files)

    agent_cards, failed_agents = _fetch_agent_cards_concurrently(a2a_agents)
    if not agent_cards and a2a_agents and failed_agents:
        return original_kickoff(self, messages, response_format, input_files)

    fake_task = Task(
        description=description,
        agent=self,
        expected_output="Result from A2A delegation",
        input_files=input_files or {},
    )
    state = A2ADelegationState(
        agent=self, task=fake_task, extension_registry=extension_registry
    )
    a2a_tools = build_a2a_tools(a2a_agents, agent_cards, state)

    augmented = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
    )
    if extension_registry:
        augmented = extension_registry.augment_prompt_with_all(augmented, {})

    if isinstance(messages, str):
        wrapped_messages: str | list[Any] = augmented
    else:
        wrapped_messages = [*messages, {"role": "user", "content": augmented}]

    with _temporarily_extend_tools(self, a2a_tools):
        return original_kickoff(self, wrapped_messages, response_format, input_files)


async def _akickoff_with_a2a(
    self: Agent,
    a2a_agents: list[A2AConfig | A2AClientConfig],
    original_kickoff_async: Callable[..., Coroutine[Any, Any, LiteAgentOutput]],
    messages: str | list[Any],
    response_format: type[Any] | None,
    input_files: dict[str, Any] | None,
    extension_registry: ExtensionRegistry,
) -> LiteAgentOutput:
    """Execute kickoff with A2A delegation support (async)."""
    description = _kickoff_description(messages)
    if not description:
        return await original_kickoff_async(
            self, messages, response_format, input_files
        )

    agent_cards, failed_agents = await _afetch_agent_cards_concurrently(a2a_agents)
    if not agent_cards and a2a_agents and failed_agents:
        return await original_kickoff_async(
            self, messages, response_format, input_files
        )

    fake_task = Task(
        description=description,
        agent=self,
        expected_output="Result from A2A delegation",
        input_files=input_files or {},
    )
    state = A2ADelegationState(
        agent=self, task=fake_task, extension_registry=extension_registry
    )
    a2a_tools = build_a2a_tools(a2a_agents, agent_cards, state)

    augmented = _augment_prompt_with_a2a(
        a2a_agents=a2a_agents,
        task_description=description,
        agent_cards=agent_cards,
        failed_agents=failed_agents,
    )
    if extension_registry:
        augmented = extension_registry.augment_prompt_with_all(augmented, {})

    if isinstance(messages, str):
        wrapped_messages: str | list[Any] = augmented
    else:
        wrapped_messages = [*messages, {"role": "user", "content": augmented}]

    with _temporarily_extend_tools(self, a2a_tools):
        return await original_kickoff_async(
            self, wrapped_messages, response_format, input_files
        )
