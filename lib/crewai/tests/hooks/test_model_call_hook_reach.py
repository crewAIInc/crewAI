"""Which model calls the ``pre_model_call`` hooks can see, and how many times.

The LLM layer used to skip the hooks whenever a call carried an agent, assuming
the executor had already dispatched them. That holds inside the executor loop
and left every other agent-bearing call — step observation, planning, plan
synthesis — invisible. The executor now marks the window where it already
dispatched, which is what keeps a call from being seen twice.

The stub stands in for a native provider: it invokes the before hooks the way
every provider does and answers without a network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from crewai.agent import Agent
from crewai.hooks.dispatch import InterceptionPoint, clear_all, on
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import get_llm_response
from crewai.utilities.types import LLMMessage
from crewai_core.printer import Printer
from pydantic import BaseModel
import pytest

from ..utils import wait_for_event_handlers


class StubProviderLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model="stub")

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
        **kwargs: Any,
    ) -> str:
        formatted: list[LLMMessage] = (
            messages
            if isinstance(messages, list)
            else [{"role": "user", "content": messages}]
        )
        self._invoke_before_llm_call_hooks(formatted, from_agent)
        return "Thought: done\nFinal Answer: ok"

    def supports_function_calling(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_all()
    yield
    # A kickoff emits events whose handlers run on a pool; draining them here
    # keeps a straggler from firing inside an unrelated test.
    wait_for_event_handlers()
    clear_all()


@pytest.fixture
def seen_agents() -> list[str | None]:
    roles: list[str | None] = []

    @on(InterceptionPoint.PRE_MODEL_CALL)
    def record(ctx: Any) -> None:
        agent = getattr(ctx, "agent", None)
        roles.append(getattr(agent, "role", None))

    return roles


def test_the_executor_loop_is_seen_exactly_once(seen_agents):
    agent = Agent(
        role="Worker",
        goal="Answer",
        backstory="You answer.",
        llm=StubProviderLLM(),
    )

    assert str(agent.kickoff("say ok")) == "ok"
    assert seen_agents == ["Worker"]


def test_a_direct_call_carrying_an_agent_is_seen_once(seen_agents):
    agent = Agent(
        role="Planner",
        goal="Plan",
        backstory="You plan.",
        llm=StubProviderLLM(),
    )

    StubProviderLLM().call([{"role": "user", "content": "hi"}], from_agent=agent)

    assert seen_agents == ["Planner"]


def test_a_direct_call_without_an_agent_is_seen_once(seen_agents):
    StubProviderLLM().call([{"role": "user", "content": "hi"}])

    assert seen_agents == [None]


def test_a_structured_litellm_call_is_seen_once(seen_agents, monkeypatch):
    pytest.importorskip("litellm")
    import instructor

    from crewai.llm import LLM

    class Answer(BaseModel):
        text: str

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: Answer(text="ok"))
        )
    )
    monkeypatch.setattr(instructor, "from_litellm", lambda *_, **__: client)

    llm = LLM(model="openai/gpt-4o-mini", is_litellm=True)
    llm.call([{"role": "user", "content": "hi"}], response_model=Answer)

    assert seen_agents == [None]


def test_an_agent_call_outside_the_executor_loop_is_seen_once(seen_agents):
    agent = Agent(
        role="Worker",
        goal="Answer",
        backstory="You answer.",
        llm=StubProviderLLM(),
    )

    get_llm_response(
        llm=StubProviderLLM(),
        messages=[{"role": "user", "content": "hi"}],
        callbacks=[],
        printer=Printer(),
        from_agent=agent,
        executor_context=None,
    )

    assert seen_agents == ["Worker"]
