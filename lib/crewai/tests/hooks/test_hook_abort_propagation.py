"""A model-call deny must reach the caller as a deny.

Two layers used to erase it. The LLM layer caught ``HookAborted`` and returned
``False``, which every provider turned into ``ValueError("LLM call blocked...")``
— losing the reason, the source, and any way to tell a policy decision from a
provider outage. Downstream, every internal model call is wrapped in
``except Exception`` so a provider hiccup degrades instead of failing the run,
and those handlers then absorbed the flattened deny, often retrying the very
call that was just denied.

A deny also owes the started event a terminal one, and owes it an honest label:
it is a decision, not an outage the provider caused.

These use a real ``LLM`` with a real ``pre_model_call`` hook: the deny fires
inside ``dispatch`` before the provider is reached, so nothing here needs a
network or a cassette.
"""

from __future__ import annotations

from typing import Any

from crewai.agent import Agent
from crewai.agents.planner_observer import PlannerObserver
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.knowledge_events import (
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
)
from crewai.events.types.llm_events import LLMCallFailedEvent
from crewai.events.types.observation_events import (
    StepObservationFailedEvent,
    StepObservationStartedEvent,
)
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    on,
)
from crewai.hooks.llm_hooks import register_before_llm_call_hook
from crewai.llm import LLM
from crewai.memory.analyze import (
    analyze_for_consolidation,
    analyze_for_save,
    analyze_query,
    extract_memories_from_content,
)
from crewai.memory.types import MemoryRecord
from crewai.task import Task
from crewai.utilities.converter import Converter
from crewai.utilities.planning_types import TodoItem
import pytest
from pydantic import BaseModel

from ..utils import wait_for_event_handlers


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_all()
    yield
    # These calls emit LLM events whose handlers run on a pool; draining them
    # here keeps a straggler from firing inside an unrelated test.
    wait_for_event_handlers()
    clear_all()


@pytest.fixture
def denied_calls() -> list[Any]:
    return []


@pytest.fixture
def denying_llm(denied_calls: list[Any]) -> LLM:
    @on(InterceptionPoint.PRE_MODEL_CALL)
    def deny(ctx: Any) -> None:
        denied_calls.append(ctx)
        raise HookAborted(reason="no model calls allowed", source="policy")

    return LLM(model="gpt-4o-mini")


class _Person(BaseModel):
    name: str


def test_a_raised_deny_keeps_its_reason_out_of_the_llm_layer(denying_llm):
    with pytest.raises(HookAborted) as exc:
        denying_llm.call([{"role": "user", "content": "hi"}])

    assert exc.value.reason == "no model calls allowed"
    assert exc.value.source == "policy"


def test_the_boolean_convention_still_blocks_with_the_documented_error():
    register_before_llm_call_hook(lambda _ctx: False)

    with pytest.raises(ValueError, match="LLM call blocked by before_llm_call hook"):
        LLM(model="gpt-4o-mini").call([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    "denied_helper",
    [
        lambda llm: extract_memories_from_content("some content", llm),
        lambda llm: analyze_query("a query", ["/"], None, llm),
        lambda llm: analyze_for_save("some content", ["/"], [], llm),
        lambda llm: analyze_for_consolidation(
            "new content",
            [MemoryRecord(id="1", content="old content", scope="/")],
            llm,
        ),
    ],
    ids=["extract", "query", "save", "consolidate"],
)
def test_memory_analysis_surfaces_a_deny_instead_of_a_safe_default(
    denied_helper, denying_llm, denied_calls
):
    with pytest.raises(HookAborted):
        denied_helper(denying_llm)

    assert denied_calls, "the helper never reached the model"


def test_a_denied_conversion_is_not_retried(denying_llm, denied_calls):
    converter = Converter(
        text="Name: Ada",
        llm=denying_llm,
        model=_Person,
        instructions="Extract the person",
        max_attempts=3,
    )

    with pytest.raises(HookAborted):
        converter.to_pydantic()

    assert len(denied_calls) == 1


def test_a_denied_knowledge_query_still_reports_the_failure_it_started(denying_llm):
    agent = Agent(
        role="Researcher",
        goal="Answer questions",
        backstory="You look things up.",
        llm=denying_llm,
    )
    task = Task(
        description="What is the capital of France?",
        expected_output="A city name.",
        agent=agent,
    )
    started: list[KnowledgeQueryStartedEvent] = []
    failed: list[KnowledgeQueryFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(KnowledgeQueryStartedEvent)
        def _on_started(_source: Any, event: KnowledgeQueryStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(KnowledgeQueryFailedEvent)
        def _on_failed(_source: Any, event: KnowledgeQueryFailedEvent) -> None:
            failed.append(event)

        with pytest.raises(HookAborted):
            agent._get_knowledge_search_query(task.description, task)

        wait_for_event_handlers()

    assert len(started) == 1
    assert len(failed) == 1
    assert failed[0].error == "no model calls allowed"


def test_a_denied_step_observation_still_reports_the_failure_it_started(denying_llm):
    agent = Agent(
        role="Planner",
        goal="Plan work",
        backstory="You plan.",
        llm=denying_llm,
    )
    observer = PlannerObserver(agent=agent, task=None)
    step = TodoItem(step_number=1, description="do the thing", result="done")
    started: list[StepObservationStartedEvent] = []
    failed: list[StepObservationFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(StepObservationStartedEvent)
        def _on_started(_source: Any, event: StepObservationStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(StepObservationFailedEvent)
        def _on_failed(_source: Any, event: StepObservationFailedEvent) -> None:
            failed.append(event)

        with pytest.raises(HookAborted):
            observer.observe(step, "done", [], [])

        wait_for_event_handlers()

    assert len(started) == 1
    assert len(failed) == 1
    assert failed[0].error == "no model calls allowed"
    assert failed[0].step_number == 1


@pytest.mark.asyncio
async def test_an_async_call_is_denied_like_a_sync_one(denying_llm):
    with pytest.raises(HookAborted) as exc:
        await denying_llm.acall([{"role": "user", "content": "hi"}])

    assert exc.value.reason == "no model calls allowed"
    assert exc.value.source == "policy"


def test_a_denied_structured_conversion_reaches_the_caller(denying_llm):
    # The function-calling path goes through Instructor, which reaches the
    # provider client without passing through ``llm.call``.
    assert denying_llm.supports_function_calling()
    converter = Converter(
        text="Name: Ada",
        llm=denying_llm,
        model=_Person,
        instructions="Extract the person",
        max_attempts=1,
    )

    with pytest.raises(HookAborted):
        converter.to_json()


def test_a_deny_is_not_reported_as_a_provider_failure(denying_llm):
    failures: list[LLMCallFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_failed(_source: Any, event: LLMCallFailedEvent) -> None:
            failures.append(event)

        with pytest.raises(HookAborted):
            denying_llm.call([{"role": "user", "content": "hi"}])

        wait_for_event_handlers()

    assert len(failures) == 1
    assert failures[0].error == "LLM call denied by policy: no model calls allowed"


def test_the_boolean_convention_is_not_reported_as_a_provider_failure():
    register_before_llm_call_hook(lambda _ctx: False)
    failures: list[LLMCallFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_failed(_source: Any, event: LLMCallFailedEvent) -> None:
            failures.append(event)

        with pytest.raises(ValueError):
            LLM(model="gpt-4o-mini").call([{"role": "user", "content": "hi"}])

        wait_for_event_handlers()

    assert len(failures) == 1
    assert failures[0].error == (
        "LLM call denied by hook: LLM call blocked by before_llm_call hook"
    )
