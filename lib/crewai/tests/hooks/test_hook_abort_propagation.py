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
from crewai.agent.planning_config import PlanningConfig
from crewai.agent.utils import (
    ahandle_knowledge_retrieval,
    handle_knowledge_retrieval,
    handle_reasoning,
)
from crewai.agents.planner_observer import PlannerObserver
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.knowledge_events import (
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
)
from crewai.events.types.llm_events import LLMCallFailedEvent
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.events.types.observation_events import (
    StepObservationFailedEvent,
    StepObservationStartedEvent,
)
from crewai.experimental.agent_executor import AgentExecutor
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
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.converter import Converter
from crewai.utilities.guardrail import process_guardrail
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


def test_the_boolean_convention_is_still_absorbed_by_a_fail_open_handler():
    # Unlike a raised abort, the boolean deny must keep degrading rather than
    # failing the run — otherwise adopting this fix breaks existing hooks.
    register_before_llm_call_hook(lambda _ctx: False)

    analysis = analyze_query("a query", ["/"], None, LLM(model="gpt-4o-mini"))

    assert analysis.recall_queries == ["a query"]


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


def test_a_denied_knowledge_retrieval_does_not_fall_back_to_the_plain_prompt(
    denying_llm,
):
    # the retrieval helper wraps the query rewrite in its own except Exception,
    # so guarding the rewrite alone still let the task run without knowledge
    agent = Agent(
        role="Researcher",
        goal="Answer questions",
        backstory="You look things up.",
        llm=denying_llm,
    )
    # only has to be truthy: the deny fires before any knowledge is queried
    agent.knowledge = object()
    task = Task(
        description="What is the capital of France?",
        expected_output="A city name.",
        agent=agent,
    )

    with pytest.raises(HookAborted):
        handle_knowledge_retrieval(
            agent,
            task,
            "the task prompt",
            {},
            lambda *_args, **_kwargs: [],
            lambda *_args, **_kwargs: [],
        )


@pytest.mark.asyncio
async def test_a_denied_async_knowledge_retrieval_reaches_the_caller(denying_llm):
    agent = Agent(
        role="Researcher",
        goal="Answer questions",
        backstory="You look things up.",
        llm=denying_llm,
    )
    agent.knowledge = object()
    task = Task(
        description="What is the capital of France?",
        expected_output="A city name.",
        agent=agent,
    )

    with pytest.raises(HookAborted):
        await ahandle_knowledge_retrieval(agent, task, "the task prompt", {})


def test_a_denied_plan_stops_the_legacy_planning_path(denying_llm):
    agent = Agent(
        role="Planner",
        goal="Plan work",
        backstory="You plan.",
        llm=denying_llm,
        planning=True,
    )
    task = Task(description="Do the thing", expected_output="A result.", agent=agent)

    with pytest.raises(HookAborted):
        handle_reasoning(agent, task)


def test_a_denied_replan_does_not_keep_executing_the_stale_plan(denying_llm):
    agent = Agent(
        role="Planner",
        goal="Plan work",
        backstory="You plan.",
        llm=denying_llm,
        planning_config=PlanningConfig(
            reasoning_effort="low",
            max_attempts=1,
            max_steps=2,
            max_replans=1,
            max_step_iterations=2,
        ),
    )
    executor = AgentExecutor(agent=agent, llm=denying_llm, task=None)
    executor._kickoff_input = "do the thing"

    with pytest.raises(HookAborted):
        executor._trigger_replan("the first plan failed")


class _DenyingMemory:
    """Stands in for unified memory whose model call was denied upstream."""

    read_only = False
    root_scope = None

    def __init__(self, error: Exception):
        self._error = error

    def drain_writes(self) -> None:
        pass

    def recall(self, *_args: Any, **_kwargs: Any):
        raise self._error

    def extract_memories(self, *_args: Any, **_kwargs: Any):
        raise self._error

    def remember_many(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def test_a_denied_memory_recall_does_not_run_the_task_without_memory():
    agent = Agent(role="Doer", goal="Do", backstory="You do.")
    agent.memory = _DenyingMemory(HookAborted(reason="no", source="policy"))
    task = Task(description="Do the thing", expected_output="A result.", agent=agent)

    with pytest.raises(HookAborted):
        agent._retrieve_memory_context(task, "the task prompt")


def test_an_ordinary_memory_recall_failure_still_degrades():
    # the guard must single out a deny: a broken store has always been allowed
    # to degrade to no memory, and turning that into a failed run is a break
    agent = Agent(role="Doer", goal="Do", backstory="You do.")
    agent.memory = _DenyingMemory(ValueError("vector store is down"))
    task = Task(description="Do the thing", expected_output="A result.", agent=agent)

    assert agent._retrieve_memory_context(task, "the prompt") == "the prompt"


def test_a_denied_memory_save_does_not_report_a_clean_kickoff():
    agent = Agent(role="Doer", goal="Do", backstory="You do.")
    agent.memory = _DenyingMemory(HookAborted(reason="no", source="policy"))

    with pytest.raises(HookAborted):
        agent._save_kickoff_to_memory("the input", "the output")


def test_an_ordinary_memory_save_failure_still_degrades():
    agent = Agent(role="Doer", goal="Do", backstory="You do.")
    agent.memory = _DenyingMemory(ValueError("vector store is down"))

    agent._save_kickoff_to_memory("the input", "the output")


def test_a_denied_guardrail_does_not_read_as_a_failed_validation(denying_llm):
    # returning (False, "Error while validating...") would feed the agent a
    # retry prompt built from a call the policy refused
    guardrail = LLMGuardrail(description="Must be polite", llm=denying_llm)
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )

    with pytest.raises(HookAborted):
        guardrail(task_output)


def test_a_denied_guardrail_still_reports_the_validation_it_started(denying_llm):
    guardrail = LLMGuardrail(description="Must be polite", llm=denying_llm)
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )
    started: list[LLMGuardrailStartedEvent] = []
    completed: list[LLMGuardrailCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def _on_started(_source: Any, event: LLMGuardrailStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_completed(_source: Any, event: LLMGuardrailCompletedEvent) -> None:
            completed.append(event)

        with pytest.raises(HookAborted):
            process_guardrail(output=task_output, guardrail=guardrail, retry_count=0)

        wait_for_event_handlers()

    assert len(started) == 1
    assert len(completed) == 1
    assert completed[0].success is False
    assert "no model calls allowed" in (completed[0].error or "")


def test_a_denied_plan_stops_the_executor_instead_of_running_unplanned(denying_llm):
    # the executor wraps planning in its own except Exception, so guarding the
    # reasoning handler alone still left the agent proceeding with no plan
    agent = Agent(
        role="Planner",
        goal="Plan work",
        backstory="You plan.",
        llm=denying_llm,
        planning_config=PlanningConfig(
            reasoning_effort="low",
            max_attempts=1,
            max_steps=2,
            max_replans=0,
            max_step_iterations=2,
        ),
    )
    executor = AgentExecutor(agent=agent, llm=denying_llm, task=None)
    executor._kickoff_input = "do the thing"

    with pytest.raises(HookAborted):
        executor.generate_plan()


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


def test_a_deny_names_the_hook_that_raised_it_rather_than_its_repr():
    def gate_on_approved_models(_ctx: Any) -> None:
        raise HookAborted(reason="model not approved", source=gate_on_approved_models)

    on(InterceptionPoint.PRE_MODEL_CALL)(gate_on_approved_models)
    failures: list[LLMCallFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMCallFailedEvent)
        def _on_failed(_source: Any, event: LLMCallFailedEvent) -> None:
            failures.append(event)

        with pytest.raises(HookAborted):
            LLM(model="gpt-4o-mini").call([{"role": "user", "content": "hi"}])

        wait_for_event_handlers()

    assert len(failures) == 1
    assert failures[0].error == (
        "LLM call denied by gate_on_approved_models: model not approved"
    )


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
