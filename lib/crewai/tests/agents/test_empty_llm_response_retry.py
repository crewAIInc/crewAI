"""An empty LLM answer is retried once, then raises ``EmptyLLMResponseError``."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.hooks.dispatch import clear_all
from crewai.lite_agent import LiteAgent
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM, EmptyLLMResponseError
from crewai.tools import BaseTool
from crewai.utilities.agent_utils import (
    aget_llm_response,
    get_llm_response,
    handle_max_iterations_exceeded,
)
from crewai_core.printer import Printer


EMPTY_MESSAGE = "The provider returned no content for model 'scripted-model'."
FINAL = "Thought: done\nFinal Answer: hello"


class ScriptedLLM(BaseLLM):
    """Returns (or raises) the scripted replies in order, recording each call."""

    def __init__(self, *replies: Any, native_tools: bool = False) -> None:
        super().__init__(model="scripted-model")
        self.replies = list(replies)
        self.native_tools = native_tools
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []

    def supports_function_calling(self) -> bool:
        return self.native_tools

    def call(self, messages: Any, **kwargs: Any) -> Any:
        self.seen_messages.append([dict(m) for m in messages])
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    calls: list[str] = []

    def _run(self, term: str = "default") -> str:
        self.calls.append(term)
        return f"value:{term}"


class Verdict(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def clean_hooks():
    clear_all()
    yield
    crewai_event_bus.flush()
    clear_all()


def _llm_response(llm: BaseLLM, *, use_async: bool) -> Any:
    kwargs: dict[str, Any] = {
        "llm": llm,
        "messages": [{"role": "user", "content": "hi"}],
        "callbacks": [],
        "printer": Printer(),
        "verbose": False,
    }
    if use_async:
        return asyncio.run(aget_llm_response(**kwargs))
    return get_llm_response(**kwargs)



@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "empty",
    [None, "", [], EmptyLLMResponseError("scripted-model")],
    ids=["none", "empty-string", "empty-list", "raised"],
)
def test_one_empty_answer_is_retried_once(empty: Any, use_async: bool) -> None:
    llm = ScriptedLLM(empty, "answer")

    assert _llm_response(llm, use_async=use_async) == "answer"
    assert llm.calls == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_two_empty_answers_raise_typed_error(use_async: bool) -> None:
    llm = ScriptedLLM("", None)

    with pytest.raises(EmptyLLMResponseError, match=EMPTY_MESSAGE) as exc:
        _llm_response(llm, use_async=use_async)

    assert isinstance(exc.value, ValueError)
    assert exc.value.model == "scripted-model"
    assert llm.calls == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "answer",
    [
        "answer",
        [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        Verdict(ok=False),
    ],
    ids=["text", "tool-calls", "structured"],
)
def test_non_empty_answer_is_not_retried(answer: Any, use_async: bool) -> None:
    llm = ScriptedLLM(answer, "unexpected second call")

    assert _llm_response(llm, use_async=use_async) == answer
    assert llm.calls == 1


def test_other_errors_are_not_retried() -> None:
    llm = ScriptedLLM(RuntimeError("provider down"), "answer")

    with pytest.raises(RuntimeError, match="provider down"):
        _llm_response(llm, use_async=False)

    assert llm.calls == 1



class CountingLimiter:
    """An RPM limiter callback that records each admitted request."""

    def __init__(self) -> None:
        self.admitted = 0

    def __call__(self) -> bool:
        self.admitted += 1
        return True


def _executor(limiter: CountingLimiter) -> SimpleNamespace:
    return SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
        request_within_rpm_limit=limiter,
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("replies", "admitted"),
    [(("answer",), 0), (("", "answer"), 1), (("", ""), 1)],
    ids=["no-retry", "retry", "retry-then-raise"],
)
def test_retry_is_admitted_by_the_rpm_limiter(
    replies: tuple[Any, ...], admitted: int, use_async: bool
) -> None:
    # The caller admits the first request; only the retry is admitted here.
    limiter = CountingLimiter()
    llm = ScriptedLLM(*replies)
    kwargs: dict[str, Any] = {
        "llm": llm,
        "messages": [],
        "callbacks": [],
        "printer": Printer(),
        "executor_context": _executor(limiter),
        "verbose": False,
    }

    with contextlib.suppress(EmptyLLMResponseError):
        if use_async:
            asyncio.run(aget_llm_response(**kwargs))
        else:
            get_llm_response(**kwargs)

    assert limiter.admitted == admitted


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("max_rpm", "replies", "minute_waits"),
    [(1, ("", FINAL), 1), (2, ("", FINAL), 0), (1, (FINAL,), 0)],
    ids=["retry-over-limit", "retry-within-limit", "no-retry"],
)
def test_agent_max_rpm_counts_both_physical_attempts(
    max_rpm: int, replies: tuple[Any, ...], minute_waits: int, use_async: bool
) -> None:
    llm = ScriptedLLM(*replies)
    agent = _agent(llm, max_rpm=max_rpm)

    with patch("crewai.utilities.rpm_controller.time") as clock:
        result = _kickoff(agent, use_async)

    assert result.raw == "hello"
    assert llm.calls == len(replies)
    assert clock.sleep.call_count == minute_waits


def test_lite_agent_retry_is_admitted_by_the_rpm_limiter() -> None:
    limiter = CountingLimiter()
    llm = ScriptedLLM("", FINAL)
    agent = LiteAgent(
        role="Tester",
        goal="Answer",
        backstory="Tests",
        llm=llm,
        request_within_rpm_limit=limiter,
    )

    assert agent.kickoff("Say hello").raw == "hello"
    assert llm.calls == 2
    assert limiter.admitted == 2


@pytest.mark.parametrize(
    ("replies", "admitted"),
    [(("Final Answer: 42",), 1), (("", "Final Answer: 42"), 2)],
    ids=["no-retry", "retry"],
)
def test_max_iterations_answer_is_admitted_by_the_rpm_limiter(
    replies: tuple[Any, ...], admitted: int
) -> None:
    limiter = CountingLimiter()
    llm = ScriptedLLM(*replies)

    result = handle_max_iterations_exceeded(
        printer=Printer(),
        messages=[],
        llm=llm,
        callbacks=[],
        verbose=False,
        request_within_rpm_limit=limiter,
    )

    assert result.output == "42"
    assert limiter.admitted == admitted


def test_max_iterations_answer_is_retried_once() -> None:
    llm = ScriptedLLM("", "Final Answer: 42")

    result = handle_max_iterations_exceeded(
        printer=Printer(), messages=[], llm=llm, callbacks=[], verbose=False
    )

    assert result.output == "42"
    assert llm.calls == 2


def test_max_iterations_empty_twice_raises_typed_error() -> None:
    llm = ScriptedLLM(None, "")

    with pytest.raises(EmptyLLMResponseError, match=EMPTY_MESSAGE):
        handle_max_iterations_exceeded(
            printer=Printer(), messages=[], llm=llm, callbacks=[], verbose=False
        )

    assert llm.calls == 2



def _chunk(content: str | None) -> dict[str, Any]:
    return {"id": "chatcmpl-1", "choices": [{"delta": {"content": content}}]}


def _empty_stream() -> list[dict[str, Any]]:
    return [_chunk(None), _chunk("")]


def _text_stream() -> list[dict[str, Any]]:
    return [_chunk("hel"), _chunk("lo")]


def _completion(content: str | None) -> Any:
    from litellm import ModelResponse

    return ModelResponse(
        id="chatcmpl-1",
        choices=[
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    )


async def _aiter(chunks: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for chunk in chunks:
        yield chunk


@pytest.fixture
def llm_events() -> Iterator[list[Any]]:
    with patch.object(crewai_event_bus, "emit") as emit:
        events: list[Any] = []
        emit.side_effect = lambda _source, event, **_kw: events.append(event)
        yield events


def _count(events: list[Any], kind: type) -> int:
    return sum(isinstance(event, kind) for event in events)


def test_litellm_empty_stream_raises_typed_error_with_one_failed_event(
    llm_events: list[Any],
) -> None:
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    with (
        patch("crewai.llm.litellm.completion", return_value=iter(_empty_stream())),
        pytest.raises(
            EmptyLLMResponseError,
            match="The provider returned no content for model 'gpt-4o-mini'.",
        ),
    ):
        llm.call("hi")

    assert _count(llm_events, LLMCallStartedEvent) == 1
    assert _count(llm_events, LLMCallFailedEvent) == 1
    assert _count(llm_events, LLMCallCompletedEvent) == 0


@pytest.mark.parametrize(
    ("stream", "use_async"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["sync", "async", "sync-stream", "async-stream"],
)
def test_litellm_empty_then_text_retries_once_with_events_per_call(
    stream: bool, use_async: bool, llm_events: list[Any]
) -> None:
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=stream)
    if stream:
        replies: list[Any] = [_empty_stream(), _text_stream()]
        sync_side_effect = [iter(r) for r in replies]
        async_side_effect = [_aiter(r) for r in replies]
    else:
        sync_side_effect = [_completion(""), _completion("hello")]
        async_side_effect = [_completion(""), _completion("hello")]

    with (
        patch("crewai.llm.litellm.completion", side_effect=sync_side_effect) as sync,
        patch("crewai.llm.litellm.acompletion", side_effect=async_side_effect) as aio,
    ):
        assert _llm_response(llm, use_async=use_async) == "hello"

    assert (aio if use_async else sync).call_count == 2
    assert _count(llm_events, LLMCallStartedEvent) == 2
    # One terminal event per actual call; sync streaming reports the empty
    # stream as a failure, the others as an empty completion.
    assert (
        _count(llm_events, LLMCallCompletedEvent) + _count(llm_events, LLMCallFailedEvent)
        == 2
    )
    assert _count(llm_events, LLMCallCompletedEvent) >= 1



def _agent(
    llm: BaseLLM, tools: list[BaseTool] | None = None, max_rpm: int | None = None
) -> Agent:
    return Agent(
        role="Tester",
        goal="Answer",
        backstory="Tests empty answers",
        llm=llm,
        tools=tools or [],
        max_retry_limit=0,
        max_rpm=max_rpm,
        verbose=False,
    )


def _kickoff(agent: Agent, use_async: bool) -> Any:
    if use_async:
        return asyncio.run(agent.kickoff_async("Say hello"))
    return agent.kickoff("Say hello")


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_agent_kickoff_recovers_from_one_empty_answer(use_async: bool) -> None:
    llm = ScriptedLLM("", FINAL)

    result = _kickoff(_agent(llm), use_async)

    assert result.raw == "hello"
    assert llm.calls == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_agent_kickoff_raises_typed_error_after_two_empty_answers(
    use_async: bool,
) -> None:
    llm = ScriptedLLM("", "", FINAL)

    with pytest.raises(EmptyLLMResponseError, match=EMPTY_MESSAGE):
        _kickoff(_agent(llm), use_async)

    assert llm.calls == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_react_follow_up_after_tool_result_is_retried(use_async: bool) -> None:
    tool = LookupTool(calls=[])
    action = 'Thought: look\nAction: lookup\nAction Input: {"term": "x"}'
    llm = ScriptedLLM(action, "", "Thought: done\nFinal Answer: value:x")

    result = _kickoff(_agent(llm, [tool]), use_async)

    assert result.raw == "value:x"
    assert tool.calls == ["x"]
    assert llm.calls == 3
    # The retry resends the same conversation, which carries the tool result.
    assert llm.seen_messages[1] == llm.seen_messages[2]
    assert "value:x" in json.dumps(llm.seen_messages[2])


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_native_follow_up_after_tool_result_is_retried(use_async: bool) -> None:
    tool = LookupTool(calls=[])
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"term": "x"})},
        }
    ]
    llm = ScriptedLLM(tool_calls, "", "Done: value:x", native_tools=True)

    result = _kickoff(_agent(llm, [tool]), use_async)

    assert result.raw == "Done: value:x"
    assert tool.calls == ["x"]
    assert llm.calls == 3
    assert llm.seen_messages[1] == llm.seen_messages[2]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_crew_task_recovers_from_one_empty_answer(use_async: bool) -> None:
    llm = ScriptedLLM("", FINAL)
    agent = _agent(llm)
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="Say hello", expected_output="A word", agent=agent)],
    )

    result = asyncio.run(crew.akickoff()) if use_async else crew.kickoff()

    assert result.raw == "hello"
    assert llm.calls == 2


def test_crew_task_raises_typed_error_after_two_empty_answers() -> None:
    llm = ScriptedLLM("", "", FINAL)
    agent = _agent(llm)
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="Say hello", expected_output="A word", agent=agent)],
    )

    with pytest.raises(EmptyLLMResponseError, match=EMPTY_MESSAGE):
        crew.kickoff()

    assert llm.calls == 2


def test_lite_agent_recovers_from_one_empty_answer() -> None:
    llm = ScriptedLLM("", FINAL)
    agent = LiteAgent(role="Tester", goal="Answer", backstory="Tests", llm=llm)

    assert agent.kickoff("Say hello").raw == "hello"
    assert llm.calls == 2


def test_lite_agent_raises_typed_error_after_two_empty_answers() -> None:
    llm = ScriptedLLM("", "", FINAL)
    agent = LiteAgent(role="Tester", goal="Answer", backstory="Tests", llm=llm)

    with pytest.raises(EmptyLLMResponseError, match=EMPTY_MESSAGE):
        agent.kickoff("Say hello")

    assert llm.calls == 2


def test_lite_agent_forced_final_answer_is_admitted_by_the_rpm_limiter() -> None:
    limiter = CountingLimiter()
    llm = ScriptedLLM("", FINAL)
    agent = LiteAgent(
        role="Tester",
        goal="Answer",
        backstory="Tests",
        llm=llm,
        max_iterations=0,
        request_within_rpm_limit=limiter,
    )

    assert agent.kickoff("Say hello").raw == "hello"
    assert llm.calls == 2
    assert limiter.admitted == 2


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_agent_forced_final_answer_counts_against_max_rpm(use_async: bool) -> None:
    action = 'Thought: look\nAction: lookup\nAction Input: {"term": "x"}'
    llm = ScriptedLLM(action, "", FINAL)
    agent = _agent(llm, [LookupTool(calls=[])], max_rpm=2)
    agent.max_iter = 1

    with patch("crewai.utilities.rpm_controller.time") as clock:
        result = _kickoff(agent, use_async)

    assert result.raw == "hello"
    assert llm.calls == 3
    assert clock.sleep.call_count == 1
