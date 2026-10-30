"""A failed LLM call must produce exactly one ``llm_call_failed`` event.

The OpenAI and Azure providers used to emit ``llm_call_failed`` inside the
completion handlers *and* again in the blanket ``except`` of ``call()``/
``acall()``. The second event is unpaired: the event bus pops whatever
enclosing scope sits below the call -- a task or agent scope in real usage --
and links the failure to that scope's id, ending the wrong trace span.

The handlers now only raise; the outer ``except`` is the sole emitter. The
fallback case matters too: when ``_call_completions`` recovers a 404 by
retrying on the Responses API, the recovered call must report success without
a spurious failure event first.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from openai import NotFoundError

from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import pop_event_scope, push_event_scope
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.openai import completion as completion_module
from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]


class FailingCompletions:
    """Fail like a 401/500 would, from inside the innermost handler."""

    def create(self, **kwargs):
        raise RuntimeError("boom: bad upstream")

    @property
    def with_raw_response(self):
        return self


class AsyncFailingCompletions:
    async def create(self, **kwargs):
        raise RuntimeError("boom: bad upstream")

    @property
    def with_raw_response(self):
        return self


class Recorder:
    """Collect LLM call events around one invocation."""

    def __init__(self):
        self.started: list[LLMCallStartedEvent] = []
        self.failed: list[LLMCallFailedEvent] = []
        self.completed: list[LLMCallCompletedEvent] = []

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_started(source, event: LLMCallStartedEvent) -> None:
            self.started.append(event)

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_failed(source, event: LLMCallFailedEvent) -> None:
            self.failed.append(event)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_completed(source, event: LLMCallCompletedEvent) -> None:
            self.completed.append(event)

        self._handlers = {
            LLMCallStartedEvent: on_started,
            LLMCallFailedEvent: on_failed,
            LLMCallCompletedEvent: on_completed,
        }
        for event_type, handler in self._handlers.items():
            crewai_event_bus.register_handler(event_type, handler)

    def close(self) -> None:
        crewai_event_bus.flush()
        for event_type, handler in self._handlers.items():
            crewai_event_bus.off(event_type, handler)

    def assert_single_failure(self) -> None:
        assert len(self.started) == 1
        assert len(self.failed) == 1, "one failure must be reported exactly once"
        assert (
            self.failed[0].started_event_id == self.started[0].event_id
        ), "the failure must pair with its own started event, not pop the "
        "enclosing scope"


@contextmanager
def enclosing_task_scope():
    """Run inside a task scope that must survive the LLM call.

    The duplicate failed event used to pop this scope off the event-bus
    stack, so its survival is the regression signal.
    """
    push_event_scope("task-started-marker", "task_started")
    try:
        yield
    finally:
        intact = pop_event_scope() == ("task-started-marker", "task_started")
        while pop_event_scope() is not None:  # drain anything a leak left
            pass
        assert intact, "the enclosing task scope was consumed by the LLM call"


@pytest.fixture(autouse=True)
def _clear_learned_models():
    """Keep the process-wide learned set from leaking between tests."""
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()
    yield
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()


def test_openai_sync_failure_emits_one_failed_event(monkeypatch):
    llm = OpenAICompletion(model="gpt-4o-mini", api_key="sk-test")
    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions())),
    )
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            with pytest.raises(RuntimeError, match="boom"):
                llm.call(MESSAGES)
    finally:
        recorder.close()

    recorder.assert_single_failure()


@pytest.mark.asyncio
async def test_openai_async_failure_emits_one_failed_event(monkeypatch):
    llm = OpenAICompletion(model="gpt-4o-mini", api_key="sk-test")
    monkeypatch.setattr(
        llm,
        "_get_async_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=AsyncFailingCompletions())),
    )
    recorder = Recorder()
    try:
        push_event_scope("task-started-marker", "task_started")
        try:
            with pytest.raises(RuntimeError, match="boom"):
                await llm.acall(MESSAGES)
        finally:
            crewai_event_bus.flush()
            intact = pop_event_scope() == ("task-started-marker", "task_started")
            while pop_event_scope() is not None:
                pass
            assert intact, "the enclosing task scope was consumed by the LLM call"
    finally:
        recorder.close()

    recorder.assert_single_failure()


def test_azure_sync_failure_emits_one_failed_event(monkeypatch):
    llm = AzureCompletion(
        model="gpt-4o", endpoint="https://demo.openai.azure.com/openai/deployments/d"
    )
    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(complete=FailingCompletions().create),
    )
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            with pytest.raises(RuntimeError, match="boom"):
                llm.call(MESSAGES)
    finally:
        recorder.close()

    recorder.assert_single_failure()


def make_responses_only_404() -> NotFoundError:
    message = (
        "This model is only supported in v1/responses and not in "
        "/v1/chat/completions."
    )
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": "model",
            "code": None,
        }
    }
    return NotFoundError(
        message,
        response=httpx.Response(
            404,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body=body,
    )


def test_openai_fallback_success_emits_no_failed_event(monkeypatch):
    """A 404 recovered on the Responses API is a success, not a failure.

    The chat-completions attempt fails for real, so the old inner emit fired
    before the fallback ran -- a failure event for a call that then succeeds,
    leaving the completed event to pop the enclosing scope.
    """
    llm = OpenAICompletion(model="gpt-5.2-pro", api_key="sk-test")

    class NotFoundCompletions:
        def create(self, **kwargs):
            raise make_responses_only_404()

        @property
        def with_raw_response(self):
            return self

    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=NotFoundCompletions())),
    )

    def fake_handle_responses(params, available_functions=None, **kwargs):
        llm._emit_call_completed_event(
            response="ok",
            call_type=LLMCallType.LLM_CALL,
            messages=params.get("input", []),
        )
        return "ok"

    monkeypatch.setattr(llm, "_handle_responses", fake_handle_responses)

    recorder = Recorder()
    try:
        with enclosing_task_scope():
            assert llm.call(MESSAGES) == "ok"
    finally:
        recorder.close()

    assert len(recorder.started) == 1
    assert recorder.failed == [], "a recovered call must not report a failure"
    assert len(recorder.completed) == 1
    assert (
        recorder.completed[0].started_event_id == recorder.started[0].event_id
    ), "the success must pair with its own started event"
