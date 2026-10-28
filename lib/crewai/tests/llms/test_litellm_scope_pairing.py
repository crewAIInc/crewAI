"""Every exit path of the legacy (LiteLLM) ``LLM.call()/acall()`` must pair.

``llm_call_started`` pushes a scope that only a terminal ``llm_call_failed``/
``llm_call_completed`` pops. The call methods used to leak it on several
paths, and one path double-reported instead:

- ``except LLMContextLengthExceededError: raise`` re-raised with no terminal
  event, so every context-length failure (the exception AgentExecutor's
  ``respect_context_window`` recovery depends on) left the scope on the stack.
- The unsupported-'stop' retry recursed through ``self.call()``/``self.acall()``,
  pushing a second started scope; the first leaked even when the retry
  succeeded.
- ``_validate_call_params()`` raised after the started event was emitted.
- ``acall()`` ran ``_aprocess_message_files`` outside any failure handling.
- ``_handle_streaming_response`` emitted its own ``llm_call_failed`` and then
  re-raised into ``call()``'s blanket except, which emitted a second one that
  popped the caller's scope.

The fix validates before emitting started, terminates the scope on the
context-length re-raise, retries 'stop' in place (one started per call), and
leaves the failed event to the blanket except alone.
"""

from contextlib import contextmanager

import litellm
import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_context import pop_event_scope, push_event_scope
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.llm import LLM
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


def make_llm(**kwargs) -> LLM:
    kwargs.setdefault("stream", False)
    return LLM(model="groq/llama-3.1-8b-instant", api_key="dummy", **kwargs)


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


@contextmanager
def enclosing_task_scope():
    """Run inside a task scope that must survive the LLM call.

    A leaked or duplicated terminal event leaves the llm scope on the stack
    or pops this scope, so its survival is the regression signal.
    """
    push_event_scope("task-started-marker", "task_started")
    try:
        yield
    finally:
        top = pop_event_scope()
        intact = top == ("task-started-marker", "task_started")
        while pop_event_scope() is not None:  # drain anything a leak left
            pass
        assert intact, f"the enclosing task scope was consumed; top was {top}"


def test_context_length_failure_pairs_with_started_event(monkeypatch):
    """The re-raise for AgentExecutor still owes the scope a terminal event."""
    llm = make_llm()

    def over_limit(**kwargs):
        raise LLMContextLengthExceededError("input too long")

    monkeypatch.setattr(llm, "_handle_non_streaming_response", over_limit)
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            with pytest.raises(LLMContextLengthExceededError):
                llm.call("hi")
    finally:
        recorder.close()

    assert len(recorder.started) == 1
    assert len(recorder.failed) == 1
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id


@pytest.mark.asyncio
async def test_async_context_length_failure_pairs_with_started_event(monkeypatch):
    llm = make_llm()

    async def over_limit(**kwargs):
        raise LLMContextLengthExceededError("input too long")

    monkeypatch.setattr(llm, "_ahandle_non_streaming_response", over_limit)
    recorder = Recorder()
    try:
        push_event_scope("task-started-marker", "task_started")
        try:
            with pytest.raises(LLMContextLengthExceededError):
                await llm.acall("hi")
        finally:
            top = pop_event_scope()
            while pop_event_scope() is not None:
                pass
            assert top == (
                "task-started-marker",
                "task_started",
            ), f"the enclosing task scope was consumed; top was {top}"
    finally:
        recorder.close()

    assert len(recorder.started) == 1
    assert len(recorder.failed) == 1
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id


def test_stop_retry_keeps_single_started_scope(monkeypatch):
    """The recovered retry must not push a second started scope.

    A call whose first attempt is rejected for the 'stop' parameter and whose
    retry succeeds is a success: one started, one completed, no failure event.
    """
    llm = make_llm()
    attempts: list[dict] = []

    def flaky(params, **kwargs):
        attempts.append(params)
        if len(attempts) == 1:
            raise Exception(
                "'stop' is an Unsupported parameter for this model"
            )
        llm._emit_call_completed_event(
            response="ok",
            call_type=LLMCallType.LLM_CALL,
            from_task=kwargs.get("from_task"),
            from_agent=kwargs.get("from_agent"),
            messages=params.get("messages"),
        )
        return "ok"

    monkeypatch.setattr(llm, "_handle_non_streaming_response", flaky)
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            assert llm.call("hi") == "ok"
    finally:
        recorder.close()

    assert len(attempts) == 2, "one rejected attempt plus one retry"
    assert len(recorder.started) == 1, "the retry must reuse the started scope"
    assert recorder.failed == [], "a recovered call must not report a failure"
    assert len(recorder.completed) == 1
    assert recorder.completed[0].started_event_id == recorder.started[0].event_id


def test_validation_failure_emits_no_events(monkeypatch):
    """A call rejected before it starts pushes no scope at all."""

    def invalid(**kwargs):
        raise ValueError("response format is not supported by this provider")

    monkeypatch.setattr(llm := make_llm(), "_validate_call_params", invalid)
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            with pytest.raises(ValueError, match="not supported"):
                llm.call("hi")
    finally:
        recorder.close()

    assert recorder.started == []
    assert recorder.failed == []
    assert recorder.completed == []


def test_streaming_failure_reports_one_failed_event(monkeypatch):
    """A mid-stream failure reaches call()'s blanket except once.

    _handle_streaming_response used to emit its own llm_call_failed before
    re-raising; the blanket except then emitted a second one, which popped
    the caller's scope.
    """
    llm = make_llm(stream=True)

    def boom(**kwargs):
        raise Exception("boom: stream broke")

    monkeypatch.setattr(litellm, "completion", boom)
    recorder = Recorder()
    try:
        with enclosing_task_scope():
            with pytest.raises(Exception, match="Failed to get streaming response"):
                llm.call("hi")
    finally:
        recorder.close()

    assert len(recorder.started) == 1
    assert len(recorder.failed) == 1, "one failure must be reported exactly once"
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id


@pytest.mark.asyncio
async def test_async_file_processing_failure_pairs_with_started_event(monkeypatch):
    llm = make_llm()

    async def boom(messages):
        raise RuntimeError("file processing failed")

    monkeypatch.setattr(llm, "_aprocess_message_files", boom)
    recorder = Recorder()
    try:
        push_event_scope("task-started-marker", "task_started")
        try:
            with pytest.raises(RuntimeError, match="file processing"):
                await llm.acall("hi")
        finally:
            top = pop_event_scope()
            while pop_event_scope() is not None:
                pass
            assert top == (
                "task-started-marker",
                "task_started",
            ), f"the enclosing task scope was consumed; top was {top}"
    finally:
        recorder.close()

    assert len(recorder.started) == 1
    assert len(recorder.failed) == 1
    assert recorder.failed[0].started_event_id == recorder.started[0].event_id
