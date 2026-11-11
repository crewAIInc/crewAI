"""Regression: the LiteLLM sync streaming path wrapped every provider failure in
a bare ``Exception("Failed to get streaming response: ...")`` and emitted
``LLMCallFailedEvent`` twice (once in the handler, once in ``call()``).

The run app streams every JSON crew, so auth, rate-limit and bad-request errors
reached users and telemetry as ``Exception``. These tests pin that the
provider's own exception reaches the caller on every LiteLLM path, with one
failed event per provider attempt, and that the error-routing callers still behave.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallFailedEvent
from crewai.events.types.task_events import TaskFailedEvent
from crewai.llm import LLM
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


litellm = pytest.importorskip("litellm")

MODEL = "groq/llama-3.1-8b-instant"

# (error, message, provider attempts): rate limits are retried by
# run_with_rate_limit_retry, and every attempt emits its own failed event.
PROVIDER_ERRORS = [
    pytest.param(
        litellm.AuthenticationError(
            message="Invalid API Key", llm_provider="groq", model=MODEL
        ),
        "Invalid API Key",
        1,
        id="auth",
    ),
    pytest.param(
        litellm.RateLimitError(
            message="Rate limit reached for model", llm_provider="groq", model=MODEL
        ),
        "Rate limit reached for model",
        3,
        id="rate-limit",
    ),
    pytest.param(
        litellm.BadRequestError(
            message="model_decommissioned", model=MODEL, llm_provider="groq"
        ),
        "model_decommissioned",
        1,
        id="bad-request",
    ),
]


@pytest.fixture
def mock_emit() -> Iterator[MagicMock]:
    with patch.object(crewai_event_bus, "emit") as mock:
        yield mock


@pytest.fixture(autouse=True)
def _no_rate_limit_backoff() -> Iterator[None]:
    with patch("crewai.llms.retry.get_retry_delay_seconds", return_value=0.0):
        yield


def _failed_events(mock_emit: MagicMock) -> list[LLMCallFailedEvent]:
    return [
        call.kwargs["event"]
        for call in mock_emit.call_args_list
        if isinstance(call.kwargs.get("event"), LLMCallFailedEvent)
    ]


def _chunk(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-partial",
        "choices": [{"index": 0, "delta": {"content": content}}],
    }


def _stream_then_raise(error: Exception) -> Iterator[dict[str, Any]]:
    yield _chunk("partial answer")
    raise error


async def _astream_raising(error: Exception) -> AsyncIterator[dict[str, Any]]:
    raise error
    yield  # pragma: no cover - makes this an async generator


@pytest.mark.parametrize(("error", "message", "attempts"), PROVIDER_ERRORS)
def test_sync_streaming_raises_provider_error(
    mock_emit: MagicMock, error: Exception, message: str, attempts: int
) -> None:
    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    with patch("crewai.llm.litellm.completion", side_effect=error) as completion:
        with pytest.raises(type(error), match=message) as exc_info:
            llm.call("Hello")

    assert exc_info.value is error
    assert completion.call_count == attempts
    assert len(_failed_events(mock_emit)) == attempts


@pytest.mark.parametrize(("error", "message", "attempts"), PROVIDER_ERRORS)
def test_sync_streaming_raises_provider_error_raised_mid_iteration(
    mock_emit: MagicMock, error: Exception, message: str, attempts: int
) -> None:
    def _empty_then_raise() -> Iterator[dict[str, Any]]:
        raise error
        yield  # pragma: no cover - makes this a generator

    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    with patch(
        "crewai.llm.litellm.completion", side_effect=lambda **_: _empty_then_raise()
    ) as completion:
        with pytest.raises(type(error), match=message) as exc_info:
            llm.call("Hello")

    assert exc_info.value is error
    assert completion.call_count == attempts
    assert len(_failed_events(mock_emit)) == attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "message", "attempts"), PROVIDER_ERRORS)
async def test_async_streaming_raises_provider_error(
    mock_emit: MagicMock, error: Exception, message: str, attempts: int
) -> None:
    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    async def _acompletion(**_: Any) -> AsyncIterator[dict[str, Any]]:
        return _astream_raising(error)

    with patch(
        "crewai.llm.litellm.acompletion", side_effect=_acompletion
    ) as completion:
        with pytest.raises(type(error), match=message) as exc_info:
            await llm.acall("Hello")

    assert exc_info.value is error
    assert completion.call_count == attempts
    assert len(_failed_events(mock_emit)) == attempts


@pytest.mark.parametrize(("error", "message", "attempts"), PROVIDER_ERRORS)
def test_sync_non_streaming_raises_provider_error(
    mock_emit: MagicMock, error: Exception, message: str, attempts: int
) -> None:
    llm = LLM(model=MODEL, is_litellm=True, stream=False)

    with patch("crewai.llm.litellm.completion", side_effect=error) as completion:
        with pytest.raises(type(error), match=message) as exc_info:
            llm.call("Hello")

    assert exc_info.value is error
    assert completion.call_count == attempts
    assert len(_failed_events(mock_emit)) == attempts


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "message", "attempts"), PROVIDER_ERRORS)
async def test_async_non_streaming_raises_provider_error(
    mock_emit: MagicMock, error: Exception, message: str, attempts: int
) -> None:
    llm = LLM(model=MODEL, is_litellm=True, stream=False)

    with patch(
        "crewai.llm.litellm.acompletion", new=AsyncMock(side_effect=error)
    ) as completion:
        with pytest.raises(type(error), match=message) as exc_info:
            await llm.acall("Hello")

    assert exc_info.value is error
    assert completion.call_count == attempts
    assert len(_failed_events(mock_emit)) == attempts


def _context_window_error() -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="This model's maximum context length is 8192 tokens",
        model=MODEL,
        llm_provider="groq",
    )


def test_sync_streaming_context_window_error_becomes_context_length_error() -> None:
    error = _context_window_error()
    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    with patch("crewai.llm.litellm.completion", side_effect=error):
        with pytest.raises(
            LLMContextLengthExceededError, match="maximum context length is 8192"
        ) as exc_info:
            llm.call("Hello")

    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_async_streaming_context_window_error_becomes_context_length_error() -> (
    None
):
    error = _context_window_error()
    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    async def _acompletion(**_: Any) -> AsyncIterator[dict[str, Any]]:
        return _astream_raising(error)

    with patch("crewai.llm.litellm.acompletion", side_effect=_acompletion):
        with pytest.raises(
            LLMContextLengthExceededError, match="maximum context length is 8192"
        ):
            await llm.acall("Hello")


def test_sync_streaming_returns_partial_response_on_mid_stream_error(
    mock_emit: MagicMock,
) -> None:
    error = litellm.APIConnectionError(
        message="connection reset", llm_provider="groq", model=MODEL
    )
    llm = LLM(model=MODEL, is_litellm=True, stream=True)

    with patch(
        "crewai.llm.litellm.completion",
        side_effect=lambda **_: _stream_then_raise(error),
    ):
        result = llm.call("Hello")

    assert result == "partial answer"
    assert _failed_events(mock_emit) == []


def test_sync_streaming_unsupported_stop_retries_without_stop(
    mock_emit: MagicMock,
) -> None:
    error = litellm.BadRequestError(
        message="Unsupported parameter: 'stop' is not supported with this model.",
        model=MODEL,
        llm_provider="groq",
    )
    llm = LLM(model=MODEL, is_litellm=True, stream=True, stop=["Observation:"])
    attempts: list[dict[str, Any]] = []

    def _completion(**params: Any) -> Iterator[dict[str, Any]]:
        attempts.append(params)
        if len(attempts) == 1:
            raise error
        return iter([_chunk("hi")])

    with patch("crewai.llm.litellm.completion", side_effect=_completion):
        result = llm.call("Hello")

    assert result == "hi"
    assert attempts[1]["additional_drop_params"] == ["stop"]
    assert _failed_events(mock_emit) == []


def test_crew_with_streaming_litellm_agent_fails_task_with_provider_error() -> None:
    error = litellm.AuthenticationError(
        message="Invalid API Key", llm_provider="groq", model=MODEL
    )
    agent = Agent(
        role="Writer",
        goal="Write",
        backstory="Writes things",
        llm=LLM(model=MODEL, is_litellm=True, stream=True),
    )
    task = Task(description="Say hi", expected_output="A greeting", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    failed: list[TaskFailedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(TaskFailedEvent)
        def _on_task_failed(_: Any, event: TaskFailedEvent) -> None:
            failed.append(event)

        with patch("crewai.llm.litellm.completion", side_effect=error) as completion:
            with pytest.raises(litellm.AuthenticationError, match="Invalid API Key"):
                crew.kickoff()
        crewai_event_bus.flush()

    assert completion.call_count == 1
    assert [event.error_type for event in failed] == [litellm.AuthenticationError]
