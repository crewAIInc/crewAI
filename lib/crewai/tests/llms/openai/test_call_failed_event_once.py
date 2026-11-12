"""A failed OpenAI call reports exactly one LLMCallFailedEvent.

``call``/``acall`` already report any error that leaves them. When the request
handlers report it too, one failure emits two ending events: the second pops
whatever scope encloses the call (an agent execution, a task) off the event
stack, and a call recovered by the Responses API fallback reports a failure the
caller never sees.
"""

from typing import Any

import httpx
import openai
import pytest

from crewai.events.event_context import _event_id_stack, event_scope
from crewai.llms.providers.openai import completion as completion_module
from crewai.llms.providers.openai.completion import OpenAICompletion


MESSAGES = [{"role": "user", "content": "hi"}]

RESPONSES_ONLY = (
    "This model is only supported in v1/responses and not in /v1/chat/completions."
)

RESPONSE_BODY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-5-pro",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 2,
    },
}


def _error(status: int, message: str, param: str | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


def _bad_request(request: httpx.Request) -> httpx.Response:
    return _error(400, "Invalid 'messages': empty content.")


def _responses_only(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return _error(404, RESPONSES_ONLY, param="model")
    return httpx.Response(200, json=RESPONSE_BODY)


@pytest.fixture(autouse=True)
def _clear_learned_models():
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()
    yield
    completion_module._LEARNED_RESPONSES_ONLY_MODELS.clear()


def _build(
    monkeypatch: pytest.MonkeyPatch, handler: Any, **kwargs: Any
) -> tuple[OpenAICompletion, list[str]]:
    llm = OpenAICompletion(
        model=kwargs.pop("model", "gpt-4o-mini"), api_key="sk-test", **kwargs
    )
    transport = httpx.MockTransport(handler)
    sync_client = openai.OpenAI(
        api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=transport)
    )
    async_client = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )
    monkeypatch.setattr(llm, "_get_sync_client", lambda: sync_client)
    monkeypatch.setattr(llm, "_get_async_client", lambda: async_client)

    failures: list[str] = []
    report_failure = llm._emit_call_failed_event

    def record(**kwargs: Any) -> None:
        failures.append(kwargs["error"])
        report_failure(**kwargs)

    monkeypatch.setattr(llm, "_emit_call_failed_event", record)
    return llm, failures


@pytest.mark.parametrize("api", ["completions", "responses"])
def test_failed_call_reports_one_failure_and_keeps_enclosing_scope(
    monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    llm, failures = _build(monkeypatch, _bad_request, api=api)

    with event_scope("agent-execution", "agent_execution_started"):
        with pytest.raises(openai.BadRequestError):
            llm.call(MESSAGES)
        stack = _event_id_stack.get()

    assert len(failures) == 1
    assert stack and stack[-1][0] == "agent-execution"


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["completions", "responses"])
async def test_failed_async_call_reports_one_failure_and_keeps_enclosing_scope(
    monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    llm, failures = _build(monkeypatch, _bad_request, api=api)

    with event_scope("agent-execution", "agent_execution_started"):
        with pytest.raises(openai.BadRequestError):
            await llm.acall(MESSAGES)
        stack = _event_id_stack.get()

    assert len(failures) == 1
    assert stack and stack[-1][0] == "agent-execution"


def test_call_recovered_on_responses_api_reports_no_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm, failures = _build(monkeypatch, _responses_only, model="gpt-5-pro")

    with event_scope("agent-execution", "agent_execution_started"):
        assert llm.call(MESSAGES) == "done"
        stack = _event_id_stack.get()

    assert failures == []
    assert stack and stack[-1][0] == "agent-execution"
