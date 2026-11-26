"""Behavior tests for JevDecisionTool.

The tool runs against the real ``typesafe-sdk`` client over an in-memory HTTP
transport: real SDK questions, responses and errors, no network and no API key.
"""

import json
from typing import Any

from crewai.tools import ToolFailure
from crewai_tools.tools.jev_decision_tool.jev_decision_tool import (
    JevDecisionTool,
    JevDecisionToolSchema,
)
import pytest


typesafe_sdk = pytest.importorskip("typesafe_sdk", reason="crewai-tools[typesafe-sdk]")
httpx2 = pytest.importorskip("httpx2")

QUESTION = "  Which team should handle\n this ticket? "
CHOICES = {"Billing & Refunds": " Payments and refunds\n", " tech ": None}
ARGS = {"state": "I was charged twice.", "question": QUESTION, "choices": CHOICES}
ANSWER = {
    "type": "choice",
    "choice": "Billing & Refunds",
    "confidence": 0.0413,
    "probabilities": {"Billing & Refunds": 0.5312, " tech ": 0.4654},
}
USAGE = {"input_tokens": 42, "output_tokens": None}


def api_response(answers: dict[str, Any], *, request_id: str | None = "req_1") -> Any:
    """Build the HTTP response the TypeSafe API would send."""
    return httpx2.Response(
        200,
        headers={"x-typesafe-request-id": request_id} if request_id else {},
        json={"model": "jev-2026-09", "usage": USAGE, "answers": answers},
    )


class FakeApi:
    """Stands in for the TypeSafe API and records the request bodies it gets."""

    def __init__(self, reply: Any = None) -> None:
        self.reply = api_response({"decision": ANSWER}) if reply is None else reply
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(json.loads(request.content))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def tool_for(api: FakeApi, **kwargs: Any) -> JevDecisionTool:
    """Build a tool around a real SDK client wired to a FakeApi."""
    client = typesafe_sdk.TypeSafeClient(
        api_key="test-key",
        model="jev-client-default",
        transport=httpx2.MockTransport(api),
        retry=typesafe_sdk.RetryPolicy(max_retries=0),
        compression=False,
    )
    return JevDecisionTool(client=client, **kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        {"question": ""},
        {"question": " \n"},
        {"state": JevDecisionToolSchema(**ARGS)},
        {"choices": {"low": 1}},
    ],
    ids=["empty question", "blank question", "model as state", "bad description"],
)
def test_run_rejects_invalid_input_without_calling_typesafe(
    invalid: dict[str, Any],
) -> None:
    api = FakeApi()

    with pytest.raises(ValueError, match="Jev Decision Tool"):
        tool_for(api).run(**{**ARGS, **invalid})

    assert api.requests == []


def test_no_labels_is_rejected_by_the_sdk_before_any_request() -> None:
    api = FakeApi()

    result = tool_for(api).run(**{**ARGS, "choices": {}})

    assert isinstance(result, ToolFailure)
    assert result.code == "TypeSafeError"
    assert api.requests == []


def test_returns_the_choice_answer_and_request_metadata() -> None:
    tool = tool_for(FakeApi())

    result = tool.run(**ARGS)

    # Every value is the SDK's own: nothing is rounded, renormalised or derived.
    assert result == {
        "choice": "Billing & Refunds",
        "confidence": 0.0413,
        "probabilities": {"Billing & Refunds": 0.5312, " tech ": 0.4654},
        "model": "jev-2026-09",
        "usage": USAGE,
        "request_id": "req_1",
    }
    assert json.loads(json.dumps(result)) == result
    assert json.loads(tool.format_output_for_agent(result)) == result  # JSON for agents


def test_reports_the_answer_as_given_when_there_is_no_request_id() -> None:
    # One supplied label has no probability, and one probability has no supplied label.
    probabilities = {" tech ": 0.9966, "not a supplied label": 0.0034}
    answer = {**ANSWER, "choice": " tech ", "probabilities": probabilities}
    tool = tool_for(FakeApi(api_response({"decision": answer}, request_id=None)))

    result = tool.run(**ARGS)

    assert (result["choice"], result["probabilities"]) == (" tech ", probabilities)
    assert result["request_id"] is None
    assert json.loads(tool.format_output_for_agent(result)) == result


@pytest.mark.parametrize(
    "state",
    [
        "  I was charged twice.\n\n" * 250,
        {"ticket": {"id": 7, "tags": ["vip"]}, "account": None},
        ["a", {"b": [1, None]}],
    ],
    ids=["long padded text", "object", "array"],
)
def test_sends_one_choice_question_with_state_and_labels_unchanged(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An injected client keeps its own default model, whatever the environment says.
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-env")
    api = FakeApi()

    tool_for(api).run(**{**ARGS, "state": state})

    question = {"type": "choice", "instructions": QUESTION, "criteria": CHOICES}
    assert api.requests == [
        {
            "model": "jev-client-default",  # no tool model: the client's own applies
            "state": state,
            "questions": {"decision": question},
        }
    ]
    assert json.dumps(api.requests[0]["state"]) == json.dumps(state)  # key order too
    assert list(api.requests[0]["questions"]["decision"]["criteria"]) == list(CHOICES)


def test_injected_client_gets_the_configured_model_and_stays_open() -> None:
    api = FakeApi()
    tool = tool_for(api, model="jev-preview")

    result = tool.run(**ARGS)
    # A closed client could not send a second request; one label is still asked.
    tool.run(**{**ARGS, "choices": {"Billing & Refunds": None}})

    assert result["model"] == "jev-2026-09"  # the model that answered
    assert [request["model"] for request in api.requests] == ["jev-preview"] * 2


@pytest.mark.parametrize(
    ("reply", "code", "detail"),
    [
        (httpx2.Response(401, json={}), "TypeSafeAuthenticationError", "401"),
        (httpx2.Response(500, json={}), "TypeSafeInternalServerError", "500"),
        (httpx2.ConnectError("no route"), "TypeSafeAPIConnectionError", "no route"),
        (
            api_response({"decision": {"type": "choice", "choice": " tech "}}),
            "TypeSafeAPIResponseValidationError",
            "confidence",
        ),
        (RuntimeError("not wrapped"), "RuntimeError", "not wrapped"),
    ],
    ids=["unauthorized", "server error", "connection", "malformed answer", "unwrapped"],
)
def test_errors_from_the_sdk_call_become_tool_failures(
    reply: Any, code: str, detail: str
) -> None:
    api = FakeApi(reply)

    result = tool_for(api).run(**ARGS)

    assert isinstance(result, ToolFailure)
    assert result.code == code
    assert result.message.startswith("Jev decision failed: ")
    assert detail in result.message
    assert len(api.requests) == 1  # the tool does not try again


@pytest.mark.parametrize(
    "answers",
    [{}, {"other": ANSWER}, {"decision": {"type": "noul", "noul": 0.9}}],
    ids=["no answers", "answer under another name", "answer of another type"],
)
def test_missing_choice_answer_is_a_tool_failure(answers: dict[str, Any]) -> None:
    api = FakeApi(api_response(answers))

    result = tool_for(api).run(**ARGS)

    assert isinstance(result, ToolFailure)
    assert result.code == "missing_choice_answer"
    assert len(api.requests) == 1  # the tool does not ask again


@pytest.mark.parametrize("label", ["BILLING & REFUNDS", "tech"], ids=["case", "padded"])
def test_label_outside_the_supplied_choices_is_a_tool_failure(label: str) -> None:
    answer = {**ANSWER, "choice": label, "probabilities": {label: 1.0}}

    result = tool_for(FakeApi(api_response({"decision": answer}))).run(**ARGS)

    assert isinstance(result, ToolFailure)
    assert result.code == "unknown_choice_label"


def test_interrupts_are_not_turned_into_failures() -> None:
    with pytest.raises(KeyboardInterrupt):
        tool_for(FakeApi(KeyboardInterrupt())).run(**ARGS)


def test_missing_api_key_is_a_tool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    result = JevDecisionTool().run(**ARGS)

    assert isinstance(result, ToolFailure)
    assert "TYPESAFE_API_KEY" in result.message


def test_creates_and_closes_its_own_client_from_sdk_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi()
    closed: list[bool] = []

    class RecordingClient(typesafe_sdk.TypeSafeClient):
        def __init__(self) -> None:
            super().__init__(transport=httpx2.MockTransport(api), compression=False)

        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-env")
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", RecordingClient)

    result = JevDecisionTool().run(**ARGS)
    JevDecisionTool(model="jev-preview").run(**ARGS)
    api.reply = httpx2.Response(401, json={"detail": "no"})
    failure = JevDecisionTool().run(**ARGS)

    assert result["choice"] == "Billing & Refunds"
    assert [r["model"] for r in api.requests[:2]] == ["jev-from-env", "jev-preview"]
    assert isinstance(failure, ToolFailure)
    assert closed == [True, True, True]  # closed after a failed call too
