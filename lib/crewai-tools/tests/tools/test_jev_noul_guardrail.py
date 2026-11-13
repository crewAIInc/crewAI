from collections.abc import Mapping
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from crewai import Task
from crewai.lite_agent_output import LiteAgentOutput
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import process_guardrail
from crewai_tools.tools.jev_noul_guardrail import (
    DEFAULT_QUESTION_NAME,
    create_jev_noul_guardrail,
)
import pytest


QUESTION = "Does the answer cite at least one source?"


def _noul_response(probability: Any, name: str = DEFAULT_QUESTION_NAME) -> Any:
    return SimpleNamespace(
        answers={name: SimpleNamespace(type="noul", noul=probability)}
    )


class FakeClient:
    """Records system_one calls and replays scripted responses or errors."""

    def __init__(self, *results: Any) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> Any:
        self.calls.append({"state": state, "questions": questions, "model": model})
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _task_output(raw: str = "Paris is the capital of France [1].") -> TaskOutput:
    return TaskOutput(description="Answer the question", raw=raw, agent="Researcher")


def test_output_at_or_above_threshold_passes_unchanged() -> None:
    client = FakeClient(_noul_response(0.91))
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.8, client=client, model="jev-test"
    )
    output = _task_output()

    assert guardrail(output) == (True, output.raw)
    assert client.calls == [
        {
            "state": output.raw,
            "questions": {
                DEFAULT_QUESTION_NAME: {"type": "noul", "instructions": QUESTION}
            },
            "model": "jev-test",
        }
    ]


def test_output_below_threshold_is_rejected_with_a_useful_reason() -> None:
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.8, client=FakeClient(_noul_response(0.42))
    )

    success, reason = guardrail(_task_output())

    assert success is False
    assert "0.420" in reason
    assert "0.8" in reason
    assert QUESTION in reason


@pytest.mark.parametrize(
    ("probability", "threshold", "expected"),
    [
        (0.7, 0.7, True),
        (0.6999, 0.7, False),
        (0.0, 0.0, True),
        (1.0, 1.0, True),
        (0.999, 1.0, False),
        (1, 1, True),
    ],
)
def test_threshold_boundary_is_inclusive(
    probability: float, threshold: float, expected: bool
) -> None:
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=threshold, client=FakeClient(_noul_response(probability))
    )

    assert guardrail(_task_output())[0] is expected


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(SimpleNamespace(answers={}), id="missing-answer"),
        pytest.param(SimpleNamespace(), id="no-answers"),
        pytest.param(SimpleNamespace(answers=None), id="answers-none"),
        pytest.param(_noul_response(0.9, name="other"), id="wrong-name"),
        pytest.param(
            SimpleNamespace(
                answers={
                    DEFAULT_QUESTION_NAME: SimpleNamespace(
                        type="choice", choice="yes", noul=0.99
                    )
                }
            ),
            id="not-noul",
        ),
        pytest.param(_noul_response(None), id="none"),
        pytest.param(_noul_response("0.99"), id="string"),
        pytest.param(_noul_response(True), id="bool"),
        pytest.param(_noul_response(float("nan")), id="nan"),
        pytest.param(_noul_response(float("inf")), id="inf"),
        pytest.param(_noul_response(-0.01), id="negative"),
        pytest.param(_noul_response(1.01), id="above-one"),
    ],
)
def test_missing_or_malformed_answers_never_pass(response: Any) -> None:
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.0, client=FakeClient(response)
    )

    success, reason = guardrail(_task_output())

    assert success is False
    assert reason.startswith("Jev Noul guardrail could not verify the output:")


def test_sdk_failure_is_raised_not_turned_into_a_result() -> None:
    class FakeSDKError(Exception):
        pass

    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.5, client=FakeClient(FakeSDKError("401 Unauthorized"))
    )

    with pytest.raises(FakeSDKError, match="401 Unauthorized"):
        guardrail(_task_output())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"threshold": float("inf")}, "threshold"),
        ({"threshold": True}, "threshold"),
        ({"threshold": "0.5"}, "threshold"),
        ({"threshold": None}, "threshold"),
        ({"question": "   "}, "question"),
        ({"question_name": ""}, "question_name"),
    ],
)
def test_invalid_configuration_raises(kwargs: dict[str, Any], message: str) -> None:
    config: dict[str, Any] = {"question": QUESTION, "threshold": 0.5, **kwargs}

    with pytest.raises(ValueError, match=message):
        create_jev_noul_guardrail(client=FakeClient(), **config)


def test_custom_question_name_is_used_in_request_and_response() -> None:
    client = FakeClient(_noul_response(0.9, name="cites_source"))
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.5, client=client, question_name="cites_source"
    )

    assert guardrail(_task_output())[0] is True
    assert list(client.calls[0]["questions"]) == ["cites_source"]


def test_lite_agent_output_is_supported() -> None:
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.5, client=FakeClient(_noul_response(0.9))
    )
    output = LiteAgentOutput(raw="An answer [1].", agent_role="Researcher")

    assert guardrail(output) == (True, "An answer [1].")


def test_default_client_is_created_per_check_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Any] = []

    class FakeTypeSafeClient(FakeClient):
        def __init__(self) -> None:
            super().__init__(_noul_response(0.9))
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(TypeSafeClient=FakeTypeSafeClient),
    )
    guardrail = create_jev_noul_guardrail(QUESTION, threshold=0.5)

    assert guardrail(_task_output())[0] is True
    assert guardrail(_task_output())[0] is True
    assert len(created) == 2
    assert all(client.closed for client in created)


def test_default_client_is_closed_when_the_sdk_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Any] = []

    class FailingTypeSafeClient(FakeClient):
        def __init__(self) -> None:
            super().__init__(ConnectionError("network down"))
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "typesafe_sdk",
        SimpleNamespace(TypeSafeClient=FailingTypeSafeClient),
    )
    guardrail = create_jev_noul_guardrail(QUESTION, threshold=0.5)

    with pytest.raises(ConnectionError):
        guardrail(_task_output())
    assert created[0].closed is True


def test_missing_sdk_without_client_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    guardrail = create_jev_noul_guardrail(QUESTION, threshold=0.5)

    with pytest.raises(ImportError, match="typesafe-sdk"):
        guardrail(_task_output())


def test_task_accepts_the_guardrail_and_events_can_describe_it() -> None:
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.5, client=FakeClient(_noul_response(0.2))
    )
    task = Task(
        description="Answer with a cited source",
        expected_output="A cited answer",
        guardrail=guardrail,
    )

    result = process_guardrail(_task_output(), task.guardrail, retry_count=0)

    assert result.success is False
    assert result.error is not None
    assert "0.200" in result.error


def test_rejection_uses_the_task_guardrail_retries() -> None:
    client = FakeClient(_noul_response(0.1), _noul_response(0.95))
    guardrail = create_jev_noul_guardrail(QUESTION, threshold=0.8, client=client)
    task = Task(
        description="Answer with a cited source",
        expected_output="A cited answer",
        guardrail=guardrail,
        guardrail_max_retries=1,
    )
    agent = MagicMock(role="Researcher", verbose=False, last_messages=[])
    agent.execute_task.return_value = "Paris [1]."

    final = task._invoke_guardrail_function(
        _task_output("Paris."), agent=agent, tools=[], guardrail=guardrail
    )

    assert final.raw == "Paris [1]."
    assert [call["state"] for call in client.calls] == ["Paris.", "Paris [1]."]
    assert agent.execute_task.call_count == 1
    assert "0.100" in agent.execute_task.call_args.kwargs["context"]


def test_task_fails_after_retries_are_exhausted() -> None:
    client = FakeClient(_noul_response(0.1), _noul_response(0.2))
    guardrail = create_jev_noul_guardrail(QUESTION, threshold=0.8, client=client)
    task = Task(
        description="Answer with a cited source",
        expected_output="A cited answer",
        guardrail=guardrail,
        guardrail_max_retries=1,
    )
    agent = MagicMock(role="Researcher", verbose=False, last_messages=[])
    agent.execute_task.return_value = "Still no source."

    with pytest.raises(Exception, match="validation after 1 retries"):
        task._invoke_guardrail_function(
            _task_output("Paris."), agent=agent, tools=[], guardrail=guardrail
        )
    assert len(client.calls) == 2


@pytest.mark.skipif(
    importlib.util.find_spec("typesafe_sdk") is None,
    reason="typesafe-sdk is not installed",
)
@pytest.mark.parametrize(("probability", "expected"), [(0.9, True), (0.3, False)])
def test_reads_real_sdk_response_types(probability: float, expected: bool) -> None:
    from typesafe_sdk import SystemOneResponse

    response = SystemOneResponse.model_validate(
        {
            "model": "jev",
            "usage": {"input_tokens": 10, "output_tokens": 1},
            "answers": {
                DEFAULT_QUESTION_NAME: {"type": "noul", "noul": probability}
            },
        }
    )
    guardrail = create_jev_noul_guardrail(
        QUESTION, threshold=0.5, client=FakeClient(response)
    )

    assert guardrail(_task_output())[0] is expected
