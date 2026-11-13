from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.jev_score_tool.jev_score_tool import (
    JevScoreTool,
    JevScoreToolSchema,
)


URGENCY_LEVELS = ["Can wait", "Needs attention this week", "Needs attention today"]


class FakeScore:
    def __init__(self, instructions: str, criteria: list[Any]) -> None:
        self.instructions = instructions
        self.criteria = criteria


@pytest.fixture
def typesafe_stub(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("typesafe_sdk")
    module.Score = FakeScore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return module


def _score_response(**answer: Any) -> SimpleNamespace:
    return SimpleNamespace(
        model="jev-latest",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 12}),
        scores={"score": SimpleNamespace(**answer)},
    )


def _client_returning(response: Any) -> MagicMock:
    client = MagicMock()
    client.system_one.return_value = response
    return client


def _run(tool: JevScoreTool, criteria: list[Any] | None = None) -> dict[str, Any]:
    return tool._run(
        state="The production database is down.",
        question="How urgent is this message?",
        criteria=URGENCY_LEVELS if criteria is None else criteria,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state": None}, "state cannot be None"),
        ({"question": "  "}, "question cannot be empty"),
        ({"criteria": []}, "criteria must contain at least one level"),
        ({"criteria": ["Low", " "]}, "criteria levels cannot be empty"),
        ({"criteria": ["Low", None]}, "criteria levels cannot be empty"),
    ],
)
def test_schema_rejects_invalid_input(kwargs: dict[str, Any], message: str) -> None:
    valid: dict[str, Any] = {
        "state": "ticket",
        "question": "How urgent?",
        "criteria": URGENCY_LEVELS,
    }

    with pytest.raises(ValueError, match=message):
        JevScoreToolSchema(**{**valid, **kwargs})


def test_run_returns_score_distribution_and_confidence(
    typesafe_stub: ModuleType,
) -> None:
    client = _client_returning(
        _score_response(
            score=1.9,
            confidence=0.88,
            legend=dict(enumerate(URGENCY_LEVELS)),
            probabilities={0: 0.02, 1: 0.06, 2: 0.92},
        )
    )

    result = _run(JevScoreTool(client=client))

    assert result == {
        "success": True,
        "score": 1.9,
        "probabilities": {0: 0.02, 1: 0.06, 2: 0.92},
        "confidence": 0.88,
        "legend": dict(enumerate(URGENCY_LEVELS)),
        "model": "jev-latest",
        "usage": {"input_tokens": 12},
    }


def test_run_sends_ordered_criteria_as_sdk_score_question(
    typesafe_stub: ModuleType,
) -> None:
    client = _client_returning(
        _score_response(score=0.0, confidence=1.0, legend={}, probabilities={0: 1.0})
    )

    _run(JevScoreTool(client=client))

    kwargs = client.system_one.call_args.kwargs
    assert kwargs["state"] == "The production database is down."
    question = kwargs["questions"]["score"]
    assert isinstance(question, FakeScore)
    assert question.instructions == "How urgent is this message?"
    assert question.criteria == URGENCY_LEVELS


@pytest.mark.parametrize("score", [0.0, 0.4, 1.7])
def test_score_is_reported_as_is_not_as_pass_fail(
    typesafe_stub: ModuleType, score: float
) -> None:
    client = _client_returning(
        _score_response(
            score=score, confidence=0.5, legend={}, probabilities={0: 0.5, 1: 0.5}
        )
    )

    result = _run(JevScoreTool(client=client))

    assert result["success"] is True
    assert result["score"] == score
    assert type(result["score"]) is float


@pytest.mark.parametrize("scores", [{}, None])
def test_run_reports_missing_score_answer(
    typesafe_stub: ModuleType, scores: dict[str, Any] | None
) -> None:
    client = _client_returning(SimpleNamespace(scores=scores))

    result = _run(JevScoreTool(client=client))

    assert result["success"] is False
    assert "did not include a Score answer" in result["error"]
    assert "score" not in result


def test_run_reports_answer_without_score(typesafe_stub: ModuleType) -> None:
    client = _client_returning(
        _score_response(score=None, confidence=0.5, legend={}, probabilities={})
    )

    result = _run(JevScoreTool(client=client))

    assert result == {
        "success": False,
        "error": "Score answer did not include a score.",
    }


def test_run_reports_invalid_input_without_calling_sdk(
    typesafe_stub: ModuleType,
) -> None:
    client = MagicMock()

    result = _run(JevScoreTool(client=client), criteria=[])

    assert result["success"] is False
    assert "criteria must contain at least one level" in result["error"]
    client.system_one.assert_not_called()


def test_run_reports_sdk_failure(typesafe_stub: ModuleType) -> None:
    client = MagicMock()
    client.system_one.side_effect = RuntimeError("service unavailable")

    result = _run(JevScoreTool(client=client))

    assert result == {"success": False, "error": "service unavailable"}


def test_missing_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    result = _run(JevScoreTool())

    assert result["success"] is False
    assert "uv add 'crewai-tools[typesafe-sdk]'" in result["error"]


def test_real_sdk_score_round_trip() -> None:
    sdk = pytest.importorskip("typesafe_sdk")
    response = sdk.SystemOneResponse.model_validate_json(
        """{
            "model": "jev-latest",
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "answers": {"score": {
                "type": "score", "score": 1.7, "confidence": 0.9,
                "legend": {"0": "Can wait", "1": "Needs attention this week",
                           "2": "Needs attention today"},
                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
            }}
        }"""
    )
    client = _client_returning(response)

    result = _run(JevScoreTool(client=client))

    question = client.system_one.call_args.kwargs["questions"]["score"]
    assert isinstance(question, sdk.Score)
    assert list(question.criteria) == URGENCY_LEVELS
    assert result["success"] is True
    assert result["score"] == 1.7
    assert result["probabilities"] == {0: 0.1, 1: 0.1, 2: 0.8}
    assert result["legend"] == dict(enumerate(URGENCY_LEVELS))


def test_public_import_and_metadata() -> None:
    from crewai_tools import JevScoreTool as PublicJevScoreTool

    tool = PublicJevScoreTool(client=MagicMock())

    assert PublicJevScoreTool is JevScoreTool
    assert tool.name == "Jev Score Tool"
    assert "typesafe-sdk" in tool.package_dependencies
    assert "TYPESAFE_API_KEY" in {env_var.name for env_var in tool.env_vars}
