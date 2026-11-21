from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.jev_decision_tool.jev_decision_tool import (
    JevDecisionTool,
    JevDecisionToolSchema,
)


class FakeChoice:
    def __init__(self, instructions: str, criteria: dict[str, object | None]) -> None:
        self.instructions = instructions
        self.criteria = criteria


def _install_typesafe_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("typesafe_sdk")
    module.Choice = FakeChoice
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)


def test_schema_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question cannot be empty"):
        JevDecisionToolSchema(
            state="ticket",
            question=" ",
            choices={"billing": None, "technical": None},
        )


def test_schema_requires_two_choices() -> None:
    with pytest.raises(ValueError, match="at least two labels"):
        JevDecisionToolSchema(
            state="ticket",
            question="Which team?",
            choices={"billing": None},
        )


def test_run_returns_choice_probabilities_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_typesafe_stub(monkeypatch)
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-latest",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10}),
        choices={
            "decision": SimpleNamespace(
                choice="billing",
                confidence=0.91,
                probabilities={"billing": 0.87, "technical": 0.13},
            )
        },
    )

    tool = JevDecisionTool(client=client)
    result = tool._run(
        state="I was charged twice.",
        question="Which team should handle this?",
        choices={
            "billing": "Payment or subscription problems",
            "technical": "Bugs or integrations",
        },
    )

    assert result == {
        "success": True,
        "choice": "billing",
        "probabilities": {"billing": 0.87, "technical": 0.13},
        "confidence": 0.91,
        "model": "jev-latest",
        "usage": {"input_tokens": 10},
    }
    client.system_one.assert_called_once()
    assert client.system_one.call_args.kwargs["state"] == "I was charged twice."
    assert "model" not in client.system_one.call_args.kwargs
    question = client.system_one.call_args.kwargs["questions"]["decision"]
    assert isinstance(question, FakeChoice)
    assert question.instructions == "Which team should handle this?"
    assert question.criteria == {
        "billing": "Payment or subscription problems",
        "technical": "Bugs or integrations",
    }


def test_run_reports_missing_choice_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_typesafe_stub(monkeypatch)
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(choices={})

    result = JevDecisionTool(client=client)._run(
        state="ticket",
        question="Which team?",
        choices={"billing": None, "technical": None},
    )

    assert result["success"] is False
    assert "did not include a Choice answer" in result["error"]


def test_run_reports_sdk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_typesafe_stub(monkeypatch)
    client = MagicMock()
    client.system_one.side_effect = RuntimeError("service unavailable")

    result = JevDecisionTool(client=client)._run(
        state="ticket",
        question="Which team?",
        choices={"billing": None, "technical": None},
    )

    assert result == {"success": False, "error": "service unavailable"}


def test_missing_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    tool = JevDecisionTool()

    result = tool._run(
        state="ticket",
        question="Which team?",
        choices={"billing": None, "technical": None},
    )

    assert result["success"] is False
    assert "typesafe-sdk" in result["error"]
    assert "uv add --optional typesafe-sdk typesafe-sdk" in result["error"]


def test_public_import_from_crewai_tools() -> None:
    from crewai_tools import JevDecisionTool as PublicJevDecisionTool

    assert PublicJevDecisionTool is JevDecisionTool


def test_tool_metadata() -> None:
    tool = JevDecisionTool(client=MagicMock())

    assert tool.name == "Jev Decision Tool"
    assert "typesafe-sdk" in tool.package_dependencies
    env_var_names = {env_var.name for env_var in tool.env_vars}
    assert "TYPESAFE_API_KEY" in env_var_names
