from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from crewai_tools.tools.jev_decision_tool.jev_decision_tool import (
    JevDecisionTool,
    JevDecisionToolSchema,
)
import pytest


class FakeChoice:
    def __init__(self, instructions: str, criteria: dict[str, object | None]) -> None:
        """Capture the instructions and criteria passed to the SDK Choice constructor."""
        self.instructions = instructions
        self.criteria = criteria


def _install_typesafe_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the SDK module with a credential-free Choice implementation."""
    module = ModuleType("typesafe_sdk")
    module.Choice = FakeChoice
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)


def test_schema_rejects_empty_question() -> None:
    """Reject questions containing only whitespace."""
    with pytest.raises(ValueError, match="question cannot be empty"):
        JevDecisionToolSchema(
            state="ticket",
            question=" ",
            choices={"billing": None, "technical": None},
        )


def test_schema_requires_two_choices() -> None:
    """Require at least two candidate labels for a decision."""
    with pytest.raises(ValueError, match="at least two labels"):
        JevDecisionToolSchema(
            state="ticket",
            question="Which team?",
            choices={"billing": None},
        )


def test_run_returns_choice_probabilities_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward the request and preserve the SDK answer and metadata."""
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
    question = client.system_one.call_args.kwargs["questions"]["decision"]
    assert isinstance(question, FakeChoice)
    assert question.instructions == "Which team should handle this?"
    assert question.criteria == {
        "billing": "Payment or subscription problems",
        "technical": "Bugs or integrations",
    }


def test_run_reports_missing_choice_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a missing decision answer as a structured failure."""
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
    """Return SDK exceptions as explicit failure results."""
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
    """Provide a direct SDK installation command when the dependency is absent."""
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    tool = JevDecisionTool()

    result = tool._run(
        state="ticket",
        question="Which team?",
        choices={"billing": None, "technical": None},
    )

    assert result["success"] is False
    assert "typesafe-sdk" in result["error"]
    assert "uv add typesafe-sdk" in result["error"]


def test_public_import_from_crewai_tools() -> None:
    """Expose the tool through the public package import."""
    from crewai_tools import JevDecisionTool as PublicJevDecisionTool

    assert PublicJevDecisionTool is JevDecisionTool


def test_tool_metadata() -> None:
    """Declare the tool name, optional package, and API-key requirement."""
    tool = JevDecisionTool(client=MagicMock())

    assert tool.name == "Jev Decision Tool"
    assert "typesafe-sdk" in tool.package_dependencies
    env_var_names = {env_var.name for env_var in tool.env_vars}
    assert "TYPESAFE_API_KEY" in env_var_names


@pytest.fixture
def decision_input() -> dict[str, Any]:
    """Provide one valid request shared by sync and async behavior checks."""
    return {
        "state": {"ticket": "charged twice"},
        "question": "Which team?",
        "choices": {"billing": "Payments", "technical": "Bugs"},
    }


@pytest.fixture
def decision_response() -> SimpleNamespace:
    """Provide an SDK-shaped answer whose confidence differs from its probability."""
    return SimpleNamespace(
        model="jev-test",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10}),
        choices={
            "decision": SimpleNamespace(
                choice="billing",
                confidence=0.91,
                probabilities={"billing": 0.87, "technical": 0.13},
            )
        },
    )


@pytest.fixture
def async_client(decision_response: SimpleNamespace) -> SimpleNamespace:
    """Provide awaitable request and cleanup methods without network access."""
    return SimpleNamespace(
        system_one=AsyncMock(return_value=decision_response), aclose=AsyncMock()
    )


@pytest.mark.asyncio
async def test_arun_matches_sync_and_awaits_request(
    monkeypatch: pytest.MonkeyPatch,
    async_client: SimpleNamespace,
    decision_response: SimpleNamespace,
    decision_input: dict[str, Any],
) -> None:
    """Await only the async client and return the same result as the sync path."""
    _install_typesafe_stub(monkeypatch)
    sync_client = MagicMock()
    sync_client.system_one.return_value = decision_response
    tool = JevDecisionTool(
        client=sync_client, async_client=async_client, model="jev-test"
    )

    result = await tool.arun(**decision_input)

    sync_client.system_one.assert_not_called()
    assert result == tool.run(**decision_input)
    async_client.system_one.assert_awaited_once()
    request = async_client.system_one.call_args.kwargs
    assert request["state"] == decision_input["state"]
    assert request["model"] == "jev-test"
    choice = request["questions"]["decision"]
    assert isinstance(choice, FakeChoice)
    assert choice.instructions == decision_input["question"]
    assert choice.criteria == decision_input["choices"]
    async_client.aclose.assert_not_awaited()
    sync_client.close.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("failure", [False, True])
async def test_async_client_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    async_client: SimpleNamespace,
    decision_input: dict[str, Any],
    owned: bool,
    failure: bool,
) -> None:
    """Close owned clients on success or failure while leaving injected clients open."""
    _install_typesafe_stub(monkeypatch)
    factory = MagicMock(return_value=async_client)
    monkeypatch.setattr(
        sys.modules["typesafe_sdk"], "AsyncTypeSafeClient", factory, raising=False
    )
    if failure:
        async_client.system_one.side_effect = RuntimeError("service unavailable")
    tool = JevDecisionTool(
        async_client=None if owned else async_client, model="jev-test"
    )

    result = await tool.arun(**decision_input)

    if failure:
        assert result == {"success": False, "error": "service unavailable"}
    else:
        assert result["success"] is True
    if owned:
        factory.assert_called_once_with(model="jev-test")
        async_client.aclose.assert_awaited_once()
    else:
        factory.assert_not_called()
        async_client.aclose.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [True, False])
async def test_cancellation_propagates_and_respects_client_ownership(
    monkeypatch: pytest.MonkeyPatch,
    async_client: SimpleNamespace,
    decision_input: dict[str, Any],
    owned: bool,
) -> None:
    """Propagate task cancellation and clean up only the client owned by the tool."""
    _install_typesafe_stub(monkeypatch)
    monkeypatch.setattr(
        sys.modules["typesafe_sdk"],
        "AsyncTypeSafeClient",
        lambda **kwargs: async_client,
        raising=False,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def pending_request(**kwargs: Any) -> None:
        """Keep the fake request pending until its task is cancelled."""
        started.set()
        await release.wait()

    async_client.system_one.side_effect = pending_request
    tool = JevDecisionTool(async_client=None if owned else async_client)
    task = asyncio.create_task(tool.arun(**decision_input))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    if owned:
        async_client.aclose.assert_awaited_once()
    else:
        async_client.aclose.assert_not_awaited()
