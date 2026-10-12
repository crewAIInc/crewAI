"""Smoke check for the scripted input provider example (crewAIInc/crewAI#8071).

Mirrors the documented ``scripted_input_provider_example.py``: a
deterministic provider implementing
``request_input(message, flow, metadata=None)`` used by ``Flow.ask()``
inside a running step. Covers the request/response metadata path and the
unavailable-input (``None``) path. No stdin, no model calls, no credentials.
"""

from __future__ import annotations

from typing import Any

from crewai.flow import Flow, start
from crewai.flow.input_provider import InputResponse


class ScriptedProvider:
    """Deterministic provider keyed by question text (mirrors the docs)."""

    def __init__(self, script: list[tuple[str, str, str]]) -> None:
        self.script = script
        self.seen_metadata: list[dict[str, Any] | None] = []

    def request_input(
        self,
        message: str,
        flow: Flow[Any],
        metadata: dict[str, Any] | None = None,
    ) -> InputResponse | None:
        self.seen_metadata.append(metadata)
        for question_part, answer, responder in self.script:
            if question_part in message:
                return InputResponse(
                    text=answer,
                    metadata={"responded_by": responder, "channel": "script"},
                )
        # Unavailable input: None, never an invented answer.
        return None


def _make_flow() -> Flow[Any]:
    class OnboardingFlow(Flow):
        input_provider = ScriptedProvider(
            script=[("team name", "spinach-labs", "abhishek")]
        )

        @start()
        def onboard(self) -> str:
            team = self.ask(
                "What is your team name?",
                metadata={"user_id": "u123", "channel": "#onboarding"},
            )
            if team is None:
                return "onboarding skipped: no team name provided"
            nickname = self.ask("Pick a nickname for the team bot?")
            if nickname is None:
                return f"team '{team}' onboarded without a nickname"
            return f"team '{team}' onboarded, bot nickname '{nickname}'"

    return OnboardingFlow()


class TestScriptedInputProviderExample:
    def test_metadata_and_unavailable_input_paths(self) -> None:
        flow = _make_flow()
        result = flow.kickoff()

        # ask() returns the response text; the None case is handled
        # explicitly, never treated as an answer.
        assert result == "team 'spinach-labs' onboarded without a nickname"

        # Request metadata reached the provider on both calls.
        provider = flow.input_provider
        assert provider.seen_metadata == [
            {"user_id": "u123", "channel": "#onboarding"},
            None,
        ]

        # input_history records both prompts, responses, and metadata.
        history = flow.input_history
        assert [entry["message"] for entry in history] == [
            "What is your team name?",
            "Pick a nickname for the team bot?",
        ]
        assert history[0]["response"] == "spinach-labs"
        assert history[0]["method_name"] == "onboard"
        assert history[0]["metadata"] == {
            "user_id": "u123",
            "channel": "#onboarding",
        }
        assert history[0]["response_metadata"] == {
            "responded_by": "abhishek",
            "channel": "script",
        }
        # Unavailable input: None response, no response metadata.
        assert history[1]["response"] is None
        assert history[1]["response_metadata"] is None
