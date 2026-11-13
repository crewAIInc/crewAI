"""Smoke test for the custom input-provider example in the Flows guide.

Mirrors the ``ScriptedInputProvider`` and ``OnboardingFlow`` shown in
``docs/edge/en/concepts/flows.mdx`` and checks the metadata and unavailable-input
paths. Deterministic: the provider returns scripted answers, so nothing blocks on
stdin and no model or transport call is made.
"""

from __future__ import annotations

from crewai.flow import Flow, start
from crewai.flow.input_provider import InputResponse


class ScriptedInputProvider:
    def __init__(self, answers):
        self.answers = answers

    def request_input(self, message, flow, metadata=None):
        return self.answers.get(message)


class OnboardingFlow(Flow):
    input_provider = ScriptedInputProvider(
        {
            "What's your name?": InputResponse(text="Ada", metadata={"responded_by": "harness"}),
            "Favorite language?": None,
        }
    )

    @start()
    def collect(self):
        name = self.ask("What's your name?", metadata={"field": "name"})
        language = self.ask("Favorite language?", metadata={"field": "language"})
        if language is None:
            language = "(not provided)"
        return f"Hello, {name}! Favourite language: {language}"


def test_ask_returns_text_and_keeps_response_metadata():
    flow = OnboardingFlow()
    result = flow.kickoff()
    assert result == "Hello, Ada! Favourite language: (not provided)"

    history = flow._input_history
    assert history[0]["response"] == "Ada"
    assert history[0]["response_metadata"] == {"responded_by": "harness"}
    assert history[0]["metadata"] == {"field": "name"}


def test_unavailable_input_is_none_not_an_answer():
    flow = OnboardingFlow()
    flow.kickoff()
    second = flow._input_history[1]
    assert second["response"] is None
    assert second["response_metadata"] is None
