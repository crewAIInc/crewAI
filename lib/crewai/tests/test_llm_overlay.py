"""`llm_overlay` swaps an agent's model by role, for the calling context only.

The overlay is read in exactly two places, the validators where `Agent` and
`LiteAgent` resolve their `llm`, so these tests build agents and look at the
model they end up with. No LLM is ever called.

`create_llm("openai/gpt-4o")` returns the native OpenAI provider, which strips
the `openai/` prefix, so the resolved model reads `"gpt-4o"`.
"""

from __future__ import annotations

import contextvars
import threading

from crewai import Agent
from crewai.lite_agent import LiteAgent
from crewai.llm_overlay import active, llm_overlay, overlay_model_for
import pytest


OVERLAY = {"Researcher": "openai/gpt-4o"}


def _agent(role: str) -> Agent:
    return Agent(role=role, goal="g", backstory="b", llm="openai/gpt-4o-mini")


def test_matching_role_gets_the_overlay_model_others_keep_their_own() -> None:
    with llm_overlay(OVERLAY):
        researcher = _agent("Researcher")
        writer = _agent("Writer")

    assert researcher.llm.model == "gpt-4o"
    assert writer.llm.model == "gpt-4o-mini"


def test_overlay_does_not_leak_past_the_block() -> None:
    with llm_overlay(OVERLAY):
        assert overlay_model_for("Researcher") == "openai/gpt-4o"

    assert active.get() is None
    assert overlay_model_for("Researcher") is None
    assert _agent("Researcher").llm.model == "gpt-4o-mini"


def test_overlay_is_reset_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), llm_overlay(OVERLAY):
        raise RuntimeError("boom")

    assert active.get() is None


def test_nested_overlay_restores_the_outer_one() -> None:
    with llm_overlay(OVERLAY):
        with llm_overlay(None):
            assert overlay_model_for("Researcher") is None
        assert overlay_model_for("Researcher") == "openai/gpt-4o"


@pytest.mark.filterwarnings("ignore:LiteAgent is deprecated")
def test_lite_agent_gets_the_overlay_model() -> None:
    with llm_overlay(OVERLAY):
        agent = LiteAgent(
            role="Researcher", goal="g", backstory="b", llm="openai/gpt-4o-mini"
        )

    assert agent.llm.model == "gpt-4o"


def test_overlay_does_not_cross_plain_threads_unless_context_is_copied() -> None:
    """Pins contextvar semantics; callers threading agents must copy the context."""
    seen: dict[str, dict[str, str] | None] = {}

    def record(key: str) -> None:
        seen[key] = active.get()

    with llm_overlay(OVERLAY):
        plain = threading.Thread(target=record, args=("plain",))
        plain.start()
        plain.join()

        ctx = contextvars.copy_context()
        copied = threading.Thread(target=ctx.run, args=(record, "copied"))
        copied.start()
        copied.join()

    assert seen["plain"] is None
    assert seen["copied"] == OVERLAY
