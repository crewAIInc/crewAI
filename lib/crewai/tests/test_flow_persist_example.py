"""Smoke test for the conversational-flow persistence example in the Flows guide.

Mirrors the ``ResearchBot`` shown in the conversational Flow guide and checks that a fresh
Flow instance resumes a persisted session across instances, and that a different session id
is a separate conversation. Deterministic and offline: a temporary SQLite database, a scripted
route, and no model calls.
"""

from __future__ import annotations

from typing import Any

from crewai.flow import Flow, listen
from crewai.flow.persistence import SQLiteFlowPersistence, persist


def _roles(messages: Any) -> list[tuple[Any, Any]]:
    rows: list[tuple[Any, Any]] = []
    for message in messages:
        if hasattr(message, "role"):
            rows.append((message.role, message.content))
        else:
            rows.append((message["role"], message["content"]))
    return rows


def test_conversation_resumes_across_fresh_instances(tmp_path: Any) -> None:
    store = SQLiteFlowPersistence(str(tmp_path / "conversation.db"))

    @persist(store)
    class ResearchBot(Flow):
        conversational = True

        def route_turn(self, context: dict[str, Any]) -> str | None:
            return "research"

        @listen("research")
        def run_research(self) -> str:
            return f"researched: {self.state.current_user_message}"

    ResearchBot().handle_turn("what is CrewAI?", session_id="demo")

    resumed = ResearchBot()  # a brand-new instance, same store + session
    resumed.handle_turn("tell me more", session_id="demo")
    assert _roles(resumed.state.messages) == [
        ("user", "what is CrewAI?"),
        ("assistant", "researched: what is CrewAI?"),
        ("user", "tell me more"),
        ("assistant", "researched: tell me more"),
    ]

    other = ResearchBot()  # a different session id is a separate conversation
    other.handle_turn("hello", session_id="other")
    assert _roles(other.state.messages) == [
        ("user", "hello"),
        ("assistant", "researched: hello"),
    ]
