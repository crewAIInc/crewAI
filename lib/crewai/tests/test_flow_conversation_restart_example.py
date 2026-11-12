"""Smoke check for the SQLite persistence + restart example (crewAIInc/crewAI#8070).

Mirrors the documented ``persist_restart_example.py``: a deterministic
conversational Flow (keyword routing, scripted replies — no LLM calls)
persisted to a temporary SQLite database, restarted on a fresh instance,
and isolated by session id.
"""

from __future__ import annotations

from typing import Any

from crewai.flow import ConversationConfig, ConversationState, Flow, listen
from crewai.flow.persistence import SQLiteFlowPersistence, persist


SESSION_ID = "support-session-001"
OTHER_SESSION_ID = "support-session-002"


def _make_bot(db_path: str) -> type[Flow[ConversationState]]:
    @ConversationConfig(defer_trace_finalization=True)
    @persist(SQLiteFlowPersistence(db_path))
    class SupportBot(Flow[ConversationState]):
        """Deterministic support bot: keyword routing, scripted replies, no LLM."""

        def route_turn(self, context: dict[str, Any]) -> str | None:
            message = (self.state.current_user_message or "").lower()
            if "order" in message:
                return "order"
            return "help"

        @listen("order")
        def handle_order(self) -> str:
            reply = (
                f"Order update: '{self.state.current_user_message}' is on its way."
            )
            self.append_assistant_message(reply)
            return reply

        @listen("help")
        def handle_help(self) -> str:
            reply = "I can check order status. Ask me about your order."
            self.append_assistant_message(reply)
            return reply

    return SupportBot


def _roles(messages: Any) -> list[tuple[Any, Any]]:
    return [(message.role, message.content) for message in messages]


class TestPersistenceRestartExample:
    def test_restart_recovers_history_without_duplicates(
        self, tmp_path: Any
    ) -> None:
        SupportBot = _make_bot(str(tmp_path / "support_chat.db"))

        bot = SupportBot()
        try:
            bot.handle_turn("Where is my order?", session_id=SESSION_ID)
            assert _roles(bot.state.messages) == [
                ("user", "Where is my order?"),
                (
                    "assistant",
                    "Order update: 'Where is my order?' is on its way.",
                ),
            ]
        finally:
            bot.finalize_session_traces()

        # Fresh instance, same DB file + session id: history restored,
        # second turn appended in order, no duplicates.
        bot2 = SupportBot()
        try:
            bot2.handle_turn(
                "What about the delivery date?", session_id=SESSION_ID
            )
            assert _roles(bot2.state.messages) == [
                ("user", "Where is my order?"),
                (
                    "assistant",
                    "Order update: 'Where is my order?' is on its way.",
                ),
                ("user", "What about the delivery date?"),
                (
                    "assistant",
                    "I can check order status. Ask me about your order.",
                ),
            ]
        finally:
            bot2.finalize_session_traces()

    def test_different_session_id_starts_separate_conversation(
        self, tmp_path: Any
    ) -> None:
        SupportBot = _make_bot(str(tmp_path / "support_chat.db"))

        bot = SupportBot()
        try:
            bot.handle_turn("Where is my order?", session_id=SESSION_ID)
        finally:
            bot.finalize_session_traces()

        other = SupportBot()
        try:
            other.handle_turn("Where is my order?", session_id=OTHER_SESSION_ID)
            assert _roles(other.state.messages) == [
                ("user", "Where is my order?"),
                (
                    "assistant",
                    "Order update: 'Where is my order?' is on its way.",
                ),
            ]
        finally:
            other.finalize_session_traces()
