"""Experimental /btw commands steer conversational flows without a user turn."""

from __future__ import annotations

from typing import Any

from crewai.experimental.conversation_commands import (
    HELP_TEXT,
    BtwKind,
    btw_commands,
    enable_btw_commands,
    get_btw_steering,
    parse_btw_line,
)
from crewai.flow import ConversationConfig, ConversationState, Flow, listen


class ConversationalFlow(Flow[ConversationState]):
    conversational = True


class TestParseBtwLine:
    def test_plain_message_is_unchanged(self) -> None:
        parsed = parse_btw_line("Where is my order?")
        assert parsed.action is None
        assert parsed.user_message == "Where is my order?"

    def test_leading_note_consumes_the_line(self) -> None:
        parsed = parse_btw_line("/btw keep answers under 20 words")
        assert parsed.action is not None
        assert parsed.action.kind is BtwKind.NOTE
        assert parsed.action.argument == "keep answers under 20 words"
        assert parsed.user_message is None

    def test_inline_note_keeps_the_utterance(self) -> None:
        parsed = parse_btw_line("What's the weather /btw keep it to one sentence")
        assert parsed.action is not None
        assert parsed.action.kind is BtwKind.NOTE
        assert parsed.action.argument == "keep it to one sentence"
        assert parsed.user_message == "What's the weather"

    def test_route_and_persist_forms(self) -> None:
        once = parse_btw_line("/btw route RESEARCH")
        assert once.action is not None
        assert once.action.kind is BtwKind.ROUTE
        assert once.action.argument == "RESEARCH"
        assert once.action.persist_route is False

        persist = parse_btw_line("/BTW stay RESEARCH")
        assert persist.action is not None
        assert persist.action.persist_route is True

    def test_help_and_bare_btw(self) -> None:
        help_line = parse_btw_line("/help")
        assert help_line.action is not None
        assert help_line.action.kind is BtwKind.HELP
        assert help_line.user_message is None

        show = parse_btw_line("/btw")
        assert show.action is not None
        assert show.action.kind is BtwKind.SHOW


@ConversationConfig(defer_trace_finalization=False)
class _RoutedChat(ConversationalFlow):
    turns: int = 0

    def route_turn(self, context: dict[str, Any]) -> str | None:
        message = (self.state.current_user_message or "").lower()
        if "research" in message:
            return "RESEARCH"
        return "work"

    @listen("work")
    def handle_work(self) -> str:
        self.turns += 1
        reply = f"worked: {self.state.current_user_message}"
        self.append_assistant_message(reply)
        return reply

    @listen("RESEARCH")
    def handle_research(self) -> str:
        self.turns += 1
        reply = f"researched: {self.state.current_user_message}"
        self.append_assistant_message(reply)
        return reply


@btw_commands
@ConversationConfig(defer_trace_finalization=False)
class _BtwChat(_RoutedChat):
    pass


class TestBtwCommandsOnFlow:
    def test_note_does_not_run_a_turn_or_append_user_history(self) -> None:
        flow = _BtwChat()
        reply = flow.handle_turn("/btw keep answers under 20 words")

        assert "Noted" in reply
        assert flow.turns == 0
        assert flow.state.messages == []
        assert flow.state.current_user_message is None
        assert get_btw_steering(flow).notes == ["keep answers under 20 words"]
        assert any(event.type == "btw_command" for event in flow.state.events)

    def test_note_is_injected_into_later_system_prompt_and_router_context(
        self,
    ) -> None:
        flow = _BtwChat()
        flow.handle_turn("/btw be terse")

        prompt = flow._resolve_system_prompt()
        assert prompt is not None
        assert "be terse" in prompt

        context = flow.build_router_context()
        assert context["steering_notes"] == ["be terse"]

    def test_later_user_turn_still_runs(self) -> None:
        flow = _BtwChat()
        flow.handle_turn("/btw be terse")
        result = flow.handle_turn("hello there")

        assert result == "worked: hello there"
        assert flow.turns == 1
        assert flow.state.messages[0].role == "user"
        assert flow.state.messages[0].content == "hello there"

    def test_forced_route_wins_for_the_next_turn_only(self) -> None:
        flow = _BtwChat()
        ack = flow.handle_turn("/btw route RESEARCH")
        assert "RESEARCH" in ack

        first = flow.handle_turn("hello there")
        assert first == "researched: hello there"
        assert flow.state.last_intent == "RESEARCH"

        second = flow.handle_turn("hello again")
        assert second == "worked: hello again"

    def test_persist_route_keeps_forcing_until_cleared(self) -> None:
        flow = _BtwChat()
        flow.handle_turn("/btw stay RESEARCH")
        assert flow.handle_turn("hello") == "researched: hello"
        assert flow.handle_turn("again") == "researched: again"

        flow.handle_turn("/btw clear")
        assert flow.handle_turn("hello") == "worked: hello"

    def test_unknown_route_is_rejected(self) -> None:
        flow = _BtwChat()
        reply = flow.handle_turn("/btw route NOPE")
        assert "Unknown route" in reply
        assert get_btw_steering(flow).forced_route is None
        assert flow.turns == 0

    def test_inline_command_steers_the_same_turn(self) -> None:
        flow = _BtwChat()
        result = flow.handle_turn("hello there /btw route RESEARCH")

        assert result == "researched: hello there"
        assert flow.state.messages[0].content == "hello there"
        assert not any(
            "/btw" in str(message.content) for message in flow.state.messages
        )

    def test_help_returns_the_catalog(self) -> None:
        flow = _BtwChat()
        reply = flow.handle_turn("/help")
        assert reply == HELP_TEXT
        assert flow.turns == 0

    def test_chat_repl_surfaces_standalone_acks(self) -> None:
        flow = _BtwChat()
        inputs = iter(["/btw be terse", "hello", "quit"])
        outputs: list[str] = []

        flow.chat(
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
            defer_trace_finalization=False,
        )

        assert any("Noted" in line for line in outputs)
        assert any("worked: hello" in line for line in outputs)
        assert flow.turns == 1

    def test_stream_turn_returns_an_ack_session_for_standalone_commands(self) -> None:
        flow = _BtwChat()
        stream = flow.stream_turn("/btw show")
        with stream:
            frames = list(stream.events)
        assert frames == []
        assert "Current /btw steering" in stream.result

    def test_enable_on_instance_does_not_change_undecorated_classes(self) -> None:
        plain = _RoutedChat()
        enabled = enable_btw_commands(_RoutedChat())

        plain_result = plain.handle_turn("/btw be terse")
        assert plain.turns == 1
        assert plain.state.messages[0].content == "/btw be terse"
        assert "worked: /btw be terse" in str(plain_result)

        ack = enabled.handle_turn("/btw be terse")
        assert "Noted" in ack
        assert enabled.turns == 0
        assert enabled.state.messages == []
