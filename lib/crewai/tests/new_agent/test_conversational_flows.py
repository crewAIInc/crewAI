"""Tests for Flow.ask() and Flow.say() with ConversationalProvider integration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.flow.flow import Flow, start
from crewai.new_agent.models import Message
from crewai.new_agent.provider import ConversationalProvider, DirectProvider


# ── Helpers ─────────────────────────────────────────────────────


class MockConversationalProvider:
    """A mock ConversationalProvider that records sent messages and
    returns pre-configured replies for receive_message().
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self._reply_index = 0
        self.sent_messages: list[Message] = []
        self.statuses: list[Any] = []

    async def send_message(self, message: Message) -> None:
        self.sent_messages.append(message)

    async def receive_message(self) -> Message:
        if self._reply_index < len(self._replies):
            content = self._replies[self._reply_index]
            self._reply_index += 1
            return Message(role="user", content=content)
        return Message(role="user", content="")

    async def send_status(self, status: Any) -> None:
        self.statuses.append(status)

    def get_history(self) -> list[Message]:
        return list(self.sent_messages)

    def save_history(self, messages: list[Message]) -> None:
        pass

    def reset_history(self) -> None:
        self.sent_messages.clear()

    def save_provenance(self, entries: list) -> None:
        pass

    def load_provenance(self) -> list:
        return []

    def get_scope(self) -> dict[str, str]:
        return {}


# ── Test Flows ──────────────────────────────────────────────────


class SimpleAskFlow(Flow):
    """Flow that asks a single question."""

    _skip_auto_memory = True

    @start()
    def greet(self):
        answer = self.ask("What is your name?")
        self.state["answer"] = answer
        return answer


class SimpleSayFlow(Flow):
    """Flow that sends a message without waiting for a response."""

    _skip_auto_memory = True

    @start()
    def notify(self):
        self.say("Processing started...")
        self.state["notified"] = True
        return "done"


class AskAndSayFlow(Flow):
    """Flow that uses both ask() and say()."""

    _skip_auto_memory = True

    @start()
    def interact(self):
        self.say("Welcome to the interactive flow!")
        name = self.ask("What is your name?")
        self.say(f"Hello, {name}! Processing your request...")
        topic = self.ask("What topic interests you?")
        self.say(f"Great choice, {name}! Researching {topic}...")
        self.state["name"] = name
        self.state["topic"] = topic
        return {"name": name, "topic": topic}


class MetadataFlow(Flow):
    """Flow that passes metadata through ask() and say()."""

    _skip_auto_memory = True

    @start()
    def with_metadata(self):
        self.say("Starting", metadata={"channel": "#ops"})
        answer = self.ask("Continue?", metadata={"user_id": "u123"})
        self.state["answer"] = answer
        return answer


# ── Tests: ConversationalProvider field ─────────────────────────


class TestConversationalProviderField:
    def test_default_is_none(self):
        flow = Flow(_skip_auto_memory=True, suppress_flow_events=True)
        assert flow.conversational_provider is None

    def test_can_set_provider(self):
        provider = MockConversationalProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        assert flow.conversational_provider is provider

    def test_provider_implements_protocol(self):
        provider = MockConversationalProvider()
        assert isinstance(provider, ConversationalProvider)


# ── Tests: ask() with ConversationalProvider ────────────────────


class TestAskWithConversationalProvider:
    def test_ask_sends_and_receives(self):
        provider = MockConversationalProvider(replies=["Alice"])
        flow = SimpleAskFlow(
            conversational_provider=provider,
            suppress_flow_events=True,
        )
        result = flow.kickoff()
        assert result == "Alice"
        assert flow.state["answer"] == "Alice"
        # The provider should have received the question
        assert len(provider.sent_messages) == 1
        assert provider.sent_messages[0].content == "What is your name?"
        assert provider.sent_messages[0].role == "agent"

    def test_ask_returns_none_on_timeout(self):
        class SlowProvider(MockConversationalProvider):
            async def receive_message(self) -> Message:
                await asyncio.sleep(10)
                return Message(role="user", content="too late")

        provider = SlowProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        result = flow.ask("Quick question?", timeout=0.1)
        assert result is None

    def test_ask_returns_none_on_provider_error(self):
        class BrokenProvider(MockConversationalProvider):
            async def receive_message(self) -> Message:
                raise ConnectionError("Provider disconnected")

        provider = BrokenProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        result = flow.ask("Hello?")
        assert result is None

    def test_ask_records_input_history(self):
        provider = MockConversationalProvider(replies=["Bob"])
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        flow.ask("Who are you?")
        assert len(flow._input_history) == 1
        entry = flow._input_history[0]
        assert entry["message"] == "Who are you?"
        assert entry["response"] == "Bob"

    def test_ask_with_metadata(self):
        provider = MockConversationalProvider(replies=["yes"])
        flow = MetadataFlow(
            conversational_provider=provider,
            suppress_flow_events=True,
        )
        result = flow.kickoff()
        assert result == "yes"
        # Check that the ask message was sent with correct metadata
        ask_msgs = [m for m in provider.sent_messages if "Continue" in m.content]
        assert len(ask_msgs) == 1
        assert ask_msgs[0].metadata == {"user_id": "u123"}


# ── Tests: say() ────────────────────────────────────────────────


class TestSayWithConversationalProvider:
    def test_say_sends_message(self):
        provider = MockConversationalProvider()
        flow = SimpleSayFlow(
            conversational_provider=provider,
            suppress_flow_events=True,
        )
        result = flow.kickoff()
        assert result == "done"
        assert flow.state["notified"] is True
        assert len(provider.sent_messages) == 1
        assert provider.sent_messages[0].content == "Processing started..."
        assert provider.sent_messages[0].role == "agent"

    def test_say_with_metadata(self):
        provider = MockConversationalProvider()
        flow = MetadataFlow(
            conversational_provider=provider,
            suppress_flow_events=True,
        )
        # We need a reply for the ask() call
        provider._replies = ["ok"]
        flow.kickoff()
        # The say("Starting") message should have metadata
        say_msgs = [m for m in provider.sent_messages if m.content == "Starting"]
        assert len(say_msgs) == 1
        assert say_msgs[0].metadata == {"channel": "#ops"}

    def test_say_does_not_block(self):
        """say() should not wait for a response -- it's fire-and-forget."""
        provider = MockConversationalProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        # say() should return None (no return value)
        result = flow.say("Hello!")
        assert result is None
        assert len(provider.sent_messages) == 1

    def test_say_gracefully_handles_provider_error(self):
        class BrokenSayProvider(MockConversationalProvider):
            async def send_message(self, message: Message) -> None:
                raise ConnectionError("Cannot send")

        provider = BrokenSayProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        # Should not raise -- errors are logged and swallowed
        flow.say("This will fail silently")


class TestSayWithoutProvider:
    def test_say_prints_to_console(self):
        flow = Flow(
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        # Without a conversational_provider, say() falls back to console
        with patch("crewai.flow.flow.Console") as MockConsole:
            mock_console = MagicMock()
            MockConsole.return_value = mock_console
            flow.say("Console message")
            mock_console.print.assert_called_once()
            # Verify the Panel was created with the message
            call_args = mock_console.print.call_args
            panel = call_args[0][0]
            # The Panel renderable should contain our message
            assert "Console message" in str(panel.renderable)


# ── Tests: Combined ask() and say() ────────────────────────────


class TestAskAndSayCombined:
    def test_full_conversation_flow(self):
        provider = MockConversationalProvider(replies=["Alice", "AI"])
        flow = AskAndSayFlow(
            conversational_provider=provider,
            suppress_flow_events=True,
        )
        result = flow.kickoff()
        assert result == {"name": "Alice", "topic": "AI"}
        assert flow.state["name"] == "Alice"
        assert flow.state["topic"] == "AI"

        # Check all sent messages in order
        contents = [m.content for m in provider.sent_messages]
        assert contents == [
            "Welcome to the interactive flow!",
            "What is your name?",
            "Hello, Alice! Processing your request...",
            "What topic interests you?",
            "Great choice, Alice! Researching AI...",
        ]

    def test_mixed_say_and_ask_message_roles(self):
        provider = MockConversationalProvider(replies=["yes"])
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        flow.say("Info message")
        flow.ask("Question?")

        # Both say() and ask() send as "agent" role
        assert all(m.role == "agent" for m in provider.sent_messages)


# ── Tests: Fallback behavior (no conversational_provider) ──────


class MockInputProvider:
    """A mock InputProvider that returns a pre-configured response."""

    def __init__(self, response: str = "fallback answer") -> None:
        self._response = response
        self.call_count = 0

    def request_input(
        self,
        message: str,
        flow: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        self.call_count += 1
        return self._response


class TestFallbackBehavior:
    def test_ask_falls_back_to_input_provider(self):
        """When no conversational_provider is set, ask() uses InputProvider."""
        mock_input_provider = MockInputProvider("fallback answer")

        flow = Flow(
            input_provider=mock_input_provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        result = flow.ask("Test question?")
        assert result == "fallback answer"
        assert mock_input_provider.call_count == 1

    def test_conversational_provider_takes_priority(self):
        """When both providers are set, conversational_provider wins for ask()."""
        conv_provider = MockConversationalProvider(replies=["conv answer"])
        input_provider = MockInputProvider("input answer")

        flow = Flow(
            conversational_provider=conv_provider,
            input_provider=input_provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        result = flow.ask("Which provider?")
        assert result == "conv answer"
        # InputProvider should NOT have been called
        assert input_provider.call_count == 0


# ── Tests: Events ───────────────────────────────────────────────


class TestFlowMessageEvents:
    def test_say_emits_flow_message_sent_event(self):
        from crewai.events.types.flow_events import FlowMessageSentEvent

        provider = MockConversationalProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        emitted_events: list[FlowMessageSentEvent] = []

        original_emit = crewai_event_bus_emit = None
        with patch.object(
            type(flow), "_Flow__class__", create=True
        ):
            pass

        # We'll check that the event is emitted by patching crewai_event_bus
        with patch("crewai.flow.flow.crewai_event_bus") as mock_bus:
            flow.say("Test message", metadata={"key": "value"})

            # Find the FlowMessageSentEvent among emitted events
            for call in mock_bus.emit.call_args_list:
                args = call[0]
                if len(args) >= 2 and isinstance(args[1], FlowMessageSentEvent):
                    event = args[1]
                    assert event.message == "Test message"
                    assert event.metadata == {"key": "value"}
                    assert event.type == "flow_message_sent"
                    emitted_events.append(event)

            assert len(emitted_events) == 1

    def test_ask_emits_input_events_with_conv_provider(self):
        from crewai.events.types.flow_events import (
            FlowInputReceivedEvent,
            FlowInputRequestedEvent,
        )

        provider = MockConversationalProvider(replies=["answer"])
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )

        with patch("crewai.flow.flow.crewai_event_bus") as mock_bus:
            flow.ask("Question?")

            requested = [
                call[0][1]
                for call in mock_bus.emit.call_args_list
                if isinstance(call[0][1], FlowInputRequestedEvent)
            ]
            received = [
                call[0][1]
                for call in mock_bus.emit.call_args_list
                if isinstance(call[0][1], FlowInputReceivedEvent)
            ]

            assert len(requested) == 1
            assert requested[0].message == "Question?"
            assert len(received) == 1
            assert received[0].response == "answer"


# ── Tests: DirectProvider as conversational_provider ────────────


class TestDirectProviderIntegration:
    def test_direct_provider_send_only(self):
        """DirectProvider supports send_message but not receive_message."""
        provider = DirectProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        # say() should work
        flow.say("Hello from flow")
        assert len(provider.get_history()) == 1
        assert provider.get_history()[0].content == "Hello from flow"

    def test_direct_provider_ask_returns_none(self):
        """DirectProvider.receive_message raises NotImplementedError,
        so ask() should return None gracefully."""
        provider = DirectProvider()
        flow = Flow(
            conversational_provider=provider,
            _skip_auto_memory=True,
            suppress_flow_events=True,
        )
        result = flow.ask("Will fail gracefully")
        assert result is None
