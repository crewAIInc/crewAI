"""Tests for Flow.ask() user input method.

This module tests the ask() method on Flow, including basic usage,
timeout behavior, provider resolution, event emission, auto-checkpoint
durability, input history tracking, and integration with flow machinery.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import BaseModel

from crewai.flow import Flow, flow_config, listen, start
from crewai.flow.async_feedback.providers import ConsoleProvider
from crewai.flow.flow import FlowState
from crewai.flow.input_provider import InputProvider, InputResponse


# ── Test helpers ─────────────────────────────────────────────────


class MockInputProvider:
    """Mock input provider that returns pre-configured responses."""

    def __init__(self, responses: list[str | None]) -> None:
        self.responses = responses
        self._call_count = 0
        self.messages: list[str] = []
        self.received_metadata: list[dict[str, Any] | None] = []

    def request_input(
        self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
    ) -> str | None:
        self.messages.append(message)
        self.received_metadata.append(metadata)
        if self._call_count >= len(self.responses):
            return None
        response = self.responses[self._call_count]
        self._call_count += 1
        return response


class SlowMockProvider:
    """Mock provider that delays before returning, for timeout tests."""

    def __init__(self, delay: float, response: str = "delayed") -> None:
        self.delay = delay
        self.response = response

    def request_input(
        self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
    ) -> str | None:
        time.sleep(self.delay)
        return self.response


# ── Basic Functionality ──────────────────────────────────────────


class TestAskBasic:
    """Tests for basic ask() functionality."""

    def test_ask_returns_user_input(self) -> None:
        """ask() returns the string from the input provider."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["hello"])

            @start()
            def my_method(self):
                return self.ask("Say something:")

        flow = TestFlow()
        result = flow.kickoff()
        assert result == "hello"

    def test_ask_in_async_method(self) -> None:
        """ask() works inside an async flow method."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["async hello"])

            @start()
            async def my_method(self):
                return self.ask("Say something:")

        flow = TestFlow()
        result = flow.kickoff()
        assert result == "async hello"

    def test_ask_in_start_method(self) -> None:
        """ask() works inside a @start() method, flow completes normally."""
        execution_log: list[str] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI"])

            @start()
            def gather(self):
                topic = self.ask("Topic?")
                execution_log.append(f"got:{topic}")
                return topic

        flow = TestFlow()
        result = flow.kickoff()
        assert result == "AI"
        assert execution_log == ["got:AI"]

    def test_ask_in_listen_method(self) -> None:
        """ask() works inside a @listen() method."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["detailed"])

            @start()
            def step1(self):
                return "topic"

            @listen("step1")
            def step2(self):
                depth = self.ask("How deep?")
                return f"researching at {depth} level"

        flow = TestFlow()
        result = flow.kickoff()
        assert result == "researching at detailed level"

    def test_ask_multiple_calls(self) -> None:
        """Multiple ask() calls in one method return correct values in order."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI", "detailed", "english"])

            @start()
            def gather(self):
                topic = self.ask("Topic?")
                depth = self.ask("Depth?")
                lang = self.ask("Language?")
                return {"topic": topic, "depth": depth, "lang": lang}

        flow = TestFlow()
        result = flow.kickoff()
        assert result == {"topic": "AI", "depth": "detailed", "lang": "english"}

    def test_ask_conditional(self) -> None:
        """ask() called conditionally based on previous answer."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI", "LLMs"])

            @start()
            def gather(self):
                topic = self.ask("Topic?")
                if topic == "AI":
                    focus = self.ask("Specific area?")
                else:
                    focus = "general"
                return {"topic": topic, "focus": focus}

        flow = TestFlow()
        result = flow.kickoff()
        assert result == {"topic": "AI", "focus": "LLMs"}

    def test_ask_returns_empty_string_on_enter(self) -> None:
        """Empty string means user pressed Enter (intentional empty input)."""

        class TestFlow(Flow):
            input_provider = MockInputProvider([""])

            @start()
            def my_method(self):
                result = self.ask("Optional input:")
                return result

        flow = TestFlow()
        result = flow.kickoff()
        assert result == ""
        assert result is not None  # Explicitly not None


# ── Timeout ──────────────────────────────────────────────────────


class TestAskTimeout:
    """Tests for timeout behavior."""

    def test_ask_timeout_returns_none(self) -> None:
        """ask() returns None when timeout expires."""

        class TestFlow(Flow):
            input_provider = SlowMockProvider(delay=5.0)

            @start()
            def my_method(self):
                return self.ask("Question?", timeout=0.1)

        flow = TestFlow()
        result = flow.kickoff()
        assert result is None

    def test_ask_timeout_in_async_method(self) -> None:
        """ask() timeout works inside an async flow method."""

        class TestFlow(Flow):
            input_provider = SlowMockProvider(delay=5.0)

            @start()
            async def my_method(self):
                return self.ask("Question?", timeout=0.1)

        flow = TestFlow()
        result = flow.kickoff()
        assert result is None

    def test_ask_loop_with_timeout_termination(self) -> None:
        """while (msg := ask(...)) is not None pattern terminates on timeout."""
        messages_received: list[str] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["hello", "world", None])

            @start()
            def chat(self):
                while (msg := self.ask("You:")) is not None:
                    messages_received.append(msg)
                return len(messages_received)

        flow = TestFlow()
        result = flow.kickoff()
        assert result == 2
        assert messages_received == ["hello", "world"]

    def test_ask_no_timeout_waits_indefinitely(self) -> None:
        """ask() with no timeout blocks until provider returns."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("Question?")  # no timeout

        flow = TestFlow()
        result = flow.kickoff()
        assert result == "answer"


# ── Provider Resolution ──────────────────────────────────────────


class TestProviderResolution:
    """Tests for provider resolution priority chain."""

    def test_ask_uses_flow_level_provider(self) -> None:
        """Per-flow input_provider is used when set."""
        provider = MockInputProvider(["from flow"])

        class TestFlow(Flow):
            input_provider = provider

            @start()
            def my_method(self):
                return self.ask("Q?")

        flow = TestFlow()
        flow.kickoff()
        assert provider.messages == ["Q?"]

    def test_ask_uses_global_config_provider(self) -> None:
        """flow_config.input_provider is used as fallback."""
        provider = MockInputProvider(["from config"])

        original = flow_config.input_provider
        try:
            flow_config.input_provider = provider

            class TestFlow(Flow):
                @start()
                def my_method(self):
                    return self.ask("Q?")

            flow = TestFlow()
            result = flow.kickoff()
            assert result == "from config"
            assert provider.messages == ["Q?"]
        finally:
            flow_config.input_provider = original

    def test_ask_defaults_to_console_provider(self) -> None:
        """When no provider configured, ConsoleProvider is used."""
        original = flow_config.input_provider
        try:
            flow_config.input_provider = None

            class TestFlow(Flow):
                # No input_provider set
                @start()
                def my_method(self):
                    return self.ask("Q?")

            flow = TestFlow()
            resolved = flow._resolve_input_provider()
            assert isinstance(resolved, ConsoleProvider)
        finally:
            flow_config.input_provider = original

    def test_flow_provider_overrides_global(self) -> None:
        """Per-flow provider takes precedence over global config."""
        flow_provider = MockInputProvider(["from flow"])
        global_provider = MockInputProvider(["from global"])

        original = flow_config.input_provider
        try:
            flow_config.input_provider = global_provider

            class TestFlow(Flow):
                input_provider = flow_provider

                @start()
                def my_method(self):
                    return self.ask("Q?")

            flow = TestFlow()
            result = flow.kickoff()
            assert result == "from flow"
            assert flow_provider.messages == ["Q?"]
            assert global_provider.messages == []  # not called
        finally:
            flow_config.input_provider = original


# ── Events ───────────────────────────────────────────────────────


class TestAskEvents:
    """Tests for event emission during ask()."""

    def test_ask_emits_input_requested_event(self) -> None:
        """FlowInputRequestedEvent is emitted when ask() is called."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import FlowInputRequestedEvent

        events_captured: list[FlowInputRequestedEvent] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("What topic?")

        flow = TestFlow()

        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowInputRequestedEvent):
                events_captured.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert len(events_captured) == 1
        assert events_captured[0].message == "What topic?"
        assert events_captured[0].type == "flow_input_requested"

    def test_ask_emits_input_received_event(self) -> None:
        """FlowInputReceivedEvent is emitted after input is received."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import FlowInputReceivedEvent

        events_captured: list[FlowInputReceivedEvent] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["my answer"])

            @start()
            def my_method(self):
                return self.ask("Question?")

        flow = TestFlow()

        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowInputReceivedEvent):
                events_captured.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert len(events_captured) == 1
        assert events_captured[0].message == "Question?"
        assert events_captured[0].response == "my answer"
        assert events_captured[0].type == "flow_input_received"

    def test_ask_timeout_emits_received_with_none(self) -> None:
        """FlowInputReceivedEvent has response=None on timeout."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import FlowInputReceivedEvent

        events_captured: list[FlowInputReceivedEvent] = []

        class TestFlow(Flow):
            input_provider = SlowMockProvider(delay=5.0)

            @start()
            def my_method(self):
                return self.ask("Question?", timeout=0.1)

        flow = TestFlow()

        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowInputReceivedEvent):
                events_captured.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert len(events_captured) == 1
        assert events_captured[0].response is None


# ── Auto-checkpoint (Durability) ─────────────────────────────────


class TestAskCheckpoint:
    """Tests for auto-checkpoint durability before ask() waits."""

    def test_ask_checkpoints_state_before_waiting(self) -> None:
        """State is saved to persistence before waiting for input."""
        mock_persistence = MagicMock()
        mock_persistence.load_state.return_value = None

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                self.state["important"] = "data"
                return self.ask("Question?")

        flow = TestFlow(persistence=mock_persistence)
        flow.kickoff()

        # Find the _ask_checkpoint call among save_state calls
        checkpoint_calls = [
            c for c in mock_persistence.save_state.call_args_list
            if c.kwargs.get("method_name") == "_ask_checkpoint"
            or (len(c.args) >= 2 and c.args[1] == "_ask_checkpoint")
        ]
        assert len(checkpoint_calls) >= 1

    def test_ask_no_checkpoint_without_persistence(self) -> None:
        """No error when persistence is not configured."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("Question?")

        flow = TestFlow()  # No persistence
        result = flow.kickoff()
        assert result == "answer"  # Works fine without persistence

    def test_state_recoverable_after_checkpoint(self) -> None:
        """State set before ask() is checkpointed and recoverable.

        The auto-checkpoint happens *before* the provider is called, so
        state values set prior to ask() are persisted. This means if the
        server crashes while waiting for input, previously gathered data
        is safe.
        """
        mock_persistence = MagicMock()
        mock_persistence.load_state.return_value = None

        class GatherFlow(Flow):
            input_provider = MockInputProvider(["AI", "detailed"])

            @start()
            def gather(self):
                # First ask: nothing in state yet
                topic = self.ask("Topic?")
                self.state["topic"] = topic
                # Second ask: state now has topic, checkpoint saves it
                depth = self.ask("Depth?")
                self.state["depth"] = depth
                return {"topic": topic, "depth": depth}

        flow = GatherFlow(persistence=mock_persistence)
        result = flow.kickoff()
        assert result == {"topic": "AI", "depth": "detailed"}

        # Find the checkpoint calls
        checkpoint_calls = [
            c for c in mock_persistence.save_state.call_args_list
            if c.kwargs.get("method_name") == "_ask_checkpoint"
            or (len(c.args) >= 2 and c.args[1] == "_ask_checkpoint")
        ]
        assert len(checkpoint_calls) == 2

        # The second checkpoint (before asking "Depth?") should have topic
        second_checkpoint = checkpoint_calls[1]
        # state_data is the third positional arg or keyword arg
        if second_checkpoint.kwargs.get("state_data"):
            state_data = second_checkpoint.kwargs["state_data"]
        else:
            state_data = second_checkpoint.args[2]
        assert state_data.get("topic") == "AI"


# ── Input History ────────────────────────────────────────────────


class TestInputHistory:
    """Tests for _input_history tracking."""

    def test_input_history_accumulated(self) -> None:
        """_input_history tracks all ask/response pairs."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI", "detailed"])

            @start()
            def gather(self):
                self.ask("Topic?")
                self.ask("Depth?")
                return "done"

        flow = TestFlow()
        flow.kickoff()

        assert len(flow._input_history) == 2
        assert flow._input_history[0]["message"] == "Topic?"
        assert flow._input_history[0]["response"] == "AI"
        assert flow._input_history[1]["message"] == "Depth?"
        assert flow._input_history[1]["response"] == "detailed"

    def test_input_history_includes_method_name(self) -> None:
        """Input history records which method called ask()."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI"])

            @start()
            def gather_info(self):
                self.ask("Topic?")
                return "done"

        flow = TestFlow()
        flow.kickoff()

        assert len(flow._input_history) == 1
        assert flow._input_history[0]["method_name"] == "gather_info"

    def test_input_history_includes_timestamp(self) -> None:
        """Input history records timestamps."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI"])

            @start()
            def my_method(self):
                self.ask("Topic?")
                return "done"

        flow = TestFlow()
        before = datetime.now()
        flow.kickoff()
        after = datetime.now()

        assert len(flow._input_history) == 1
        ts = flow._input_history[0]["timestamp"]
        assert isinstance(ts, datetime)
        assert before <= ts <= after

    def test_input_history_records_none_on_timeout(self) -> None:
        """Input history records None response on timeout."""

        class TestFlow(Flow):
            input_provider = SlowMockProvider(delay=5.0)

            @start()
            def my_method(self):
                self.ask("Question?", timeout=0.1)
                return "done"

        flow = TestFlow()
        flow.kickoff()

        assert len(flow._input_history) == 1
        assert flow._input_history[0]["response"] is None


# ── Integration ──────────────────────────────────────────────────


class TestAskIntegration:
    """Integration tests for ask() with other flow features."""

    def test_ask_works_with_listen_chain(self) -> None:
        """ask() in a start method, result flows to listener."""
        execution_log: list[str] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI agents"])

            @start()
            def gather(self):
                topic = self.ask("Topic?")
                execution_log.append(f"gathered:{topic}")
                return topic

            @listen("gather")
            def process(self):
                execution_log.append("processing")
                return "processed"

        flow = TestFlow()
        result = flow.kickoff()
        assert "gathered:AI agents" in execution_log
        assert "processing" in execution_log

    def test_ask_with_structured_state(self) -> None:
        """ask() works with Pydantic-based flow state."""

        class ResearchState(FlowState):
            topic: str = ""
            depth: str = ""

        class TestFlow(Flow[ResearchState]):
            initial_state = ResearchState
            input_provider = MockInputProvider(["AI", "detailed"])

            @start()
            def gather(self):
                self.state.topic = self.ask("Topic?")
                self.state.depth = self.ask("Depth?")
                return {"topic": self.state.topic, "depth": self.state.depth}

        flow = TestFlow()
        result = flow.kickoff()
        assert result == {"topic": "AI", "depth": "detailed"}
        assert flow.state.topic == "AI"
        assert flow.state.depth == "detailed"

    def test_ask_in_async_method_with_listen_chain(self) -> None:
        """ask() in an async start method, result flows to listener."""
        execution_log: list[str] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["async topic"])

            @start()
            async def gather(self):
                topic = self.ask("Topic?")
                execution_log.append(f"gathered:{topic}")
                return topic

            @listen("gather")
            def process(self):
                execution_log.append("processing")
                return "processed"

        flow = TestFlow()
        flow.kickoff()
        assert "gathered:async topic" in execution_log
        assert "processing" in execution_log

    def test_ask_with_state_persistence_recovery(self) -> None:
        """Ask checkpoints state so previously gathered values survive."""
        mock_persistence = MagicMock()
        mock_persistence.load_state.return_value = None

        class RecoverableFlow(Flow):
            input_provider = MockInputProvider(["AI", "detailed"])

            @start()
            def gather(self):
                if not self.state.get("topic"):
                    self.state["topic"] = self.ask("Topic?")
                if not self.state.get("depth"):
                    self.state["depth"] = self.ask("Depth?")
                return {
                    "topic": self.state["topic"],
                    "depth": self.state["depth"],
                }

        flow = RecoverableFlow(persistence=mock_persistence)
        result = flow.kickoff()
        assert result["topic"] == "AI"
        assert result["depth"] == "detailed"

        # Verify checkpoints were made
        checkpoint_calls = [
            c for c in mock_persistence.save_state.call_args_list
            if c.kwargs.get("method_name") == "_ask_checkpoint"
            or (len(c.args) >= 2 and c.args[1] == "_ask_checkpoint")
        ]
        # Two ask() calls = two checkpoints
        assert len(checkpoint_calls) == 2

    def test_ask_and_human_feedback_coexist(self) -> None:
        """ask() and @human_feedback can be used in the same flow."""
        from crewai.flow import human_feedback

        class TestFlow(Flow):
            input_provider = MockInputProvider(["AI"])

            @start()
            def gather(self):
                topic = self.ask("Topic?")
                return topic

            @listen("gather")
            @human_feedback(message="Review this topic:")
            def review(self):
                return f"Researching: {self.state.get('_last_topic', 'unknown')}"

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value="looks good"):
            result = flow.kickoff()

        # Flow completed with both ask and human_feedback
        assert flow.last_human_feedback is not None

    def test_ask_preserves_flow_lifecycle(self) -> None:
        """Flow events (started, finished) still fire normally with ask()."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import (
            FlowFinishedEvent,
            FlowStartedEvent,
        )

        events_seen: list[str] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("Q?")

        flow = TestFlow()

        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowStartedEvent):
                events_seen.append("started")
            elif isinstance(event, FlowFinishedEvent):
                events_seen.append("finished")
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert "started" in events_seen
        assert "finished" in events_seen


# ── Console Provider ─────────────────────────────────────────────


class TestConsoleProviderInput:
    """Tests for ConsoleProvider.request_input() (used by Flow.ask())."""

    def test_console_provider_pauses_live_updates(self) -> None:
        """ConsoleProvider pauses and resumes formatter live updates."""
        from crewai.events.event_listener import event_listener

        mock_formatter = MagicMock()
        mock_formatter.console = MagicMock()

        provider = ConsoleProvider(verbose=True)

        with (
            patch.object(event_listener, "formatter", mock_formatter),
            patch("builtins.input", return_value="test input"),
        ):
            result = provider.request_input("Question?", MagicMock())

        mock_formatter.pause_live_updates.assert_called_once()
        mock_formatter.resume_live_updates.assert_called_once()
        assert result == "test input"

    def test_console_provider_displays_message(self) -> None:
        """ConsoleProvider displays the message with Rich console."""
        from crewai.events.event_listener import event_listener

        mock_formatter = MagicMock()
        mock_console = MagicMock()
        mock_formatter.console = mock_console

        provider = ConsoleProvider(verbose=True)

        with (
            patch.object(event_listener, "formatter", mock_formatter),
            patch("builtins.input", return_value="answer"),
        ):
            provider.request_input("What topic?", MagicMock())

        # Verify the message was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("What topic?" in c for c in print_calls)

    def test_console_provider_non_verbose(self) -> None:
        """ConsoleProvider in non-verbose mode uses plain input."""
        from crewai.events.event_listener import event_listener

        mock_formatter = MagicMock()
        mock_formatter.console = MagicMock()

        provider = ConsoleProvider(verbose=False)

        with (
            patch.object(event_listener, "formatter", mock_formatter),
            patch("builtins.input", return_value="plain answer") as mock_input,
        ):
            result = provider.request_input("Q?", MagicMock())

        assert result == "plain answer"
        mock_input.assert_called_once_with("Q? ")

    def test_console_provider_strips_response(self) -> None:
        """ConsoleProvider strips whitespace from response."""
        from crewai.events.event_listener import event_listener

        mock_formatter = MagicMock()
        mock_formatter.console = MagicMock()

        provider = ConsoleProvider(verbose=False)

        with (
            patch.object(event_listener, "formatter", mock_formatter),
            patch("builtins.input", return_value="  spaced answer  "),
        ):
            result = provider.request_input("Q?", MagicMock())

        assert result == "spaced answer"

    def test_console_provider_implements_protocol(self) -> None:
        """ConsoleProvider satisfies the InputProvider protocol."""
        provider = ConsoleProvider()
        assert isinstance(provider, InputProvider)


# ── InputProvider Protocol ───────────────────────────────────────


class TestInputProviderProtocol:
    """Tests for the InputProvider protocol."""

    def test_custom_provider_satisfies_protocol(self) -> None:
        """A class with request_input satisfies the InputProvider protocol."""

        class MyProvider:
            def request_input(self, message: str, flow: Flow[Any]) -> str | None:
                return "custom"

        provider = MyProvider()
        assert isinstance(provider, InputProvider)

    def test_mock_provider_satisfies_protocol(self) -> None:
        """MockInputProvider satisfies the InputProvider protocol."""
        provider = MockInputProvider(["test"])
        assert isinstance(provider, InputProvider)


# ── Error Handling ───────────────────────────────────────────────


class TestAskErrorHandling:
    """Tests for error handling in ask()."""

    def test_ask_returns_none_on_provider_error(self) -> None:
        """ask() returns None if provider raises an exception."""

        class FailingProvider:
            def request_input(self, message: str, flow: Flow[Any]) -> str | None:
                raise RuntimeError("Provider failed")

        class TestFlow(Flow):
            input_provider = FailingProvider()

            @start()
            def my_method(self):
                return self.ask("Question?")

        flow = TestFlow()
        result = flow.kickoff()
        assert result is None

    def test_ask_in_async_method_returns_none_on_provider_error(self) -> None:
        """ask() returns None if provider raises in an async method."""

        class FailingProvider:
            def request_input(self, message: str, flow: Flow[Any]) -> str | None:
                raise RuntimeError("Provider failed")

        class TestFlow(Flow):
            input_provider = FailingProvider()

            @start()
            async def my_method(self):
                return self.ask("Question?")

        flow = TestFlow()
        result = flow.kickoff()
        assert result is None


# ── Metadata ─────────────────────────────────────────────────────


class TestAskMetadata:
    """Tests for bidirectional metadata support in ask()."""

    def test_ask_passes_metadata_to_provider(self) -> None:
        """Provider receives the metadata dict from ask()."""
        provider = MockInputProvider(["answer"])

        class TestFlow(Flow):
            input_provider = provider

            @start()
            def my_method(self):
                return self.ask("Q?", metadata={"user_id": "u123"})

        flow = TestFlow()
        flow.kickoff()
        assert provider.received_metadata == [{"user_id": "u123"}]

    def test_ask_metadata_none_by_default(self) -> None:
        """Provider receives None metadata when not provided."""
        provider = MockInputProvider(["answer"])

        class TestFlow(Flow):
            input_provider = provider

            @start()
            def my_method(self):
                return self.ask("Q?")

        flow = TestFlow()
        flow.kickoff()
        assert provider.received_metadata == [None]

    def test_ask_provider_returns_input_response(self) -> None:
        """Provider returns InputResponse with response metadata."""

        class MetadataProvider:
            def request_input(
                self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
            ) -> InputResponse:
                return InputResponse(
                    text="the answer",
                    metadata={"responded_by": "u456", "thread_id": "t789"},
                )

        class TestFlow(Flow):
            input_provider = MetadataProvider()

            @start()
            def my_method(self):
                return self.ask("Q?", metadata={"user_id": "u123"})

        flow = TestFlow()
        result = flow.kickoff()

        # ask() still returns plain string
        assert result == "the answer"

        # History has both metadata dicts
        assert len(flow._input_history) == 1
        entry = flow._input_history[0]
        assert entry["metadata"] == {"user_id": "u123"}
        assert entry["response_metadata"] == {"responded_by": "u456", "thread_id": "t789"}

    def test_ask_provider_returns_string_with_metadata_sent(self) -> None:
        """Provider returns plain string; history has metadata but no response_metadata."""

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("Q?", metadata={"channel": "#research"})

        flow = TestFlow()
        flow.kickoff()

        entry = flow._input_history[0]
        assert entry["metadata"] == {"channel": "#research"}
        assert entry["response_metadata"] is None

    def test_ask_metadata_in_requested_event(self) -> None:
        """FlowInputRequestedEvent carries metadata."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import FlowInputRequestedEvent

        events_captured: list[FlowInputRequestedEvent] = []

        class TestFlow(Flow):
            input_provider = MockInputProvider(["answer"])

            @start()
            def my_method(self):
                return self.ask("Q?", metadata={"user_id": "u123"})

        flow = TestFlow()
        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowInputRequestedEvent):
                events_captured.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert len(events_captured) == 1
        assert events_captured[0].metadata == {"user_id": "u123"}

    def test_ask_metadata_in_received_event(self) -> None:
        """FlowInputReceivedEvent carries both metadata and response_metadata."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.flow_events import FlowInputReceivedEvent

        events_captured: list[FlowInputReceivedEvent] = []

        class MetadataProvider:
            def request_input(
                self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
            ) -> InputResponse:
                return InputResponse(text="answer", metadata={"responded_by": "u456"})

        class TestFlow(Flow):
            input_provider = MetadataProvider()

            @start()
            def my_method(self):
                return self.ask("Q?", metadata={"user_id": "u123"})

        flow = TestFlow()
        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(event, FlowInputReceivedEvent):
                events_captured.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.kickoff()

        assert len(events_captured) == 1
        assert events_captured[0].metadata == {"user_id": "u123"}
        assert events_captured[0].response_metadata == {"responded_by": "u456"}
        assert events_captured[0].response == "answer"

    def test_ask_input_response_with_none_text(self) -> None:
        """Provider returns InputResponse with text=None."""

        class NoneTextProvider:
            def request_input(
                self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
            ) -> InputResponse:
                return InputResponse(text=None, metadata={"reason": "user_declined"})

        class TestFlow(Flow):
            input_provider = NoneTextProvider()

            @start()
            def my_method(self):
                return self.ask("Q?")

        flow = TestFlow()
        result = flow.kickoff()
        assert result is None

        entry = flow._input_history[0]
        assert entry["response"] is None
        assert entry["response_metadata"] == {"reason": "user_declined"}

    def test_ask_metadata_thread_safe(self) -> None:
        """Concurrent ask() calls with different metadata don't cross-contaminate."""
        import threading

        call_log: list[dict[str, Any]] = []
        log_lock = threading.Lock()

        class TrackingProvider:
            def request_input(
                self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
            ) -> InputResponse:
                # Small delay to increase chance of interleaving
                time.sleep(0.05)
                with log_lock:
                    call_log.append({"message": message, "metadata": metadata})
                user = metadata.get("user", "unknown") if metadata else "unknown"
                return InputResponse(
                    text=f"answer from {user}",
                    metadata={"responded_by": user},
                )

        class TestFlow(Flow):
            input_provider = TrackingProvider()

            @start()
            def trigger(self):
                return "go"

            @listen("trigger")
            def listener_a(self):
                return self.ask("Question A?", metadata={"user": "alice"})

            @listen("trigger")
            def listener_b(self):
                return self.ask("Question B?", metadata={"user": "bob"})

        flow = TestFlow()
        flow.kickoff()

        # Both calls should have recorded their own metadata
        assert len(flow._input_history) == 2

        alice_entry = next(
            (e for e in flow._input_history if e["metadata"] and e["metadata"].get("user") == "alice"),
            None,
        )
        bob_entry = next(
            (e for e in flow._input_history if e["metadata"] and e["metadata"].get("user") == "bob"),
            None,
        )

        assert alice_entry is not None
        assert alice_entry["response"] == "answer from alice"
        assert alice_entry["response_metadata"] == {"responded_by": "alice"}

        assert bob_entry is not None
        assert bob_entry["response"] == "answer from bob"
        assert bob_entry["response_metadata"] == {"responded_by": "bob"}
