"""Tests for system signal events."""

import signal
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.system_events import (
    SIGNAL_EVENT_TYPES,
    SignalEvent,
    SignalType,
    SigContEvent,
    SigHupEvent,
    SigIntEvent,
    SigTermEvent,
    SigTStpEvent,
    on_signal,
    signal_event_adapter,
)


class TestSignalType:
    """Tests for SignalType enum."""

    def test_signal_type_values(self) -> None:
        """Verify SignalType maps to correct signal numbers."""
        assert SignalType.SIGTERM == signal.SIGTERM
        assert SignalType.SIGINT == signal.SIGINT
        assert SignalType.SIGHUP == getattr(signal, "SIGHUP", 1)
        assert SignalType.SIGTSTP == getattr(signal, "SIGTSTP", 20)
        assert SignalType.SIGCONT == getattr(signal, "SIGCONT", 18)


class TestSignalEvents:
    """Tests for individual signal event classes."""

    def test_sigterm_event_defaults(self) -> None:
        """Test SigTermEvent has correct defaults."""
        event = SigTermEvent()
        assert event.type == "SIGTERM"
        assert event.signal_number == SignalType.SIGTERM
        assert event.reason is None

    def test_sigterm_event_with_reason(self) -> None:
        """Test SigTermEvent can be created with a reason."""
        event = SigTermEvent(reason="graceful shutdown")
        assert event.reason == "graceful shutdown"

    def test_sigint_event_defaults(self) -> None:
        """Test SigIntEvent has correct defaults."""
        event = SigIntEvent()
        assert event.type == "SIGINT"
        assert event.signal_number == SignalType.SIGINT

    def test_sighup_event_defaults(self) -> None:
        """Test SigHupEvent has correct defaults."""
        event = SigHupEvent()
        assert event.type == "SIGHUP"
        assert event.signal_number == SignalType.SIGHUP

    def test_sigtstp_event_defaults(self) -> None:
        """Test SigTStpEvent has correct defaults."""
        event = SigTStpEvent()
        assert event.type == "SIGTSTP"
        assert event.signal_number == SignalType.SIGTSTP

    def test_sigcont_event_defaults(self) -> None:
        """Test SigContEvent has correct defaults."""
        event = SigContEvent()
        assert event.type == "SIGCONT"
        assert event.signal_number == SignalType.SIGCONT


class TestSignalEventAdapter:
    """Tests for the Pydantic discriminated union adapter."""

    def test_adapter_parses_sigterm(self) -> None:
        """Test adapter correctly parses SIGTERM event."""
        data = {"type": "SIGTERM", "reason": "test"}
        event = signal_event_adapter.validate_python(data)
        assert isinstance(event, SigTermEvent)
        assert event.reason == "test"

    def test_adapter_parses_sigint(self) -> None:
        """Test adapter correctly parses SIGINT event."""
        data = {"type": "SIGINT"}
        event = signal_event_adapter.validate_python(data)
        assert isinstance(event, SigIntEvent)

    def test_adapter_parses_sighup(self) -> None:
        """Test adapter correctly parses SIGHUP event."""
        data = {"type": "SIGHUP"}
        event = signal_event_adapter.validate_python(data)
        assert isinstance(event, SigHupEvent)

    def test_adapter_parses_sigtstp(self) -> None:
        """Test adapter correctly parses SIGTSTP event."""
        data = {"type": "SIGTSTP"}
        event = signal_event_adapter.validate_python(data)
        assert isinstance(event, SigTStpEvent)

    def test_adapter_parses_sigcont(self) -> None:
        """Test adapter correctly parses SIGCONT event."""
        data = {"type": "SIGCONT"}
        event = signal_event_adapter.validate_python(data)
        assert isinstance(event, SigContEvent)

    def test_adapter_rejects_invalid_type(self) -> None:
        """Test adapter rejects unknown signal type."""
        data = {"type": "SIGKILL"}
        with pytest.raises(Exception):
            signal_event_adapter.validate_python(data)


class TestSignalEventTypes:
    """Tests for SIGNAL_EVENT_TYPES constant."""

    def test_contains_all_event_types(self) -> None:
        """Verify SIGNAL_EVENT_TYPES contains all signal events."""
        assert SigTermEvent in SIGNAL_EVENT_TYPES
        assert SigIntEvent in SIGNAL_EVENT_TYPES
        assert SigHupEvent in SIGNAL_EVENT_TYPES
        assert SigTStpEvent in SIGNAL_EVENT_TYPES
        assert SigContEvent in SIGNAL_EVENT_TYPES
        assert len(SIGNAL_EVENT_TYPES) == 5


class TestOnSignalDecorator:
    """Tests for the @on_signal decorator."""

    def test_decorator_registers_for_all_signals(self) -> None:
        """Test that @on_signal registers handler for all signal event types."""
        import threading

        received_types: set[str] = set()
        condition = threading.Condition()
        expected_count = len(SIGNAL_EVENT_TYPES)

        @on_signal
        def test_handler(source: object, event: SignalEvent) -> None:
            with condition:
                received_types.add(event.type)
                condition.notify_all()

        for event_class in SIGNAL_EVENT_TYPES:
            crewai_event_bus.emit(self, event_class())

        with condition:
            condition.wait_for(lambda: len(received_types) >= expected_count, timeout=5.0)

        assert "SIGTERM" in received_types
        assert "SIGINT" in received_types
        assert "SIGHUP" in received_types
        assert "SIGTSTP" in received_types
        assert "SIGCONT" in received_types

    def test_decorator_returns_original_function(self) -> None:
        """Test that @on_signal returns the original function."""

        def my_handler(source: object, event: SignalEvent) -> None:
            pass

        decorated = on_signal(my_handler)
        assert decorated is my_handler

    def test_decorator_preserves_function_name(self) -> None:
        """Test that @on_signal preserves function metadata."""

        @on_signal
        def my_named_handler(source: object, event: SignalEvent) -> None:
            """My docstring."""
            pass

        assert my_named_handler.__name__ == "my_named_handler"
        assert my_named_handler.__doc__ == "My docstring."


class TestSignalEventSerialization:
    """Tests for event serialization."""

    def test_sigterm_to_dict(self) -> None:
        """Test SigTermEvent serializes correctly."""
        event = SigTermEvent(reason="test reason")
        data = event.model_dump()
        assert data["type"] == "SIGTERM"
        assert data["signal_number"] == signal.SIGTERM
        assert data["reason"] == "test reason"

    def test_roundtrip_serialization(self) -> None:
        """Test events can be serialized and deserialized."""
        original = SigTermEvent(reason="roundtrip test")
        serialized = original.model_dump()
        restored = signal_event_adapter.validate_python(serialized)
        assert isinstance(restored, SigTermEvent)
        assert restored.reason == original.reason
        assert restored.type == original.type