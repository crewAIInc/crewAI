"""Tests for tracing disabled message suppression (issue #5665).

Verifies that:
- Users who explicitly declined tracing are NOT nagged with the message.
- The CREWAI_SUPPRESS_TRACING_MESSAGES env var suppresses the message.
- The message is shown only when tracing is disabled and user hasn't declined.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.events.listeners.tracing.utils import (
    set_suppress_tracing_messages,
    should_suppress_tracing_messages,
)


class TestShouldSuppressTracingMessages:
    """Tests for the should_suppress_tracing_messages utility function."""

    def test_suppress_false_by_default(self):
        """By default, messages should NOT be suppressed."""
        token = set_suppress_tracing_messages(False)
        try:
            assert should_suppress_tracing_messages() is False
        finally:
            from crewai.events.listeners.tracing.utils import (
                _suppress_tracing_messages,
            )
            _suppress_tracing_messages.reset(token)

    def test_suppress_via_context_var(self):
        """Setting the context var should suppress messages."""
        token = set_suppress_tracing_messages(True)
        try:
            assert should_suppress_tracing_messages() is True
        finally:
            from crewai.events.listeners.tracing.utils import (
                _suppress_tracing_messages,
            )
            _suppress_tracing_messages.reset(token)

    @pytest.mark.parametrize("env_value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_suppress_via_env_var(self, env_value, monkeypatch):
        """CREWAI_SUPPRESS_TRACING_MESSAGES env var should suppress messages."""
        token = set_suppress_tracing_messages(False)
        try:
            monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", env_value)
            assert should_suppress_tracing_messages() is True
        finally:
            from crewai.events.listeners.tracing.utils import (
                _suppress_tracing_messages,
            )
            _suppress_tracing_messages.reset(token)

    @pytest.mark.parametrize("env_value", ["false", "False", "0", "no", ""])
    def test_no_suppress_with_falsy_env_var(self, env_value, monkeypatch):
        """Falsy values for the env var should NOT suppress messages."""
        token = set_suppress_tracing_messages(False)
        try:
            monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", env_value)
            assert should_suppress_tracing_messages() is False
        finally:
            from crewai.events.listeners.tracing.utils import (
                _suppress_tracing_messages,
            )
            _suppress_tracing_messages.reset(token)

    def test_context_var_takes_precedence_over_env(self, monkeypatch):
        """Context var set to True should suppress even if env var is false."""
        token = set_suppress_tracing_messages(True)
        try:
            monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", "false")
            assert should_suppress_tracing_messages() is True
        finally:
            from crewai.events.listeners.tracing.utils import (
                _suppress_tracing_messages,
            )
            _suppress_tracing_messages.reset(token)


class TestShowTracingDisabledMessage:
    """Tests that _show_tracing_disabled_message does not nag declined users."""

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={"first_execution_done": True, "trace_consent": False},
    )
    def test_crew_no_message_when_user_declined(self, mock_load):
        """Crew._show_tracing_disabled_message should not print when user declined."""
        from crewai.crew import Crew

        with patch("crewai.crew.Console") as MockConsole:
            Crew._show_tracing_disabled_message()
            MockConsole.return_value.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={"first_execution_done": True, "trace_consent": False},
    )
    def test_flow_no_message_when_user_declined(self, mock_load):
        """Flow._show_tracing_disabled_message should not print when user declined."""
        from crewai.flow.flow import Flow

        with patch("crewai.flow.flow.Console") as MockConsole:
            Flow._show_tracing_disabled_message()
            MockConsole.return_value.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={"first_execution_done": True, "trace_consent": False},
    )
    def test_trace_listener_no_message_when_user_declined(self, mock_load):
        """TraceCollectionListener._show_tracing_disabled_message should not print when user declined."""
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )

        listener = TraceCollectionListener.__new__(TraceCollectionListener)
        with patch("rich.console.Console") as MockConsole:
            listener._show_tracing_disabled_message()
            MockConsole.return_value.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    def test_crew_shows_message_when_user_has_not_decided(self, mock_load):
        """Crew._show_tracing_disabled_message should print when user hasn't decided yet."""
        from crewai.crew import Crew

        with patch("crewai.crew.Console") as MockConsole:
            mock_console_instance = MockConsole.return_value
            Crew._show_tracing_disabled_message()
            mock_console_instance.print.assert_called_once()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    def test_crew_no_message_when_suppress_env_set(self, mock_load, monkeypatch):
        """Crew._show_tracing_disabled_message should not print when env var suppresses."""
        from crewai.crew import Crew

        monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", "true")
        with patch("crewai.crew.Console") as MockConsole:
            Crew._show_tracing_disabled_message()
            MockConsole.return_value.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    def test_flow_no_message_when_suppress_env_set(self, mock_load, monkeypatch):
        """Flow._show_tracing_disabled_message should not print when env var suppresses."""
        from crewai.flow.flow import Flow

        monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", "true")
        with patch("crewai.flow.flow.Console") as MockConsole:
            Flow._show_tracing_disabled_message()
            MockConsole.return_value.print.assert_not_called()


class TestConsoleFormatterTracingMessage:
    """Tests for console_formatter._show_tracing_disabled_message_if_needed."""

    def _make_formatter(self):
        from crewai.events.utils.console_formatter import ConsoleFormatter

        formatter = ConsoleFormatter.__new__(ConsoleFormatter)
        formatter.console = MagicMock()
        formatter.verbose = True
        return formatter

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={"first_execution_done": True, "trace_consent": False},
    )
    def test_no_message_when_user_declined(self, mock_load):
        """Console formatter should not show the message when user declined tracing."""
        formatter = self._make_formatter()

        with patch(
            "crewai.events.listeners.tracing.trace_listener.TraceCollectionListener"
        ) as mock_listener_cls:
            mock_listener_cls._instance = None
            formatter._show_tracing_disabled_message_if_needed()

        formatter.console.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    def test_no_message_when_suppress_env_set(self, mock_load, monkeypatch):
        """Console formatter should not show the message when env var is set."""
        monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", "true")
        formatter = self._make_formatter()

        formatter._show_tracing_disabled_message_if_needed()

        formatter.console.print.assert_not_called()

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    @patch(
        "crewai.events.listeners.tracing.utils.is_tracing_enabled_in_context",
        return_value=False,
    )
    def test_message_shown_when_tracing_disabled_and_not_declined(
        self, mock_tracing_ctx, mock_load
    ):
        """Console formatter should show the message when tracing disabled and user hasn't declined."""
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )

        formatter = self._make_formatter()

        mock_instance = MagicMock()
        mock_instance.first_time_handler.is_first_time = False
        original_instance = TraceCollectionListener._instance

        try:
            TraceCollectionListener._instance = mock_instance  # type: ignore[misc]
            formatter._show_tracing_disabled_message_if_needed()
            formatter.console.print.assert_called_once()
        finally:
            TraceCollectionListener._instance = original_instance  # type: ignore[misc]

    @patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={},
    )
    @patch(
        "crewai.events.listeners.tracing.utils.is_tracing_enabled_in_context",
        return_value=True,
    )
    def test_no_message_when_tracing_enabled(self, mock_tracing_ctx, mock_load):
        """Console formatter should not show the message when tracing is enabled."""
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )

        formatter = self._make_formatter()

        mock_instance = MagicMock()
        mock_instance.first_time_handler.is_first_time = False
        original_instance = TraceCollectionListener._instance

        try:
            TraceCollectionListener._instance = mock_instance  # type: ignore[misc]
            formatter._show_tracing_disabled_message_if_needed()
            formatter.console.print.assert_not_called()
        finally:
            TraceCollectionListener._instance = original_instance  # type: ignore[misc]
