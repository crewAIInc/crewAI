"""Test verbose control for Flow and Crew."""

import os
from unittest.mock import patch

from crewai.events.event_listener import EventListener
from crewai.flow.flow import Flow, start, listen
from crewai.utilities.logger_utils import should_enable_verbose


class TestShouldEnableVerbose:
    """Test the should_enable_verbose utility function."""

    def test_override_true_returns_true(self):
        """Test that explicit override=True always returns True."""
        assert should_enable_verbose(override=True) is True

    def test_override_false_returns_false(self):
        """Test that explicit override=False always returns False."""
        assert should_enable_verbose(override=False) is False

    def test_env_var_false_disables_verbose(self):
        """Test that CREWAI_VERBOSE=false disables verbose."""
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "false"}):
            assert should_enable_verbose() is False

    def test_env_var_0_disables_verbose(self):
        """Test that CREWAI_VERBOSE=0 disables verbose."""
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "0"}):
            assert should_enable_verbose() is False

    def test_env_var_true_enables_verbose(self):
        """Test that CREWAI_VERBOSE=true enables verbose."""
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "true"}):
            assert should_enable_verbose() is True

    def test_env_var_1_enables_verbose(self):
        """Test that CREWAI_VERBOSE=1 enables verbose."""
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "1"}):
            assert should_enable_verbose() is True

    def test_no_env_var_defaults_to_true(self):
        """Test that no CREWAI_VERBOSE env var defaults to True."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove CREWAI_VERBOSE if it exists
            os.environ.pop("CREWAI_VERBOSE", None)
            assert should_enable_verbose() is True

    def test_override_takes_precedence_over_env_var(self):
        """Test that explicit override takes precedence over env var."""
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "false"}):
            assert should_enable_verbose(override=True) is True

        with patch.dict(os.environ, {"CREWAI_VERBOSE": "true"}):
            assert should_enable_verbose(override=False) is False


class TestFlowVerboseControl:
    """Test verbose control in Flow class."""

    def test_flow_verbose_default_is_true(self):
        """Test that Flow verbose defaults to True when no env var is set."""
        # Remove CREWAI_VERBOSE if it exists
        os.environ.pop("CREWAI_VERBOSE", None)

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        flow = SimpleFlow()
        assert flow.verbose is True

    def test_flow_verbose_false_disables_logging(self):
        """Test that Flow with verbose=False disables logging."""

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        flow = SimpleFlow(verbose=False)
        assert flow.verbose is False

        # Verify EventListener is also set to verbose=False
        event_listener = EventListener()
        assert event_listener.verbose is False
        assert event_listener.formatter.verbose is False

    def test_flow_verbose_true_enables_logging(self):
        """Test that Flow with verbose=True enables logging."""

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        flow = SimpleFlow(verbose=True)
        assert flow.verbose is True

        # Verify EventListener is also set to verbose=True
        event_listener = EventListener()
        assert event_listener.verbose is True
        assert event_listener.formatter.verbose is True

    def test_flow_respects_env_var_false(self):
        """Test that Flow respects CREWAI_VERBOSE=false env var."""

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        with patch.dict(os.environ, {"CREWAI_VERBOSE": "false"}, clear=False):
            flow = SimpleFlow()
            assert flow.verbose is False

    def test_flow_respects_env_var_true(self):
        """Test that Flow respects CREWAI_VERBOSE=true env var."""

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        with patch.dict(os.environ, {"CREWAI_VERBOSE": "true"}, clear=False):
            flow = SimpleFlow()
            assert flow.verbose is True

    def test_flow_explicit_verbose_overrides_env_var(self):
        """Test that explicit verbose parameter overrides env var."""

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                return "done"

        # Explicit verbose=True overrides CREWAI_VERBOSE=false
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "false"}, clear=False):
            flow = SimpleFlow(verbose=True)
            assert flow.verbose is True

        # Explicit verbose=False overrides CREWAI_VERBOSE=true
        with patch.dict(os.environ, {"CREWAI_VERBOSE": "true"}, clear=False):
            flow = SimpleFlow(verbose=False)
            assert flow.verbose is False


class TestFlowVerboseExecution:
    """Test that verbose setting actually suppresses output during Flow execution."""

    def test_flow_verbose_false_suppresses_console_output(self):
        """Test that Flow with verbose=False suppresses console output."""
        execution_order = []

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                execution_order.append("step_1")
                return "step_1_done"

            @listen(step_1)
            def step_2(self):
                execution_order.append("step_2")
                return "step_2_done"

        # Create flow with verbose=False
        flow = SimpleFlow(verbose=False)

        # Verify the formatter's verbose is False
        event_listener = EventListener()
        assert event_listener.formatter.verbose is False

        # Execute the flow
        result = flow.kickoff()

        # Flow should still execute correctly
        assert execution_order == ["step_1", "step_2"]
        assert result == "step_2_done"

    def test_flow_verbose_true_allows_console_output(self):
        """Test that Flow with verbose=True allows console output."""
        execution_order = []

        class SimpleFlow(Flow):
            @start()
            def step_1(self):
                execution_order.append("step_1")
                return "step_1_done"

            @listen(step_1)
            def step_2(self):
                execution_order.append("step_2")
                return "step_2_done"

        # Create flow with verbose=True
        flow = SimpleFlow(verbose=True)

        # Verify the formatter's verbose is True
        event_listener = EventListener()
        assert event_listener.formatter.verbose is True

        # Execute the flow
        result = flow.kickoff()

        # Flow should execute correctly
        assert execution_order == ["step_1", "step_2"]
        assert result == "step_2_done"


class TestConsoleFormatterVerbose:
    """Test that ConsoleFormatter respects verbose setting."""

    def test_console_formatter_print_panel_respects_verbose_false(self):
        """Test that print_panel does not print when verbose=False."""
        from rich.text import Text
        from crewai.events.utils.console_formatter import ConsoleFormatter

        formatter = ConsoleFormatter(verbose=False)

        # Create a mock to capture print calls
        with patch.object(formatter, "print") as mock_print:
            content = Text("Test content")
            formatter.print_panel(content, "Test Title", "blue", is_flow=True)

            # print should not be called when verbose=False
            mock_print.assert_not_called()

    def test_console_formatter_print_panel_respects_verbose_true(self):
        """Test that print_panel prints when verbose=True."""
        from rich.text import Text
        from crewai.events.utils.console_formatter import ConsoleFormatter

        formatter = ConsoleFormatter(verbose=True)

        # Create a mock to capture print calls
        with patch.object(formatter, "print") as mock_print:
            content = Text("Test content")
            formatter.print_panel(content, "Test Title", "blue", is_flow=True)

            # print should be called when verbose=True
            assert mock_print.call_count >= 1

    def test_console_formatter_flow_events_respect_verbose_false(self):
        """Test that flow events are suppressed when verbose=False."""
        from rich.text import Text
        from crewai.events.utils.console_formatter import ConsoleFormatter

        formatter = ConsoleFormatter(verbose=False)

        # Create a mock to capture print calls
        with patch.object(formatter, "print") as mock_print:
            content = Text("Flow event content")
            # is_flow=True should still respect verbose=False
            formatter.print_panel(content, "Flow Event", "blue", is_flow=True)

            # print should not be called even for flow events when verbose=False
            mock_print.assert_not_called()
