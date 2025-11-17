"""Test Flow verbose output control functionality."""

import logging

import pytest

from crewai.events.utils.console_formatter import ConsoleFormatter
from crewai.flow.flow import Flow, listen, start


def test_flow_verbose_defaults_to_false():
    """Test that Flow verbose defaults to False."""
    class SimpleFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    flow = SimpleFlow()
    assert flow.verbose is False


def test_flow_verbose_can_be_set_to_true():
    """Test that Flow verbose can be explicitly set to True."""
    class SimpleFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    flow = SimpleFlow(verbose=True)
    assert flow.verbose is True


def test_flow_verbose_can_be_set_to_false():
    """Test that Flow verbose can be explicitly set to False."""
    class SimpleFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    flow = SimpleFlow(verbose=False)
    assert flow.verbose is False


def test_flow_log_flow_event_respects_verbose_false(caplog):
    """Test that _log_flow_event does not print/log when verbose=False."""
    caplog.set_level(logging.INFO)
    
    class QuietFlow(Flow):
        @start()
        def step_1(self):
            self._log_flow_event("This should not appear")
            return "result"

    flow = QuietFlow(verbose=False)
    flow.kickoff()
    
    # Verify no logs were written
    assert "This should not appear" not in caplog.text


def test_flow_log_flow_event_respects_verbose_true(caplog):
    """Test that _log_flow_event prints/logs when verbose=True."""
    caplog.set_level(logging.INFO)
    
    class VerboseFlow(Flow):
        @start()
        def step_1(self):
            self._log_flow_event("This should appear", level="info")
            return "result"

    flow = VerboseFlow(verbose=True)
    flow.kickoff()
    
    # Verify logs were written
    assert "This should appear" in caplog.text


def test_flow_verbose_independent_of_tracing():
    """Test that Flow verbose is independent of tracing parameter."""
    class TestFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    flow_with_tracing = TestFlow(verbose=False, tracing=True)
    assert flow_with_tracing.verbose is False
    assert flow_with_tracing.tracing is True

    flow_without_tracing = TestFlow(verbose=True, tracing=False)
    assert flow_without_tracing.verbose is True
    assert flow_without_tracing.tracing is False


def test_flow_verbose_independent_of_persistence():
    """Test that Flow verbose is independent of persistence parameter."""
    class TestFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    flow_with_persistence = TestFlow(verbose=False, persistence=None)
    assert flow_with_persistence.verbose is False

    flow_without_persistence = TestFlow(verbose=True, persistence=None)
    assert flow_without_persistence.verbose is True


def test_flow_console_formatter_respects_verbose_when_is_flow_true(capsys):
    """Test that ConsoleFormatter respects verbose when is_flow=True."""
    formatter = ConsoleFormatter(verbose=False)
    
    from rich.text import Text
    content = Text("Test content")
    
    # When verbose=False, even is_flow=True should not print
    formatter.print_panel(content, "Test Title", "blue", is_flow=True)
    
    captured = capsys.readouterr()
    # When verbose=False, nothing should be printed even if is_flow=True
    # Rich outputs escape sequences, but the content should not appear
    assert len(captured.out.strip()) == 0


def test_flow_console_formatter_prints_when_verbose_true(capsys):
    """Test that ConsoleFormatter prints when verbose=True and is_flow=True."""
    formatter = ConsoleFormatter(verbose=True)
    
    from rich.text import Text
    content = Text("Test content")
    
    # When verbose=True, is_flow=True should print
    formatter.print_panel(content, "Test Title", "blue", is_flow=True)
    
    captured = capsys.readouterr()
    # When verbose=True, output should be printed
    # Rich will output escape codes, so we check that something was output
    assert len(captured.out) > 0


def test_flow_verbose_with_multiple_methods(caplog):
    """Test that verbose flag affects all flow methods."""
    caplog.set_level(logging.INFO)
    
    class MultiMethodFlow(Flow):
        @start()
        def step_1(self):
            self._log_flow_event("Step 1 message")
            return "result1"

        @listen(step_1)
        def step_2(self):
            self._log_flow_event("Step 2 message")
            return "result2"

    # Test with verbose=False - should produce no logs
    quiet_flow = MultiMethodFlow(verbose=False)
    quiet_flow.kickoff()
    
    # Clear the log before next test
    caplog.clear()
    
    # Test with verbose=True - should produce logs
    verbose_flow = MultiMethodFlow(verbose=True)
    verbose_flow.kickoff()
    
    # Verify logs were written when verbose=True
    assert "Step 1 message" in caplog.text
    assert "Step 2 message" in caplog.text


def test_flow_verbose_output_parameter_consistency():
    """Test that Flow verbose parameter behaves consistently with Crew verbose."""
    class TestFlow(Flow):
        @start()
        def step_1(self):
            return "result"

    # Test default (False)
    flow_default = TestFlow()
    assert flow_default.verbose is False

    # Test explicit True
    flow_verbose = TestFlow(verbose=True)
    assert flow_verbose.verbose is True
    
    # Test explicit False
    flow_quiet = TestFlow(verbose=False)
    assert flow_quiet.verbose is False
    
    # Verify flow still executes regardless of verbose setting
    result = flow_quiet.kickoff()
    assert result is not None


