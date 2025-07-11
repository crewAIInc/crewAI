import logging
from io import StringIO
from unittest.mock import MagicMock, patch
from rich.logging import RichHandler
from rich.tree import Tree
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from crewai.utilities.events.event_listener import EventListener


class TestRichLiveCleanup:
    """Test that Rich Live sessions are properly cleaned up after CrewAI operations."""

    def test_logging_works_after_tree_rendering(self):
        """Test that logging output appears after tree rendering with proper cleanup."""
        formatter = ConsoleFormatter()
        
        tree = Tree("Test Flow")
        formatter.print(tree)
        
        assert formatter._live is not None
        
        formatter.stop_live()
        
        assert formatter._live is None
        
        with patch.object(formatter.console, 'print') as mock_print:
            formatter.print("This should appear immediately")
            mock_print.assert_called_once_with("This should appear immediately")

    def test_event_listener_cleanup_integration(self):
        """Test that EventListener properly cleans up Live sessions."""
        event_listener = EventListener()
        formatter = event_listener.formatter
        
        tree = Tree("Test Crew")
        formatter.print(tree)
        assert formatter._live is not None
        
        formatter.stop_live()
        assert formatter._live is None

    def test_stop_live_restores_normal_output(self):
        """Test that stop_live properly restores normal console output behavior."""
        formatter = ConsoleFormatter()
        
        tree = Tree("Test Tree")
        formatter.print(tree)
        
        assert formatter._live is not None
        
        formatter.stop_live()
        
        assert formatter._live is None
        
        with patch.object(formatter.console, 'print') as mock_print:
            formatter.print("Normal output")
            mock_print.assert_called_once_with("Normal output")
