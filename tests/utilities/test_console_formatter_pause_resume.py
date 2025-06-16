from unittest.mock import MagicMock, patch
from rich.tree import Tree
from rich.live import Live
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterPauseResume:
    """Test ConsoleFormatter pause/resume functionality."""

    def test_pause_live_updates_with_active_session(self):
        """Test pausing when Live session is active."""
        formatter = ConsoleFormatter()
        
        mock_live = MagicMock(spec=Live)
        formatter._live = mock_live
        formatter._live_paused = False
        
        formatter.pause_live_updates()
        
        mock_live.stop.assert_called_once()
        assert formatter._live_paused

    def test_pause_live_updates_when_already_paused(self):
        """Test pausing when already paused does nothing."""
        formatter = ConsoleFormatter()
        
        mock_live = MagicMock(spec=Live)
        formatter._live = mock_live
        formatter._live_paused = True
        
        formatter.pause_live_updates()
        
        mock_live.stop.assert_not_called()
        assert formatter._live_paused

    def test_pause_live_updates_with_no_session(self):
        """Test pausing when no Live session exists."""
        formatter = ConsoleFormatter()
        
        formatter._live = None
        formatter._live_paused = False
        
        formatter.pause_live_updates()
        
        assert formatter._live_paused

    def test_resume_live_updates_when_paused(self):
        """Test resuming when paused."""
        formatter = ConsoleFormatter()
        
        formatter._live_paused = True
        
        formatter.resume_live_updates()
        
        assert not formatter._live_paused

    def test_resume_live_updates_when_not_paused(self):
        """Test resuming when not paused does nothing."""
        formatter = ConsoleFormatter()
        
        formatter._live_paused = False
        
        formatter.resume_live_updates()
        
        assert not formatter._live_paused

    def test_print_after_resume_restarts_live_session(self):
        """Test that printing a Tree after resume creates new Live session."""
        formatter = ConsoleFormatter()
        
        formatter._live_paused = True
        formatter._live = None
        
        formatter.resume_live_updates()
        assert not formatter._live_paused
        
        tree = Tree("Test")
        
        with patch('crewai.utilities.events.utils.console_formatter.Live') as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance
            
            formatter.print(tree)
            
            mock_live_class.assert_called_once()
            mock_live_instance.start.assert_called_once()
            assert formatter._live == mock_live_instance

    def test_multiple_pause_resume_cycles(self):
        """Test multiple pause/resume cycles work correctly."""
        formatter = ConsoleFormatter()
        
        mock_live = MagicMock(spec=Live)
        formatter._live = mock_live
        formatter._live_paused = False
        
        formatter.pause_live_updates()
        assert formatter._live_paused
        mock_live.stop.assert_called_once()
        assert formatter._live is None  # Live session should be cleared
        
        formatter.resume_live_updates()
        assert not formatter._live_paused
        
        formatter.pause_live_updates()
        assert formatter._live_paused
        
        formatter.resume_live_updates()
        assert not formatter._live_paused

    def test_pause_resume_state_initialization(self):
        """Test that _live_paused is properly initialized."""
        formatter = ConsoleFormatter()
        
        assert hasattr(formatter, '_live_paused')
        assert not formatter._live_paused
