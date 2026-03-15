from unittest.mock import MagicMock, patch
from rich.live import Live
from crewai.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterPauseResume:
    """Test ConsoleFormatter pause/resume functionality for HITL features."""

    def test_pause_stops_active_streaming_session(self):
        """Test pausing stops an active streaming Live session."""
        formatter = ConsoleFormatter()

        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        formatter.pause_live_updates()

        mock_live.stop.assert_called_once()
        assert formatter._streaming_live is None

    def test_pause_is_safe_when_no_session(self):
        """Test pausing when no streaming session exists doesn't error."""
        formatter = ConsoleFormatter()
        formatter._streaming_live = None

        # Should not raise
        formatter.pause_live_updates()

        assert formatter._streaming_live is None

    def test_multiple_pauses_are_safe(self):
        """Test calling pause multiple times is safe."""
        formatter = ConsoleFormatter()

        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        formatter.pause_live_updates()
        mock_live.stop.assert_called_once()
        assert formatter._streaming_live is None

        # Second pause should not error (no session to stop)
        formatter.pause_live_updates()

    def test_resume_is_safe(self):
        """Test resume method exists and doesn't error."""
        formatter = ConsoleFormatter()

        # Should not raise
        formatter.resume_live_updates()

    def test_streaming_after_pause_resume_creates_new_session(self):
        """Test that streaming after pause/resume creates new Live session."""
        formatter = ConsoleFormatter()
        formatter.verbose = True

        # Simulate having an active session
        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        # Pause stops the session
        formatter.pause_live_updates()
        assert formatter._streaming_live is None

        # Resume (no-op, sessions created on demand)
        formatter.resume_live_updates()

        # After resume, streaming should be able to start a new session
        with patch("crewai.events.utils.console_formatter.Live") as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance

            # Simulate streaming chunk (this creates a new Live session)
            formatter.handle_llm_stream_chunk("test chunk", call_type=None)

            mock_live_class.assert_called_once()
            mock_live_instance.start.assert_called_once()
            assert formatter._streaming_live == mock_live_instance

    def test_pause_resume_cycle_with_streaming(self):
        """Test full pause/resume cycle during streaming."""
        formatter = ConsoleFormatter()
        formatter.verbose = True

        with patch("crewai.events.utils.console_formatter.Live") as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance

            # Start streaming
            formatter.handle_llm_stream_chunk("chunk 1", call_type=None)
            assert formatter._streaming_live == mock_live_instance

            # Pause should stop the session
            formatter.pause_live_updates()
            mock_live_instance.stop.assert_called_once()
            assert formatter._streaming_live is None

            # Resume (no-op)
            formatter.resume_live_updates()

            # Create a new mock for the next session
            mock_live_instance_2 = MagicMock()
            mock_live_class.return_value = mock_live_instance_2

            # Streaming again creates new session
            formatter.handle_llm_stream_chunk("chunk 2", call_type=None)
            assert formatter._streaming_live == mock_live_instance_2
