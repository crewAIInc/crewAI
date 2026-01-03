from unittest.mock import MagicMock, patch

import pytest
from crewai.events.event_listener import event_listener


class TestFlowHumanInputIntegration:
    """Test integration between Flow execution and human input functionality."""

    def test_console_formatter_pause_resume_methods_exist(self):
        """Test that ConsoleFormatter pause/resume methods exist and are callable."""
        formatter = event_listener.formatter

        # Methods should exist and be callable
        assert hasattr(formatter, "pause_live_updates")
        assert hasattr(formatter, "resume_live_updates")
        assert callable(formatter.pause_live_updates)
        assert callable(formatter.resume_live_updates)

        # Should not raise
        formatter.pause_live_updates()
        formatter.resume_live_updates()

    @patch("builtins.input", return_value="")
    def test_human_input_pauses_flow_updates(self, mock_input):
        """Test that human input pauses Flow status updates."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import (
            CrewAgentExecutorMixin,
        )

        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()

        formatter = event_listener.formatter

        with (
            patch.object(formatter, "pause_live_updates") as mock_pause,
            patch.object(formatter, "resume_live_updates") as mock_resume,
        ):
            result = executor._ask_human_input("Test result")

            mock_pause.assert_called_once()
            mock_resume.assert_called_once()
            mock_input.assert_called_once()
            assert result == ""

    @patch("builtins.input", side_effect=["feedback", ""])
    def test_multiple_human_input_rounds(self, mock_input):
        """Test multiple rounds of human input with Flow status management."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import (
            CrewAgentExecutorMixin,
        )

        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()

        formatter = event_listener.formatter

        pause_calls = []
        resume_calls = []

        def track_pause():
            pause_calls.append(True)

        def track_resume():
            resume_calls.append(True)

        with (
            patch.object(formatter, "pause_live_updates", side_effect=track_pause),
            patch.object(
                formatter, "resume_live_updates", side_effect=track_resume
            ),
        ):
            result1 = executor._ask_human_input("Test result 1")
            assert result1 == "feedback"

            result2 = executor._ask_human_input("Test result 2")
            assert result2 == ""

            assert len(pause_calls) == 2
            assert len(resume_calls) == 2

    def test_pause_resume_with_no_live_session(self):
        """Test pause/resume methods handle case when no Live session exists."""
        formatter = event_listener.formatter

        original_streaming_live = formatter._streaming_live

        try:
            formatter._streaming_live = None

            # Should not raise when no session exists
            formatter.pause_live_updates()
            formatter.resume_live_updates()

            assert formatter._streaming_live is None
        finally:
            formatter._streaming_live = original_streaming_live

    def test_pause_resume_exception_handling(self):
        """Test that resume is called even if exception occurs during human input."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import (
            CrewAgentExecutorMixin,
        )

        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()

        formatter = event_listener.formatter

        with (
            patch.object(formatter, "pause_live_updates") as mock_pause,
            patch.object(formatter, "resume_live_updates") as mock_resume,
            patch(
                "builtins.input", side_effect=KeyboardInterrupt("Test exception")
            ),
        ):
            with pytest.raises(KeyboardInterrupt):
                executor._ask_human_input("Test result")

            mock_pause.assert_called_once()
            mock_resume.assert_called_once()

    def test_training_mode_human_input(self):
        """Test human input in training mode."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import (
            CrewAgentExecutorMixin,
        )

        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = True
        executor._printer = MagicMock()

        formatter = event_listener.formatter

        with (
            patch.object(formatter, "pause_live_updates") as mock_pause,
            patch.object(formatter, "resume_live_updates") as mock_resume,
            patch.object(formatter.console, "print") as mock_console_print,
            patch("builtins.input", return_value="training feedback"),
        ):
            result = executor._ask_human_input("Test result")

            mock_pause.assert_called_once()
            mock_resume.assert_called_once()
            assert result == "training feedback"

            # Verify the training panel was printed via formatter's console
            mock_console_print.assert_called()
            # Check that a Panel with training title was printed
            call_args = mock_console_print.call_args_list
            training_panel_found = any(
                hasattr(call[0][0], "title") and "Training" in str(call[0][0].title)
                for call in call_args
                if call[0]
            )
            assert training_panel_found
