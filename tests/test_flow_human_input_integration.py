import pytest
from unittest.mock import patch, MagicMock
from crewai.utilities.events.event_listener import event_listener


class TestFlowHumanInputIntegration:
    """Test integration between Flow execution and human input functionality."""

    def test_console_formatter_pause_resume_methods(self):
        """Test that ConsoleFormatter pause/resume methods work correctly."""
        formatter = event_listener.formatter
        
        original_paused_state = formatter._live_paused
        
        try:
            formatter._live_paused = False
            
            formatter.pause_live_updates()
            assert formatter._live_paused
            
            formatter.resume_live_updates()
            assert not formatter._live_paused
        finally:
            formatter._live_paused = original_paused_state

    @patch('builtins.input', return_value='')
    def test_human_input_pauses_flow_updates(self, mock_input):
        """Test that human input pauses Flow status updates."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
        
        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()
        
        formatter = event_listener.formatter
        
        original_paused_state = formatter._live_paused
        
        try:
            formatter._live_paused = False
            
            with patch.object(formatter, 'pause_live_updates') as mock_pause, \
                 patch.object(formatter, 'resume_live_updates') as mock_resume:
                
                result = executor._ask_human_input("Test result")
                
                mock_pause.assert_called_once()
                mock_resume.assert_called_once()
                mock_input.assert_called_once()
                assert result == ''
        finally:
            formatter._live_paused = original_paused_state

    @patch('builtins.input', side_effect=['feedback', ''])
    def test_multiple_human_input_rounds(self, mock_input):
        """Test multiple rounds of human input with Flow status management."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
        
        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()
        
        formatter = event_listener.formatter
        
        original_paused_state = formatter._live_paused
        
        try:
            pause_calls = []
            resume_calls = []
            
            def track_pause():
                pause_calls.append(True)
                
            def track_resume():
                resume_calls.append(True)
            
            with patch.object(formatter, 'pause_live_updates', side_effect=track_pause), \
                 patch.object(formatter, 'resume_live_updates', side_effect=track_resume):
                
                result1 = executor._ask_human_input("Test result 1")
                assert result1 == 'feedback'
                
                result2 = executor._ask_human_input("Test result 2")
                assert result2 == ''
                
                assert len(pause_calls) == 2
                assert len(resume_calls) == 2
        finally:
            formatter._live_paused = original_paused_state

    def test_pause_resume_with_no_live_session(self):
        """Test pause/resume methods handle case when no Live session exists."""
        formatter = event_listener.formatter
        
        original_live = formatter._live
        original_paused_state = formatter._live_paused
        
        try:
            formatter._live = None
            formatter._live_paused = False
            
            formatter.pause_live_updates()
            formatter.resume_live_updates()
            
            assert not formatter._live_paused
        finally:
            formatter._live = original_live
            formatter._live_paused = original_paused_state

    def test_pause_resume_exception_handling(self):
        """Test that resume is called even if exception occurs during human input."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
        
        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = False
        executor._printer = MagicMock()
        
        formatter = event_listener.formatter
        
        original_paused_state = formatter._live_paused
        
        try:
            with patch.object(formatter, 'pause_live_updates') as mock_pause, \
                 patch.object(formatter, 'resume_live_updates') as mock_resume, \
                 patch('builtins.input', side_effect=KeyboardInterrupt("Test exception")):
                
                with pytest.raises(KeyboardInterrupt):
                    executor._ask_human_input("Test result")
                
                mock_pause.assert_called_once()
                mock_resume.assert_called_once()
        finally:
            formatter._live_paused = original_paused_state

    def test_training_mode_human_input(self):
        """Test human input in training mode."""
        from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
        
        executor = CrewAgentExecutorMixin()
        executor.crew = MagicMock()
        executor.crew._train = True
        executor._printer = MagicMock()
        
        formatter = event_listener.formatter
        
        original_paused_state = formatter._live_paused
        
        try:
            with patch.object(formatter, 'pause_live_updates') as mock_pause, \
                 patch.object(formatter, 'resume_live_updates') as mock_resume, \
                 patch('builtins.input', return_value='training feedback'):
                
                result = executor._ask_human_input("Test result")
                
                mock_pause.assert_called_once()
                mock_resume.assert_called_once()
                assert result == 'training feedback'
                
                executor._printer.print.assert_called()
                call_args = [call[1]['content'] for call in executor._printer.print.call_args_list]
                training_prompt_found = any('TRAINING MODE' in content for content in call_args)
                assert training_prompt_found
        finally:
            formatter._live_paused = original_paused_state
