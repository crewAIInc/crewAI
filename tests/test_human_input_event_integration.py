import pytest
import uuid
from unittest.mock import patch, MagicMock

from crewai.utilities.events.task_events import HumanInputRequiredEvent, HumanInputCompletedEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin


class TestHumanInputEventIntegration:
    """Test integration between human input flow and event system"""

    def setup_method(self):
        """Setup test environment"""
        self.executor = CrewAgentExecutorMixin()
        self.executor.crew = MagicMock()
        self.executor.crew.id = str(uuid.uuid4())
        self.executor.crew._train = False
        self.executor.task = MagicMock()
        self.executor.task.id = str(uuid.uuid4())
        self.executor.agent = MagicMock()
        self.executor.agent.id = str(uuid.uuid4())
        self.executor._printer = MagicMock()

    @patch('builtins.input', return_value='test feedback')
    def test_human_input_emits_required_event(self, mock_input):
        """Test that human input emits HumanInputRequiredEvent"""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event):
            result = self.executor._ask_human_input("Test result")
            
            assert result == 'test feedback'
            assert len(events_captured) == 2
            
            required_event = events_captured[0][1]
            completed_event = events_captured[1][1]
            
            assert isinstance(required_event, HumanInputRequiredEvent)
            assert isinstance(completed_event, HumanInputCompletedEvent)
            
            assert required_event.execution_id == str(self.executor.crew.id)
            assert required_event.crew_id == str(self.executor.crew.id)
            assert required_event.task_id == str(self.executor.task.id)
            assert required_event.agent_id == str(self.executor.agent.id)
            assert "HUMAN FEEDBACK" in required_event.prompt
            assert required_event.context == "Test result"
            assert required_event.event_id is not None
            
            assert completed_event.execution_id == str(self.executor.crew.id)
            assert completed_event.human_feedback == 'test feedback'
            assert completed_event.event_id == required_event.event_id

    @patch('builtins.input', return_value='training feedback')
    def test_training_mode_human_input_events(self, mock_input):
        """Test human input events in training mode"""
        self.executor.crew._train = True
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event):
            result = self.executor._ask_human_input("Test result")
            
            assert result == 'training feedback'
            assert len(events_captured) == 2
            
            required_event = events_captured[0][1]
            assert isinstance(required_event, HumanInputRequiredEvent)
            assert "TRAINING MODE" in required_event.prompt

    @patch('builtins.input', return_value='')
    def test_empty_feedback_events(self, mock_input):
        """Test events with empty feedback"""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event):
            result = self.executor._ask_human_input("Test result")
            
            assert result == ''
            assert len(events_captured) == 2
            
            completed_event = events_captured[1][1]
            assert isinstance(completed_event, HumanInputCompletedEvent)
            assert completed_event.human_feedback == ''

    def test_event_payload_structure(self):
        """Test that event payload matches GitHub issue specification"""
        event = HumanInputRequiredEvent(
            execution_id="test-execution-id",
            crew_id="test-crew-id",
            task_id="test-task-id",
            agent_id="test-agent-id",
            prompt="Test prompt",
            context="Test context",
            reason_flags={"ambiguity": True, "missing_field": False},
            event_id="test-event-id"
        )
        
        payload = event.to_json()
        
        assert payload["type"] == "human_input_required"
        assert payload["execution_id"] == "test-execution-id"
        assert payload["crew_id"] == "test-crew-id"
        assert payload["task_id"] == "test-task-id"
        assert payload["agent_id"] == "test-agent-id"
        assert payload["prompt"] == "Test prompt"
        assert payload["context"] == "Test context"
        assert payload["reason_flags"]["ambiguity"] is True
        assert payload["reason_flags"]["missing_field"] is False
        assert payload["event_id"] == "test-event-id"
        assert "timestamp" in payload

    def test_completed_event_payload_structure(self):
        """Test that completed event payload is correct"""
        event = HumanInputCompletedEvent(
            execution_id="test-execution-id",
            crew_id="test-crew-id",
            task_id="test-task-id",
            agent_id="test-agent-id",
            event_id="test-event-id",
            human_feedback="User feedback"
        )
        
        payload = event.to_json()
        
        assert payload["type"] == "human_input_completed"
        assert payload["execution_id"] == "test-execution-id"
        assert payload["crew_id"] == "test-crew-id"
        assert payload["task_id"] == "test-task-id"
        assert payload["agent_id"] == "test-agent-id"
        assert payload["event_id"] == "test-event-id"
        assert payload["human_feedback"] == "User feedback"
        assert "timestamp" in payload

    @patch('builtins.input', side_effect=KeyboardInterrupt("Test interrupt"))
    def test_human_input_exception_handling(self, mock_input):
        """Test that events are still emitted even if input is interrupted"""
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event):
            with pytest.raises(KeyboardInterrupt):
                self.executor._ask_human_input("Test result")
            
            assert len(events_captured) == 1
            required_event = events_captured[0][1]
            assert isinstance(required_event, HumanInputRequiredEvent)

    def test_human_input_without_crew(self):
        """Test human input events when crew is None"""
        self.executor.crew = None
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event), \
             patch('builtins.input', return_value='test'):
            
            result = self.executor._ask_human_input("Test result")
            
            assert len(events_captured) == 2
            required_event = events_captured[0][1]
            assert required_event.execution_id is None
            assert required_event.crew_id is None

    def test_human_input_without_task(self):
        """Test human input events when task is None"""
        self.executor.task = None
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event), \
             patch('builtins.input', return_value='test'):
            
            result = self.executor._ask_human_input("Test result")
            
            assert len(events_captured) == 2
            required_event = events_captured[0][1]
            assert required_event.task_id is None

    def test_human_input_without_agent(self):
        """Test human input events when agent is None"""
        self.executor.agent = None
        events_captured = []
        
        def capture_event(event):
            events_captured.append(event)
        
        with patch.object(crewai_event_bus, 'emit', side_effect=capture_event), \
             patch('builtins.input', return_value='test'):
            
            result = self.executor._ask_human_input("Test result")
            
            assert len(events_captured) == 2
            required_event = events_captured[0][1]
            assert required_event.agent_id is None
