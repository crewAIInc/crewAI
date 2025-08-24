from unittest.mock import Mock, patch
from crewai.utilities.events.third_party.agentops_listener import AgentOpsListener
from crewai.utilities.events.crew_events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
)
from crewai.utilities.events.task_events import TaskEvaluationEvent
from crewai.utilities.events.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageErrorEvent,
)
from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus


class TestAgentOpsListener:
    def test_agentops_listener_initialization_with_agentops_installed(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops"):
            listener = AgentOpsListener()
            assert listener.agentops is not None

    def test_agentops_listener_initialization_without_agentops_installed(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops", side_effect=ImportError):
            listener = AgentOpsListener()
            assert listener.agentops is None

    def test_setup_listeners_with_agentops_installed(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops"):
            listener = AgentOpsListener()
            mock_event_bus = Mock(spec=CrewAIEventsBus)
            
            listener.setup_listeners(mock_event_bus)
            
            assert mock_event_bus.register_handler.call_count == 5
            mock_event_bus.register_handler.assert_any_call(
                CrewKickoffStartedEvent, listener._handle_crew_kickoff_started
            )
            mock_event_bus.register_handler.assert_any_call(
                CrewKickoffCompletedEvent, listener._handle_crew_kickoff_completed
            )
            mock_event_bus.register_handler.assert_any_call(
                ToolUsageStartedEvent, listener._handle_tool_usage_started
            )
            mock_event_bus.register_handler.assert_any_call(
                ToolUsageErrorEvent, listener._handle_tool_usage_error
            )
            mock_event_bus.register_handler.assert_any_call(
                TaskEvaluationEvent, listener._handle_task_evaluation
            )

    def test_setup_listeners_without_agentops_installed(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops", side_effect=ImportError):
            listener = AgentOpsListener()
            mock_event_bus = Mock(spec=CrewAIEventsBus)
            
            listener.setup_listeners(mock_event_bus)
            
            mock_event_bus.register_handler.assert_not_called()

    def test_handle_crew_kickoff_started_with_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            listener = AgentOpsListener()
            event = CrewKickoffStartedEvent(crew_id="test-crew")
            
            listener._handle_crew_kickoff_started(event)
            
            mock_agentops.start_session.assert_called_once()
            call_args = mock_agentops.start_session.call_args
            assert call_args[1]["tags"] == ["crewai", "crew_kickoff"]

    def test_handle_crew_kickoff_started_without_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops", side_effect=ImportError):
            listener = AgentOpsListener()
            event = CrewKickoffStartedEvent(crew_id="test-crew")
            
            listener._handle_crew_kickoff_started(event)

    def test_handle_crew_kickoff_completed_with_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            listener = AgentOpsListener()
            event = CrewKickoffCompletedEvent(crew_id="test-crew", crew_output=Mock())
            
            listener._handle_crew_kickoff_completed(event)
            
            mock_agentops.end_session.assert_called_once_with("Success")

    def test_handle_crew_kickoff_completed_without_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops", side_effect=ImportError):
            listener = AgentOpsListener()
            event = CrewKickoffCompletedEvent(crew_id="test-crew", crew_output=Mock())
            
            listener._handle_crew_kickoff_completed(event)

    def test_handle_tool_usage_started_with_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            listener = AgentOpsListener()
            event = ToolUsageStartedEvent(
                tool_name="test_tool",
                arguments={"arg1": "value1"},
                agent_id="test-agent",
                task_id="test-task"
            )
            
            listener._handle_tool_usage_started(event)
            
            mock_agentops.record.assert_called_once()
            call_args = mock_agentops.record.call_args[0][0]
            assert hasattr(call_args, "action_type")

    def test_handle_tool_usage_error_with_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            listener = AgentOpsListener()
            event = ToolUsageErrorEvent(
                tool_name="test_tool",
                arguments={"arg1": "value1"},
                error="Test error",
                agent_id="test-agent",
                task_id="test-task"
            )
            
            listener._handle_tool_usage_error(event)
            
            mock_agentops.record.assert_called_once()

    def test_handle_task_evaluation_with_agentops(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            listener = AgentOpsListener()
            event = TaskEvaluationEvent(
                task_id="test-task",
                score=0.85,
                feedback="Good performance"
            )
            
            listener._handle_task_evaluation(event)
            
            mock_agentops.record.assert_called_once()

    def test_handle_crew_kickoff_started_with_exception(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            mock_agentops.start_session.side_effect = Exception("Test exception")
            listener = AgentOpsListener()
            event = CrewKickoffStartedEvent(crew_id="test-crew")
            
            listener._handle_crew_kickoff_started(event)

    def test_handle_crew_kickoff_completed_with_exception(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            mock_agentops.end_session.side_effect = Exception("Test exception")
            listener = AgentOpsListener()
            event = CrewKickoffCompletedEvent(crew_id="test-crew", crew_output=Mock())
            
            listener._handle_crew_kickoff_completed(event)

    def test_handle_tool_usage_started_with_exception(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            mock_agentops.record.side_effect = Exception("Test exception")
            listener = AgentOpsListener()
            event = ToolUsageStartedEvent(
                tool_name="test_tool",
                arguments={"arg1": "value1"},
                agent_id="test-agent",
                task_id="test-task"
            )
            
            listener._handle_tool_usage_started(event)

    def test_handle_tool_usage_error_with_exception(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            mock_agentops.record.side_effect = Exception("Test exception")
            listener = AgentOpsListener()
            event = ToolUsageErrorEvent(
                tool_name="test_tool",
                arguments={"arg1": "value1"},
                error="Test error",
                agent_id="test-agent",
                task_id="test-task"
            )
            
            listener._handle_tool_usage_error(event)

    def test_handle_task_evaluation_with_exception(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops") as mock_agentops:
            mock_agentops.record.side_effect = Exception("Test exception")
            listener = AgentOpsListener()
            event = TaskEvaluationEvent(
                task_id="test-task",
                score=0.85,
                feedback="Good performance"
            )
            
            listener._handle_task_evaluation(event)

    def test_agentops_listener_instance_creation(self):
        with patch("crewai.utilities.events.third_party.agentops_listener.agentops"):
            from crewai.utilities.events.third_party.agentops_listener import agentops_listener
            assert agentops_listener is not None
            assert isinstance(agentops_listener, AgentOpsListener)
