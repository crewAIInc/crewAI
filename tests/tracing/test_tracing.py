import os
import pytest
from unittest.mock import patch, MagicMock

from crewai import Agent, Task, Crew
from crewai.utilities.events.listeners.tracing.trace_listener import (
    TraceCollectionListener,
)
from crewai.utilities.events.listeners.tracing.trace_batch_manager import (
    TraceBatchManager,
)
from crewai.utilities.events.listeners.tracing.types import TraceEvent


class TestTraceListenerSetup:
    """Test TraceListener is properly setup and collecting events"""

    @pytest.fixture(autouse=True)
    def clear_event_bus(self):
        """Clear event bus listeners before each test"""
        from crewai.utilities.events import crewai_event_bus

        crewai_event_bus._handlers.clear()
        yield

    @pytest.fixture(autouse=True)
    def mock_plus_api_calls(self):
        """Mock all PlusAPI HTTP calls to avoid authentication and network requests"""
        with (
            patch(
                "crewai.cli.authentication.token.get_auth_token",
                return_value="mock_token",
            ),
            patch("requests.post") as mock_post,
            patch("requests.get") as mock_get,
            patch("requests.put") as mock_put,
            patch("requests.delete") as mock_delete,
        ):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "mock_trace_batch_id",
                "status": "success",
                "message": "Batch created successfully",
            }
            mock_response.raise_for_status.return_value = None

            mock_post.return_value = mock_response
            mock_get.return_value = mock_response
            mock_put.return_value = mock_response
            mock_delete.return_value = mock_response

            yield {
                "post": mock_post,
                "get": mock_get,
                "put": mock_put,
                "delete": mock_delete,
            }

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_trace_listener_collects_crew_events(self):
        """Test that trace listener properly collects events from crew execution"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello to the world",
                expected_output="hello world",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=True)

            original_initialize = TraceCollectionListener._initialize_batch

            with (
                patch.object(
                    TraceCollectionListener,
                    "_initialize_batch",
                    side_effect=original_initialize,
                ) as initialize_mock,
                patch.object(
                    TraceBatchManager,
                    "_finalize_backend_batch",
                    return_value=True,  # Mock the return but let it be called
                ) as finalize_mock,
            ):
                crew.kickoff()

                # The trace listener should have been called during crew execution
                initialize_mock.assert_called_once()
                finalize_mock.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_batch_manager_finalizes_batch_clears_buffer(self):
        """Test that batch manager properly finalizes batch and clears buffer"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )

            task = Task(
                description="Say hello to the world",
                expected_output="hello world",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task], verbose=True)

            with patch.object(
                TraceBatchManager, "_cleanup_batch_data", return_value=True
            ) as cleanup_mock:
                crew.kickoff()

                cleanup_mock.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_events_collection_batch_manager(self, mock_plus_api_calls):
        """Test that trace listener properly collects events from crew execution"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello to the world",
                expected_output="hello world",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=True)

            from crewai.utilities.events import crewai_event_bus

            trace_listener = None
            for handler_list in crewai_event_bus._handlers.values():
                for handler in handler_list:
                    if hasattr(handler, "__self__") and isinstance(
                        handler.__self__, TraceCollectionListener
                    ):
                        trace_listener = handler.__self__
                        break
                if trace_listener:
                    break

            if trace_listener and trace_listener.batch_manager:
                with patch.object(
                    trace_listener.batch_manager,
                    "add_event",
                    wraps=trace_listener.batch_manager.add_event,
                ) as add_event_mock:
                    crew.kickoff()

                    assert add_event_mock.call_count >= 2

                    completion_events = [
                        call.args[0]
                        for call in add_event_mock.call_args_list
                        if call.args[0].type == "crew_kickoff_completed"
                    ]
                    assert len(completion_events) == 1

                    completion_event = completion_events[0]
                    assert "crew_name" in completion_event.event_data
                    assert completion_event.event_data["crew_name"] == "crew"

                    for call in add_event_mock.call_args_list:
                        event = call.args[0]
                        assert isinstance(event, TraceEvent)
                        assert hasattr(event, "event_data")
                        assert hasattr(event, "type")
            else:
                test_trace_listener = TraceCollectionListener()
                test_trace_listener.setup_listeners(crewai_event_bus)

                with patch.object(
                    test_trace_listener.batch_manager, "add_event"
                ) as add_event_mock:
                    crew.kickoff()

                    assert add_event_mock.call_count >= 2

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_trace_listener_disabled_when_env_false(self):
        """Test that trace listener doesn't make HTTP calls when tracing is disabled"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "false"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello to the world",
                expected_output="hello world",
                agent=agent,
            )

            crew = Crew(agents=[agent], tasks=[task], verbose=True)
            result = crew.kickoff()
            assert result is not None

    def test_trace_listener_setup_correctly(self):
        """Test that trace listener is set up correctly when enabled"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
            trace_listener = TraceCollectionListener()

            assert trace_listener.trace_enabled is True

            assert trace_listener.batch_manager is not None

            assert trace_listener.trace_sender is not None
