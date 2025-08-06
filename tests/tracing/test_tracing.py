import os
import pytest
from unittest.mock import patch

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

    @pytest.mark.vcr(filter_headers=["authorization"])
    @patch(
        "crewai.cli.authentication.token.get_auth_token",
        return_value="mock_token_12345",
    )
    def test_trace_listener_collects_crew_events(self, mock_get_auth_token):
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

            with (
                patch.object(
                    TraceCollectionListener, "_initialize_batch"
                ) as initialize_mock,
                patch.object(
                    TraceBatchManager, "_finalize_backend_batch"
                ) as finalize_mock,
            ):
                finalize_mock.return_value = True

                crew.kickoff()

                initialize_mock.assert_called_once()
                finalize_mock.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    @patch(
        "crewai.cli.authentication.token.get_auth_token",
        return_value="mock_token_12345",
    )
    def test_batch_manager_finalizes_batch_clears_buffer(self, mock_get_auth_token):
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

            with patch.object(TraceBatchManager, "_cleanup_batch_data") as cleanup_mock:
                cleanup_mock.return_value = True

                crew.kickoff()

                cleanup_mock.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    @patch(
        "crewai.cli.authentication.token.get_auth_token",
        return_value="mock_token_12345",
    )
    def test_events_collection_batch_manager(self, mock_get_auth_token):
        """Test that trace listener properly collects events from crew execution"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test bastory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello to the world",
                expected_output="hello world",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=True)

            with patch.object(TraceBatchManager, "add_event") as add_event_mock:
                crew.kickoff()

                assert add_event_mock.call_count == 8

                last_call_args = add_event_mock.call_args[0][0]
                assert last_call_args.type == "crew_kickoff_completed"
                assert "crew_name" in last_call_args.event_data
                assert last_call_args.event_data["crew_name"] == "crew"

                for call in add_event_mock.call_args_list:
                    event = call[0][0]
                    assert isinstance(event, TraceEvent)
                    assert hasattr(event, "event_data")
                    assert hasattr(event, "type")
