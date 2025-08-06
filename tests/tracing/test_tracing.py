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


class TestTraceListenerSetup:
    """Test TraceListener is properly setup and collecting events"""

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
