import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from crewai import Agent, Crew, Task
from crewai.events.listeners.tracing.first_time_trace_handler import (
    FirstTimeTraceHandler,
)
from crewai.events.listeners.tracing.trace_batch_manager import (
    TraceBatchManager,
)
from crewai.events.listeners.tracing.trace_listener import (
    TraceCollectionListener,
)
from crewai.events.listeners.tracing.types import TraceEvent
from crewai.flow.flow import Flow, start


class TestTraceListenerSetup:
    """Test TraceListener is properly setup and collecting events"""

    @pytest.fixture(autouse=True)
    def mock_auth_token(self):
        """Mock authentication token for all tests in this class"""
        # Need to patch all the places where get_auth_token is imported/used
        with (
            patch(
                "crewai.cli.authentication.token.get_auth_token",
                return_value="mock_token_12345",
            ),
            patch(
                "crewai.events.listeners.tracing.trace_listener.get_auth_token",
                return_value="mock_token_12345",
            ),
            patch(
                "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token",
                return_value="mock_token_12345",
            ),
        ):
            yield

    @pytest.fixture(autouse=True)
    def clear_event_bus(self):
        """Clear event bus listeners before and after each test"""
        from crewai.events.event_bus import crewai_event_bus

        # Store original handlers
        original_handlers = crewai_event_bus._handlers.copy()

        # Clear for test
        crewai_event_bus._handlers.clear()

        yield

        # Restore original state
        crewai_event_bus._handlers.clear()
        crewai_event_bus._handlers.update(original_handlers)

    @pytest.fixture(autouse=True)
    def reset_tracing_singletons(self):
        """Reset tracing singleton instances between tests"""
        # Reset TraceCollectionListener singleton
        if hasattr(TraceCollectionListener, "_instance"):
            TraceCollectionListener._instance = None
            TraceCollectionListener._initialized = False

        yield

        # Clean up after test
        if hasattr(TraceCollectionListener, "_instance"):
            TraceCollectionListener._instance = None
            TraceCollectionListener._initialized = False

    @pytest.fixture(autouse=True)
    def mock_plus_api_calls(self):
        """Mock all PlusAPI HTTP calls to avoid network requests"""
        with (
            patch("requests.post") as mock_post,
            patch("requests.get") as mock_get,
            patch("requests.put") as mock_put,
            patch("requests.delete") as mock_delete,
            patch.object(TraceBatchManager, "_cleanup_batch_data", return_value=True),
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

            mock_mark_failed = MagicMock()
            mock_mark_failed.return_value = mock_response

            yield {
                "post": mock_post,
                "get": mock_get,
                "put": mock_put,
                "delete": mock_delete,
                "mark_trace_batch_as_failed": mock_mark_failed,
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

            trace_listener = TraceCollectionListener()
            from crewai.events.event_bus import crewai_event_bus

            trace_listener.setup_listeners(crewai_event_bus)

            with patch.object(
                trace_listener.batch_manager,
                "initialize_batch",
                return_value=None,
            ) as initialize_mock:
                crew.kickoff()

                assert initialize_mock.call_count >= 1

                call_args = initialize_mock.call_args_list[0]
                assert len(call_args[0]) == 2  # user_context, execution_metadata
                _, execution_metadata = call_args[0]
                assert isinstance(execution_metadata, dict)
                assert "crew_name" in execution_metadata

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

            from crewai.events.event_bus import crewai_event_bus

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

            if not trace_listener:
                pytest.skip(
                    "No trace listener found - tracing may not be properly enabled"
                )

            with patch.object(
                trace_listener.batch_manager,
                "finalize_batch",
                wraps=trace_listener.batch_manager.finalize_batch,
            ) as finalize_mock:
                crew.kickoff()

                assert finalize_mock.call_count >= 1

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

            from crewai.events.event_bus import crewai_event_bus

            # Create and setup trace listener explicitly
            trace_listener = TraceCollectionListener()
            trace_listener.setup_listeners(crewai_event_bus)

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
                assert len(completion_events) >= 1

                # Verify the first completion event has proper structure
                completion_event = completion_events[0]
                assert "crew_name" in completion_event.event_data
                assert completion_event.event_data["crew_name"] == "crew"

                # Verify all events have proper structure
                for call in add_event_mock.call_args_list:
                    event = call.args[0]
                    assert isinstance(event, TraceEvent)
                    assert hasattr(event, "event_data")
                    assert hasattr(event, "type")

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

            from crewai.events.event_bus import crewai_event_bus

            trace_handlers = []
            for handlers in crewai_event_bus._handlers.values():
                for handler in handlers:
                    if hasattr(handler, "__self__") and isinstance(
                        handler.__self__, TraceCollectionListener
                    ):
                        trace_handlers.append(handler)
                    elif hasattr(handler, "__name__") and any(
                        trace_name in handler.__name__
                        for trace_name in [
                            "on_crew_started",
                            "on_crew_completed",
                            "on_flow_started",
                        ]
                    ):
                        trace_handlers.append(handler)

            assert len(trace_handlers) == 0, (
                f"Found {len(trace_handlers)} trace handlers when tracing should be disabled"
            )

    def test_trace_listener_setup_correctly_for_crew(self):
        """Test that trace listener is set up correctly when enabled"""

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
            with patch.object(
                TraceCollectionListener, "setup_listeners"
            ) as mock_listener_setup:
                Crew(agents=[agent], tasks=[task], verbose=True)
                assert mock_listener_setup.call_count >= 1

    def test_trace_listener_setup_correctly_for_flow(self):
        """Test that trace listener is set up correctly when enabled"""

        with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):

            class FlowExample(Flow):
                @start()
                def start(self):
                    pass

            with patch.object(
                TraceCollectionListener, "setup_listeners"
            ) as mock_listener_setup:
                FlowExample()
                assert mock_listener_setup.call_count >= 1

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_trace_listener_ephemeral_batch(self):
        """Test that trace listener properly handles ephemeral batches"""
        with (
            patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}),
            patch(
                "crewai.events.listeners.tracing.trace_listener.TraceCollectionListener._check_authenticated",
                return_value=False,
            ),
        ):
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
            crew = Crew(agents=[agent], tasks=[task], tracing=True)

            with patch.object(TraceBatchManager, "initialize_batch") as mock_initialize:
                crew.kickoff()

                assert mock_initialize.call_count >= 1
                assert mock_initialize.call_args_list[0][1]["use_ephemeral"] is True

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_trace_listener_with_authenticated_user(self):
        """Test that trace listener properly handles authenticated batches"""
        with (
            patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}),
            patch(
                "crewai.events.listeners.tracing.trace_batch_manager.PlusAPI"
            ) as mock_plus_api_class,
        ):
            mock_plus_api_instance = MagicMock()
            mock_plus_api_class.return_value = mock_plus_api_instance

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

            with (
                patch.object(TraceBatchManager, "initialize_batch") as mock_initialize,
                patch.object(
                    TraceBatchManager, "finalize_batch"
                ) as mock_finalize_backend_batch,
            ):
                crew = Crew(agents=[agent], tasks=[task], tracing=True)
                crew.kickoff()

                mock_plus_api_class.assert_called_with(api_key="mock_token_12345")

                assert mock_initialize.call_count >= 1
                mock_finalize_backend_batch.assert_called_with()
                assert mock_finalize_backend_batch.call_count >= 1

    # Helper method to ensure cleanup
    def teardown_method(self):
        """Cleanup after each test method"""
        from crewai.events.event_bus import crewai_event_bus

        crewai_event_bus._handlers.clear()

    @classmethod
    def teardown_class(cls):
        """Final cleanup after all tests in this class"""
        from crewai.events.event_bus import crewai_event_bus

        crewai_event_bus._handlers.clear()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_first_time_user_trace_collection_with_timeout(self, mock_plus_api_calls):
        """Test first-time user trace collection logic with timeout behavior"""

        with (
            patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "false"}),
            patch(
                "crewai.events.listeners.tracing.utils._is_test_environment",
                return_value=False,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.is_first_execution",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.prompt_user_for_trace_viewing",
                return_value=False,
            ) as mock_prompt,
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.mark_first_execution_completed"
            ) as mock_mark_completed,
        ):
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

            from crewai.events.event_bus import crewai_event_bus

            trace_listener = TraceCollectionListener()
            trace_listener.setup_listeners(crewai_event_bus)

            assert trace_listener.first_time_handler.is_first_time is True
            assert trace_listener.first_time_handler.collected_events is False

            with (
                patch.object(
                    trace_listener.first_time_handler,
                    "handle_execution_completion",
                    wraps=trace_listener.first_time_handler.handle_execution_completion,
                ) as mock_handle_completion,
                patch.object(
                    trace_listener.batch_manager,
                    "add_event",
                    wraps=trace_listener.batch_manager.add_event,
                ) as mock_add_event,
            ):
                result = crew.kickoff()
                assert result is not None

                assert mock_handle_completion.call_count >= 1
                assert mock_add_event.call_count >= 1

                assert trace_listener.first_time_handler.collected_events is True

                mock_prompt.assert_called_once_with(timeout_seconds=20)

                mock_mark_completed.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_first_time_user_trace_collection_user_accepts(self, mock_plus_api_calls):
        """Test first-time user trace collection when user accepts viewing traces"""

        with (
            patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "false"}),
            patch(
                "crewai.events.listeners.tracing.utils._is_test_environment",
                return_value=False,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.is_first_execution",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.prompt_user_for_trace_viewing",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.mark_first_execution_completed"
            ) as mock_mark_completed,
        ):
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

            from crewai.events.event_bus import crewai_event_bus

            trace_listener = TraceCollectionListener()
            trace_listener.setup_listeners(crewai_event_bus)

            assert trace_listener.first_time_handler.is_first_time is True

            with (
                patch.object(
                    trace_listener.first_time_handler,
                    "_initialize_backend_and_send_events",
                    wraps=trace_listener.first_time_handler._initialize_backend_and_send_events,
                ) as mock_init_backend,
                patch.object(
                    trace_listener.first_time_handler, "_display_ephemeral_trace_link"
                ) as mock_display_link,
                patch.object(
                    trace_listener.first_time_handler,
                    "handle_execution_completion",
                    wraps=trace_listener.first_time_handler.handle_execution_completion,
                ) as mock_handle_completion,
            ):
                trace_listener.batch_manager.ephemeral_trace_url = (
                    "https://crewai.com/trace/mock-id"
                )

                crew.kickoff()

                assert mock_handle_completion.call_count >= 1, (
                    "handle_execution_completion should be called"
                )

                assert trace_listener.first_time_handler.collected_events is True, (
                    "Events should be marked as collected"
                )

                mock_init_backend.assert_called_once()

                mock_display_link.assert_called_once()

                mock_mark_completed.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_first_time_user_trace_consolidation_logic(self, mock_plus_api_calls):
        """Test the consolidation logic for first-time users vs regular tracing"""

        with (
            patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "false"}),
            patch(
                "crewai.events.listeners.tracing.utils._is_test_environment",
                return_value=False,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.utils.is_first_execution",
                return_value=True,
            ),
        ):
            from crewai.events.event_bus import crewai_event_bus

            crewai_event_bus._handlers.clear()

            trace_listener = TraceCollectionListener()
            trace_listener.setup_listeners(crewai_event_bus)

            assert trace_listener.first_time_handler.is_first_time is True

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Test task", expected_output="test output", agent=agent
            )
            crew = Crew(agents=[agent], tasks=[task])

            with patch.object(TraceBatchManager, "initialize_batch") as mock_initialize:
                result = crew.kickoff()

                assert mock_initialize.call_count >= 1
                assert mock_initialize.call_args_list[0][1]["use_ephemeral"] is True
                assert result is not None

    def test_first_time_handler_timeout_behavior(self):
        """Test the timeout behavior of the first-time trace prompt"""

        with (
            patch(
                "crewai.events.listeners.tracing.utils._is_test_environment",
                return_value=False,
            ),
            patch("threading.Thread") as mock_thread,
        ):
            from crewai.events.listeners.tracing.utils import (
                prompt_user_for_trace_viewing,
            )

            mock_thread_instance = Mock()
            mock_thread_instance.is_alive.return_value = True
            mock_thread.return_value = mock_thread_instance

            result = prompt_user_for_trace_viewing(timeout_seconds=5)

            assert result is False
            mock_thread.assert_called_once()
            call_args = mock_thread.call_args
            assert call_args[1]["daemon"] is True

            mock_thread_instance.start.assert_called_once()
            mock_thread_instance.join.assert_called_once_with(timeout=5)
            mock_thread_instance.is_alive.assert_called_once()

    def test_first_time_handler_graceful_error_handling(self):
        """Test graceful error handling in first-time trace logic"""

        with (
            patch(
                "crewai.events.listeners.tracing.utils.should_auto_collect_first_time_traces",
                return_value=True,
            ),
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.prompt_user_for_trace_viewing",
                side_effect=Exception("Prompt failed"),
            ),
            patch(
                "crewai.events.listeners.tracing.first_time_trace_handler.mark_first_execution_completed"
            ) as mock_mark_completed,
        ):
            handler = FirstTimeTraceHandler()
            handler.is_first_time = True
            handler.collected_events = True

            handler.handle_execution_completion()

            mock_mark_completed.assert_called_once()

    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_trace_batch_marked_as_failed_on_finalize_error(self, mock_plus_api_calls):
        """Test that trace batch is marked as failed when finalization returns non-200 status"""

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

            trace_listener = TraceCollectionListener()
            from crewai.events.event_bus import crewai_event_bus

            trace_listener.setup_listeners(crewai_event_bus)

            with (
                patch.object(
                    trace_listener.batch_manager.plus_api,
                    "send_trace_events",
                    return_value=MagicMock(status_code=200),
                ),
                patch.object(
                    trace_listener.batch_manager.plus_api,
                    "finalize_trace_batch",
                    return_value=MagicMock(
                        status_code=500, text="Internal Server Error"
                    ),
                ),
                patch.object(
                    trace_listener.batch_manager.plus_api,
                    "mark_trace_batch_as_failed",
                    wraps=mock_plus_api_calls["mark_trace_batch_as_failed"],
                ) as mock_mark_failed,
            ):
                crew.kickoff()

                mock_mark_failed.assert_called_once()
                call_args = mock_mark_failed.call_args_list[0]
                assert call_args[0][1] == "Error sending events to backend"
