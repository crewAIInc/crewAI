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
from tests.utils import wait_for_event_handlers


class TestTraceListenerSetup:
    """Test TraceListener is properly setup and collecting events"""

    @pytest.fixture(autouse=True)
    def mock_user_data_file_io(self):
        """Mock user data file I/O to prevent file system pollution between tests"""
        with (
            patch(
                "crewai.events.listeners.tracing.utils._load_user_data",
                return_value={},
            ),
            patch(
                "crewai.events.listeners.tracing.utils._save_user_data",
                return_value=None,
            ),
        ):
            yield

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
    def reset_tracing_singletons(self):
        """Reset tracing singleton instances between tests"""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_listener import EventListener
        from crewai.events.listeners.tracing.utils import _tracing_enabled

        # Reset the tracing enabled contextvar
        try:
            _tracing_enabled.set(None)
        except (LookupError, AttributeError):
            pass

        # Clear event bus handlers BEFORE creating any new singletons
        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers = {}
            crewai_event_bus._async_handlers = {}
            crewai_event_bus._handler_dependencies = {}
            crewai_event_bus._execution_plan_cache = {}

        # Reset TraceCollectionListener singleton - must reset instance attributes too
        if TraceCollectionListener._instance is not None:
            # Reset instance attributes that shadow class attributes (only if they exist as instance attrs)
            instance_dict = TraceCollectionListener._instance.__dict__
            if "_initialized" in instance_dict:
                del TraceCollectionListener._instance._initialized
            if "_listeners_setup" in instance_dict:
                del TraceCollectionListener._instance._listeners_setup

        # Reset class attributes
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False

        # Reset EventListener singleton
        if hasattr(EventListener, "_instance"):
            EventListener._instance = None

        yield

        # Clean up after test
        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers = {}
            crewai_event_bus._async_handlers = {}
            crewai_event_bus._handler_dependencies = {}
            crewai_event_bus._execution_plan_cache = {}

        # Reset TraceCollectionListener singleton - must reset instance attributes too
        if TraceCollectionListener._instance is not None:
            # Reset instance attributes that shadow class attributes (only if they exist as instance attrs)
            instance_dict = TraceCollectionListener._instance.__dict__
            if "_initialized" in instance_dict:
                del TraceCollectionListener._instance._initialized
            if "_listeners_setup" in instance_dict:
                del TraceCollectionListener._instance._listeners_setup

        # Reset class attributes
        TraceCollectionListener._instance = None
        TraceCollectionListener._initialized = False
        TraceCollectionListener._listeners_setup = False

        if hasattr(EventListener, "_instance"):
            EventListener._instance = None

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

    @pytest.mark.vcr()
    def test_trace_listener_collects_crew_events(self):
        """Test that trace listener properly collects events from crew execution"""

        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
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

            from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
            trace_listener = TraceCollectionListener()

            crew.kickoff()

            initialized = trace_listener.batch_manager.wait_for_batch_initialization(timeout=5.0)

            assert initialized, "Batch should have been initialized"
            assert trace_listener.batch_manager.is_batch_initialized()
            assert trace_listener.batch_manager.current_batch is not None

    @pytest.mark.vcr()
    def test_batch_manager_finalizes_batch_clears_buffer(self):
        """Test that batch manager properly finalizes batch and clears buffer"""

        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
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

            trace_listener = None
            with crewai_event_bus._rwlock.r_locked():
                for handler_set in crewai_event_bus._sync_handlers.values():
                    for handler in handler_set:
                        if hasattr(handler, "__self__") and isinstance(
                            handler.__self__, TraceCollectionListener
                        ):
                            trace_listener = handler.__self__
                            break
                    if trace_listener:
                        break
                if not trace_listener:
                    for handler_set in crewai_event_bus._async_handlers.values():
                        for handler in handler_set:
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

    @pytest.mark.vcr()
    def test_events_collection_batch_manager(self, mock_plus_api_calls):
        """Test that trace listener properly collects events from crew execution"""

        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
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

            # Create and setup trace listener explicitly
            trace_listener = TraceCollectionListener()
            trace_listener.setup_listeners(crewai_event_bus)

            with patch.object(
                trace_listener.batch_manager,
                "add_event",
                wraps=trace_listener.batch_manager.add_event,
            ) as add_event_mock:
                crew.kickoff()
                wait_for_event_handlers()

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

    @pytest.mark.vcr()
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
            with crewai_event_bus._rwlock.r_locked():
                for handlers in crewai_event_bus._sync_handlers.values():
                    for handler in handlers:
                        if hasattr(handler, "__self__") and isinstance(
                            handler.__self__, TraceCollectionListener
                        ):
                            trace_handlers.append(handler)
                for handlers in crewai_event_bus._async_handlers.values():
                    for handler in handlers:
                        if hasattr(handler, "__self__") and isinstance(
                            handler.__self__, TraceCollectionListener
                        ):
                            trace_handlers.append(handler)

            assert len(trace_handlers) == 0, (
                f"Found {len(trace_handlers)} TraceCollectionListener handlers when tracing should be disabled"
            )

    def test_trace_listener_setup_correctly_for_crew(self):
        """Test that trace listener is set up correctly when enabled"""

        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
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
            with patch.object(
                TraceCollectionListener, "setup_listeners"
            ) as mock_listener_setup:
                Crew(agents=[agent], tasks=[task], verbose=True)
                assert mock_listener_setup.call_count >= 1

    @pytest.mark.vcr()
    def test_trace_listener_setup_correctly_for_flow(self):
        """Test that trace listener is set up correctly when enabled"""

        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ):
            class FlowExample(Flow):
                @start()
                def start(self):
                    pass

            with patch.object(
                TraceCollectionListener, "setup_listeners"
            ) as mock_listener_setup:
                FlowExample()
                assert mock_listener_setup.call_count >= 1

    @pytest.mark.vcr()
    def test_trace_listener_ephemeral_batch(self):
        """Test that trace listener properly handles ephemeral batches"""
        with (
            patch.dict(
                os.environ,
                {
                    "CREWAI_TRACING_ENABLED": "true",
                    "CREWAI_DISABLE_TELEMETRY": "false",
                    "CREWAI_DISABLE_TRACKING": "false",
                    "OTEL_SDK_DISABLED": "false",
                },
            ),
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

            from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
            trace_listener = TraceCollectionListener()

            crew.kickoff()

            initialized = trace_listener.batch_manager.wait_for_batch_initialization(timeout=5.0)
            assert initialized, (
                "Batch should have been initialized for unauthenticated user"
            )

            wait_for_event_handlers()

    @pytest.mark.vcr()
    def test_trace_listener_with_authenticated_user(self):
        """Test that trace listener properly handles authenticated batches"""
        with patch.dict(
            os.environ,
            {
                "CREWAI_TRACING_ENABLED": "true",
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
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

            from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
            trace_listener = TraceCollectionListener()

            crew = Crew(agents=[agent], tasks=[task], tracing=True)
            crew.kickoff()

            initialized = trace_listener.batch_manager.wait_for_batch_initialization(timeout=5.0)
            assert initialized, (
                "Batch should have been initialized for authenticated user"
            )

            wait_for_event_handlers()

    # Helper method to ensure cleanup
    def teardown_method(self):
        """Cleanup after each test method"""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_listener import EventListener

        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers = {}
            crewai_event_bus._async_handlers = {}
            crewai_event_bus._handler_dependencies = {}
            crewai_event_bus._execution_plan_cache = {}

        # Reset EventListener singleton
        if hasattr(EventListener, "_instance"):
            EventListener._instance = None

    @classmethod
    def teardown_class(cls):
        """Final cleanup after all tests in this class"""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_listener import EventListener

        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers = {}
            crewai_event_bus._async_handlers = {}
            crewai_event_bus._handler_dependencies = {}
            crewai_event_bus._execution_plan_cache = {}

        # Reset EventListener singleton
        if hasattr(EventListener, "_instance"):
            EventListener._instance = None

    @pytest.mark.vcr()
    def test_first_time_user_trace_collection_with_timeout(self, mock_plus_api_calls):
        """Test first-time user trace collection logic with timeout behavior"""

        with (
            patch.dict(
                os.environ,
                {
                    "CREWAI_TRACING_ENABLED": "false",
                    "CREWAI_DISABLE_TELEMETRY": "false",
                    "CREWAI_DISABLE_TRACKING": "false",
                    "OTEL_SDK_DISABLED": "false",
                },
            ),
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

            trace_listener.first_time_handler = FirstTimeTraceHandler()
            if trace_listener.first_time_handler.initialize_for_first_time_user():
                trace_listener.first_time_handler.set_batch_manager(trace_listener.batch_manager)

            assert trace_listener.first_time_handler.is_first_time is True
            assert trace_listener.first_time_handler.collected_events is False

            trace_listener.batch_manager.batch_owner_type = "crew"

            result = crew.kickoff()
            wait_for_event_handlers()
            assert result is not None

            assert trace_listener.first_time_handler.collected_events is True, (
                "Events should have been collected"
            )

            mock_prompt.assert_called_once()

            mock_mark_completed.assert_called_once()

    @pytest.mark.vcr()
    def test_first_time_user_trace_collection_user_accepts(self, mock_plus_api_calls):
        """Test first-time user trace collection when user accepts viewing traces"""

        with (
            patch.dict(
                os.environ,
                {
                    "CREWAI_TRACING_ENABLED": "false",
                    "CREWAI_DISABLE_TELEMETRY": "false",
                    "CREWAI_DISABLE_TRACKING": "false",
                    "OTEL_SDK_DISABLED": "false",
                },
            ),
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

            # Re-initialize first-time handler after patches are applied to ensure clean state
            trace_listener.first_time_handler = FirstTimeTraceHandler()
            if trace_listener.first_time_handler.initialize_for_first_time_user():
                trace_listener.first_time_handler.set_batch_manager(trace_listener.batch_manager)

            trace_listener.batch_manager.ephemeral_trace_url = (
                "https://crewai.com/trace/mock-id"
            )

            assert trace_listener.first_time_handler.is_first_time is True

            trace_listener.first_time_handler.collected_events = True

            with (
                patch.object(
                    trace_listener.first_time_handler,
                    "_initialize_backend_and_send_events",
                    wraps=trace_listener.first_time_handler._initialize_backend_and_send_events,
                ) as mock_init_backend,
                patch.object(
                    trace_listener.first_time_handler, "_display_ephemeral_trace_link"
                ) as mock_display_link,
            ):
                crew.kickoff()
                wait_for_event_handlers()

                mock_init_backend.assert_called_once()

                mock_display_link.assert_called_once()

            mock_mark_completed.assert_called_once()

    @pytest.mark.vcr()
    def test_first_time_user_trace_consolidation_logic(self, mock_plus_api_calls):
        """Test the consolidation logic for first-time users vs regular tracing"""
        with (
            patch.dict(
                os.environ,
                {
                    "CREWAI_TRACING_ENABLED": "",
                    "CREWAI_DISABLE_TELEMETRY": "false",
                    "CREWAI_DISABLE_TRACKING": "false",
                    "OTEL_SDK_DISABLED": "false",
                },
            ),
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

            with crewai_event_bus._rwlock.w_locked():
                crewai_event_bus._sync_handlers = {}
                crewai_event_bus._async_handlers = {}

            trace_listener = TraceCollectionListener()

            # Re-initialize first-time handler after patches are applied to ensure clean state
            # This is necessary because the singleton may have been created before patches were active
            trace_listener.first_time_handler = FirstTimeTraceHandler()
            if trace_listener.first_time_handler.initialize_for_first_time_user():
                trace_listener.first_time_handler.set_batch_manager(trace_listener.batch_manager)

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

            result = crew.kickoff()

            wait_for_event_handlers()

            assert trace_listener.batch_manager.is_batch_initialized(), (
                "Batch should have been initialized for first-time user"
            )
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

    def test_trace_batch_marked_as_failed_on_finalize_error(self):
        """Test that trace batch is marked as failed when finalization returns non-200 status"""
        # Test the error handling logic directly in TraceBatchManager
        with patch("crewai.events.listeners.tracing.trace_batch_manager.is_tracing_enabled_in_context", return_value=True):
            batch_manager = TraceBatchManager()

            # Initialize a batch
            batch_manager.current_batch = batch_manager.initialize_batch(
                user_context={"privacy_level": "standard"},
                execution_metadata={
                    "execution_type": "crew",
                    "crew_name": "test_crew",
                },
            )
            batch_manager.trace_batch_id = "test_batch_id_12345"
            batch_manager.backend_initialized = True

            # Mock the API responses
            with (
                patch.object(
                    batch_manager.plus_api,
                    "send_trace_events",
                    return_value=MagicMock(status_code=200),
                ),
                patch.object(
                    batch_manager.plus_api,
                    "finalize_trace_batch",
                    return_value=MagicMock(status_code=500, text="Internal Server Error"),
                ),
                patch.object(
                    batch_manager.plus_api,
                    "mark_trace_batch_as_failed",
                ) as mock_mark_failed,
            ):
                # Call finalize_batch directly
                batch_manager.finalize_batch()

                # Verify that mark_trace_batch_as_failed was called with the error message
                mock_mark_failed.assert_called_once_with(
                    "test_batch_id_12345", "Internal Server Error"
                )
