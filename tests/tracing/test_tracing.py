# import os
# import pytest
# from unittest.mock import patch, MagicMock

# # Remove the module-level patch
# from crewai import Agent, Task, Crew
# from crewai.utilities.events.listeners.tracing.trace_listener import (
#     TraceCollectionListener,
# )
# from crewai.utilities.events.listeners.tracing.trace_batch_manager import (
#     TraceBatchManager,
# )
# from crewai.utilities.events.listeners.tracing.types import TraceEvent


# class TestTraceListenerSetup:
#     """Test TraceListener is properly setup and collecting events"""

#     @pytest.fixture(autouse=True)
#     def mock_auth_token(self):
#         """Mock authentication token for all tests in this class"""
#         # Need to patch all the places where get_auth_token is imported/used
#         with (
#             patch(
#                 "crewai.cli.authentication.token.get_auth_token",
#                 return_value="mock_token_12345",
#             ),
#             patch(
#                 "crewai.utilities.events.listeners.tracing.trace_listener.get_auth_token",
#                 return_value="mock_token_12345",
#             ),
#             patch(
#                 "crewai.utilities.events.listeners.tracing.trace_batch_manager.get_auth_token",
#                 return_value="mock_token_12345",
#             ),
#             patch(
#                 "crewai.utilities.events.listeners.tracing.interfaces.get_auth_token",
#                 return_value="mock_token_12345",
#             ),
#         ):
#             yield

#     @pytest.fixture(autouse=True)
#     def clear_event_bus(self):
#         """Clear event bus listeners before and after each test"""
#         from crewai.utilities.events import crewai_event_bus

#         # Store original handlers
#         original_handlers = crewai_event_bus._handlers.copy()

#         # Clear for test
#         crewai_event_bus._handlers.clear()

#         yield

#         # Restore original state
#         crewai_event_bus._handlers.clear()
#         crewai_event_bus._handlers.update(original_handlers)

#     @pytest.fixture(autouse=True)
#     def reset_tracing_singletons(self):
#         """Reset tracing singleton instances between tests"""
#         # Reset TraceCollectionListener singleton
#         if hasattr(TraceCollectionListener, "_instance"):
#             TraceCollectionListener._instance = None
#             TraceCollectionListener._initialized = False

#         yield

#         # Clean up after test
#         if hasattr(TraceCollectionListener, "_instance"):
#             TraceCollectionListener._instance = None
#             TraceCollectionListener._initialized = False

#     @pytest.fixture(autouse=True)
#     def mock_plus_api_calls(self):
#         """Mock all PlusAPI HTTP calls to avoid network requests"""
#         with (
#             patch("requests.post") as mock_post,
#             patch("requests.get") as mock_get,
#             patch("requests.put") as mock_put,
#             patch("requests.delete") as mock_delete,
#             patch.object(TraceBatchManager, "initialize_batch", return_value=None),
#             patch.object(
#                 TraceBatchManager, "_finalize_backend_batch", return_value=True
#             ),
#             patch.object(TraceBatchManager, "_cleanup_batch_data", return_value=True),
#         ):
#             mock_response = MagicMock()
#             mock_response.status_code = 200
#             mock_response.json.return_value = {
#                 "id": "mock_trace_batch_id",
#                 "status": "success",
#                 "message": "Batch created successfully",
#             }
#             mock_response.raise_for_status.return_value = None

#             mock_post.return_value = mock_response
#             mock_get.return_value = mock_response
#             mock_put.return_value = mock_response
#             mock_delete.return_value = mock_response

#             yield {
#                 "post": mock_post,
#                 "get": mock_get,
#                 "put": mock_put,
#                 "delete": mock_delete,
#             }

#     @pytest.mark.vcr(filter_headers=["authorization"])
#     def test_trace_listener_collects_crew_events(self):
#         """Test that trace listener properly collects events from crew execution"""

#         with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
#             agent = Agent(
#                 role="Test Agent",
#                 goal="Test goal",
#                 backstory="Test backstory",
#                 llm="gpt-4o-mini",
#             )
#             task = Task(
#                 description="Say hello to the world",
#                 expected_output="hello world",
#                 agent=agent,
#             )
#             crew = Crew(agents=[agent], tasks=[task], verbose=True)

#             trace_listener = TraceCollectionListener()
#             from crewai.utilities.events import crewai_event_bus

#             trace_listener.setup_listeners(crewai_event_bus)

#             with patch.object(
#                 trace_listener.batch_manager,
#                 "initialize_batch",
#                 return_value=None,
#             ) as initialize_mock:
#                 crew.kickoff()

#                 assert initialize_mock.call_count >= 1

#                 call_args = initialize_mock.call_args_list[0]
#                 assert len(call_args[0]) == 2  # user_context, execution_metadata
#                 _, execution_metadata = call_args[0]
#                 assert isinstance(execution_metadata, dict)
#                 assert "crew_name" in execution_metadata

#     @pytest.mark.vcr(filter_headers=["authorization"])
#     def test_batch_manager_finalizes_batch_clears_buffer(self):
#         """Test that batch manager properly finalizes batch and clears buffer"""

#         with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
#             agent = Agent(
#                 role="Test Agent",
#                 goal="Test goal",
#                 backstory="Test backstory",
#                 llm="gpt-4o-mini",
#             )

#             task = Task(
#                 description="Say hello to the world",
#                 expected_output="hello world",
#                 agent=agent,
#             )

#             crew = Crew(agents=[agent], tasks=[task], verbose=True)

#             from crewai.utilities.events import crewai_event_bus

#             trace_listener = None
#             for handler_list in crewai_event_bus._handlers.values():
#                 for handler in handler_list:
#                     if hasattr(handler, "__self__") and isinstance(
#                         handler.__self__, TraceCollectionListener
#                     ):
#                         trace_listener = handler.__self__
#                         break
#                 if trace_listener:
#                     break

#             if not trace_listener:
#                 pytest.skip(
#                     "No trace listener found - tracing may not be properly enabled"
#                 )

#             with patch.object(
#                 trace_listener.batch_manager,
#                 "finalize_batch",
#                 wraps=trace_listener.batch_manager.finalize_batch,
#             ) as finalize_mock:
#                 crew.kickoff()

#                 assert finalize_mock.call_count >= 1

#     @pytest.mark.vcr(filter_headers=["authorization"])
#     def test_events_collection_batch_manager(self, mock_plus_api_calls):
#         """Test that trace listener properly collects events from crew execution"""

#         with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
#             agent = Agent(
#                 role="Test Agent",
#                 goal="Test goal",
#                 backstory="Test backstory",
#                 llm="gpt-4o-mini",
#             )
#             task = Task(
#                 description="Say hello to the world",
#                 expected_output="hello world",
#                 agent=agent,
#             )
#             crew = Crew(agents=[agent], tasks=[task], verbose=True)

#             from crewai.utilities.events import crewai_event_bus

#             # Create and setup trace listener explicitly
#             trace_listener = TraceCollectionListener()
#             trace_listener.setup_listeners(crewai_event_bus)

#             with patch.object(
#                 trace_listener.batch_manager,
#                 "add_event",
#                 wraps=trace_listener.batch_manager.add_event,
#             ) as add_event_mock:
#                 crew.kickoff()

#                 assert add_event_mock.call_count >= 2

#                 completion_events = [
#                     call.args[0]
#                     for call in add_event_mock.call_args_list
#                     if call.args[0].type == "crew_kickoff_completed"
#                 ]
#                 assert len(completion_events) >= 1

#                 # Verify the first completion event has proper structure
#                 completion_event = completion_events[0]
#                 assert "crew_name" in completion_event.event_data
#                 assert completion_event.event_data["crew_name"] == "crew"

#                 # Verify all events have proper structure
#                 for call in add_event_mock.call_args_list:
#                     event = call.args[0]
#                     assert isinstance(event, TraceEvent)
#                     assert hasattr(event, "event_data")
#                     assert hasattr(event, "type")

#     @pytest.mark.vcr(filter_headers=["authorization"])
#     def test_trace_listener_disabled_when_env_false(self):
#         """Test that trace listener doesn't make HTTP calls when tracing is disabled"""

#         with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "false"}):
#             agent = Agent(
#                 role="Test Agent",
#                 goal="Test goal",
#                 backstory="Test backstory",
#                 llm="gpt-4o-mini",
#             )
#             task = Task(
#                 description="Say hello to the world",
#                 expected_output="hello world",
#                 agent=agent,
#             )

#             crew = Crew(agents=[agent], tasks=[task], verbose=True)
#             result = crew.kickoff()
#             assert result is not None

#             from crewai.utilities.events import crewai_event_bus

#             trace_handlers = []
#             for handlers in crewai_event_bus._handlers.values():
#                 for handler in handlers:
#                     if hasattr(handler, "__self__") and isinstance(
#                         handler.__self__, TraceCollectionListener
#                     ):
#                         trace_handlers.append(handler)
#                     elif hasattr(handler, "__name__") and any(
#                         trace_name in handler.__name__
#                         for trace_name in [
#                             "on_crew_started",
#                             "on_crew_completed",
#                             "on_flow_started",
#                         ]
#                     ):
#                         trace_handlers.append(handler)

#             assert len(trace_handlers) == 0, (
#                 f"Found {len(trace_handlers)} trace handlers when tracing should be disabled"
#             )

#     def test_trace_listener_setup_correctly(self):
#         """Test that trace listener is set up correctly when enabled"""

#         with patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}):
#             trace_listener = TraceCollectionListener()

#             assert trace_listener.trace_enabled is True
#             assert trace_listener.batch_manager is not None
#             assert trace_listener.trace_sender is not None

#     # Helper method to ensure cleanup
#     def teardown_method(self):
#         """Cleanup after each test method"""
#         from crewai.utilities.events import crewai_event_bus

#         crewai_event_bus._handlers.clear()

#     @classmethod
#     def teardown_class(cls):
#         """Final cleanup after all tests in this class"""
#         from crewai.utilities.events import crewai_event_bus

#         crewai_event_bus._handlers.clear()
