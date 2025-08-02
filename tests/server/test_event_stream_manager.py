from unittest.mock import MagicMock, patch

from crewai.server.event_stream_manager import EventStreamManager
from crewai.utilities.events.task_events import HumanInputRequiredEvent


class TestEventStreamManager:
    """Test the event stream manager"""

    def setup_method(self):
        """Setup test environment"""
        self.manager = EventStreamManager()
        self.manager._websocket_connections.clear()
        self.manager._sse_connections.clear()
        self.manager._polling_events.clear()

    def test_websocket_connection_management(self):
        """Test WebSocket connection management"""
        execution_id = "test-execution"
        websocket1 = MagicMock()
        websocket2 = MagicMock()

        self.manager.add_websocket_connection(execution_id, websocket1)
        assert execution_id in self.manager._websocket_connections
        assert websocket1 in self.manager._websocket_connections[execution_id]

        self.manager.add_websocket_connection(execution_id, websocket2)
        assert len(self.manager._websocket_connections[execution_id]) == 2

        self.manager.remove_websocket_connection(execution_id, websocket1)
        assert websocket1 not in self.manager._websocket_connections[execution_id]
        assert websocket2 in self.manager._websocket_connections[execution_id]

        self.manager.remove_websocket_connection(execution_id, websocket2)
        assert execution_id not in self.manager._websocket_connections

    def test_sse_connection_management(self):
        """Test SSE connection management"""
        execution_id = "test-execution"
        queue1 = MagicMock()
        queue2 = MagicMock()

        self.manager.add_sse_connection(execution_id, queue1)
        assert execution_id in self.manager._sse_connections
        assert queue1 in self.manager._sse_connections[execution_id]

        self.manager.add_sse_connection(execution_id, queue2)
        assert len(self.manager._sse_connections[execution_id]) == 2

        self.manager.remove_sse_connection(execution_id, queue1)
        assert queue1 not in self.manager._sse_connections[execution_id]
        assert queue2 in self.manager._sse_connections[execution_id]

        self.manager.remove_sse_connection(execution_id, queue2)
        assert execution_id not in self.manager._sse_connections

    def test_polling_events_storage(self):
        """Test polling events storage and retrieval"""
        execution_id = "test-execution"
        
        event1 = {"event_id": "event-1", "type": "test", "data": "test1"}
        event2 = {"event_id": "event-2", "type": "test", "data": "test2"}
        
        self.manager._store_polling_event(execution_id, event1)
        self.manager._store_polling_event(execution_id, event2)
        
        events = self.manager.get_polling_events(execution_id)
        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2

    def test_polling_events_with_last_event_id(self):
        """Test polling events retrieval with last_event_id"""
        execution_id = "test-execution"
        
        event1 = {"event_id": "event-1", "type": "test", "data": "test1"}
        event2 = {"event_id": "event-2", "type": "test", "data": "test2"}
        event3 = {"event_id": "event-3", "type": "test", "data": "test3"}
        
        self.manager._store_polling_event(execution_id, event1)
        self.manager._store_polling_event(execution_id, event2)
        self.manager._store_polling_event(execution_id, event3)
        
        events = self.manager.get_polling_events(execution_id, "event-1")
        assert len(events) == 2
        assert events[0] == event2
        assert events[1] == event3

    def test_polling_events_limit(self):
        """Test polling events storage limit"""
        execution_id = "test-execution"
        
        for i in range(105):
            event = {"event_id": f"event-{i}", "type": "test", "data": f"test{i}"}
            self.manager._store_polling_event(execution_id, event)
        
        events = self.manager.get_polling_events(execution_id)
        assert len(events) == 100
        assert events[0]["event_id"] == "event-5"
        assert events[-1]["event_id"] == "event-104"

    def test_event_serialization(self):
        """Test event serialization"""
        event = HumanInputRequiredEvent(
            execution_id="test-execution",
            crew_id="test-crew",
            task_id="test-task",
            prompt="Test prompt"
        )
        
        serialized = self.manager._serialize_event(event)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "human_input_required"
        assert serialized["execution_id"] == "test-execution"
        assert "event_id" in serialized

    def test_broadcast_websocket(self):
        """Test WebSocket broadcasting"""
        execution_id = "test-execution"
        websocket = MagicMock()
        
        self.manager.add_websocket_connection(execution_id, websocket)
        
        event_data = {"type": "test", "data": "test"}
        
        with patch('asyncio.create_task') as mock_create_task:
            self.manager._broadcast_websocket(execution_id, event_data)
            mock_create_task.assert_called_once()

    def test_broadcast_sse(self):
        """Test SSE broadcasting"""
        execution_id = "test-execution"
        queue = MagicMock()
        
        self.manager.add_sse_connection(execution_id, queue)
        
        event_data = {"type": "test", "data": "test"}
        self.manager._broadcast_sse(execution_id, event_data)
        
        queue.put_nowait.assert_called_once_with(event_data)

    def test_broadcast_event(self):
        """Test complete event broadcasting"""
        execution_id = "test-execution"
        
        event = HumanInputRequiredEvent(
            execution_id=execution_id,
            crew_id="test-crew",
            task_id="test-task",
            prompt="Test prompt"
        )
        
        with patch.object(self.manager, '_broadcast_websocket') as mock_ws, \
             patch.object(self.manager, '_broadcast_sse') as mock_sse, \
             patch.object(self.manager, '_store_polling_event') as mock_poll:
            
            self.manager._broadcast_event(event)
            
            mock_ws.assert_called_once()
            mock_sse.assert_called_once()
            mock_poll.assert_called_once()

    def test_cleanup_execution(self):
        """Test execution cleanup"""
        execution_id = "test-execution"
        
        websocket = MagicMock()
        queue = MagicMock()
        event = {"event_id": "event-1", "type": "test"}
        
        self.manager.add_websocket_connection(execution_id, websocket)
        self.manager.add_sse_connection(execution_id, queue)
        self.manager._store_polling_event(execution_id, event)
        
        assert execution_id in self.manager._websocket_connections
        assert execution_id in self.manager._sse_connections
        assert execution_id in self.manager._polling_events
        
        self.manager.cleanup_execution(execution_id)
        
        assert execution_id not in self.manager._websocket_connections
        assert execution_id not in self.manager._sse_connections
        assert execution_id not in self.manager._polling_events

    def test_register_event_listeners(self):
        """Test event listener registration"""
        with patch('crewai.utilities.events.crewai_event_bus.crewai_event_bus.on') as mock_on:
            self.manager.register_event_listeners()
            assert mock_on.call_count == 2
            
            self.manager.register_event_listeners()
            assert mock_on.call_count == 2

    def test_broadcast_event_without_execution_id(self):
        """Test broadcasting event without execution_id"""
        event = HumanInputRequiredEvent(
            crew_id="test-crew",
            task_id="test-task",
            prompt="Test prompt"
        )
        
        with patch.object(self.manager, '_broadcast_websocket') as mock_ws, \
             patch.object(self.manager, '_broadcast_sse') as mock_sse, \
             patch.object(self.manager, '_store_polling_event') as mock_poll:
            
            self.manager._broadcast_event(event)
            
            mock_ws.assert_not_called()
            mock_sse.assert_not_called()
            mock_poll.assert_not_called()
