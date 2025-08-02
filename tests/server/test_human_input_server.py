import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock

try:
    from fastapi.testclient import TestClient
    from crewai.server.human_input_server import HumanInputServer
    from crewai.server.event_stream_manager import event_stream_manager
    from crewai.utilities.events.task_events import HumanInputRequiredEvent
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI dependencies not available")
class TestHumanInputServer:
    """Test the human input server endpoints"""

    def setup_method(self):
        """Setup test environment"""
        self.server = HumanInputServer(host="localhost", port=8001, api_key="test-key")
        self.client = TestClient(self.server.app)
        event_stream_manager._websocket_connections.clear()
        event_stream_manager._sse_connections.clear()
        event_stream_manager._polling_events.clear()

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_polling_endpoint_unauthorized(self):
        """Test polling endpoint without authentication"""
        response = self.client.get("/poll/human-input/test-execution-id")
        assert response.status_code == 401

    def test_polling_endpoint_authorized(self):
        """Test polling endpoint with authentication"""
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get("/poll/human-input/test-execution-id", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_polling_endpoint_with_events(self):
        """Test polling endpoint returns stored events"""
        execution_id = "test-execution-id"
        
        event = HumanInputRequiredEvent(
            execution_id=execution_id,
            crew_id="test-crew",
            task_id="test-task",
            agent_id="test-agent",
            prompt="Test prompt",
            context="Test context",
            event_id="test-event-1"
        )
        
        event_stream_manager._store_polling_event(execution_id, event.to_json())
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get(f"/poll/human-input/{execution_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["type"] == "human_input_required"
        assert data["events"][0]["execution_id"] == execution_id

    def test_polling_endpoint_with_last_event_id(self):
        """Test polling endpoint with last_event_id parameter"""
        execution_id = "test-execution-id"
        
        event1 = HumanInputRequiredEvent(
            execution_id=execution_id,
            event_id="event-1"
        )
        event2 = HumanInputRequiredEvent(
            execution_id=execution_id,
            event_id="event-2"
        )
        
        event_stream_manager._store_polling_event(execution_id, event1.to_json())
        event_stream_manager._store_polling_event(execution_id, event2.to_json())
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get(
            f"/poll/human-input/{execution_id}?last_event_id=event-1",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["event_id"] == "event-2"

    def test_sse_endpoint_unauthorized(self):
        """Test SSE endpoint without authentication"""
        response = self.client.get("/events/human-input/test-execution-id")
        assert response.status_code == 401

    def test_sse_endpoint_authorized(self):
        """Test SSE endpoint with authentication"""
        headers = {"Authorization": "Bearer test-key"}
        with self.client.stream("GET", "/events/human-input/test-execution-id", headers=headers) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_websocket_endpoint_unauthorized(self):
        """Test WebSocket endpoint without authentication"""
        with pytest.raises(Exception):
            with self.client.websocket_connect("/ws/human-input/test-execution-id"):
                pass

    def test_websocket_endpoint_authorized(self):
        """Test WebSocket endpoint with authentication"""
        with self.client.websocket_connect("/ws/human-input/test-execution-id?token=test-key") as websocket:
            assert websocket is not None

    def test_server_without_api_key(self):
        """Test server initialization without API key"""
        server = HumanInputServer(host="localhost", port=8002)
        client = TestClient(server.app)
        
        response = client.get("/poll/human-input/test-execution-id")
        assert response.status_code == 200
        
        response = client.get("/events/human-input/test-execution-id")
        assert response.status_code == 200


@pytest.mark.skipif(FASTAPI_AVAILABLE, reason="Testing import error handling")
def test_server_without_fastapi():
    """Test server initialization without FastAPI dependencies"""
    with pytest.raises(ImportError, match="FastAPI dependencies not available"):
        HumanInputServer()
