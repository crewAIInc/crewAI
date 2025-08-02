import asyncio
import json
import uuid
from typing import Dict, List, Optional, Set

from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.task_events import HumanInputRequiredEvent, HumanInputCompletedEvent


class EventStreamManager:
    """Manages event streams for human input events"""

    def __init__(self):
        self._websocket_connections: Dict[str, Set] = {}
        self._sse_connections: Dict[str, Set] = {}
        self._polling_events: Dict[str, List] = {}
        self._event_listeners_registered = False

    def register_event_listeners(self):
        """Register event listeners for human input events"""
        if self._event_listeners_registered:
            return

        @crewai_event_bus.on(HumanInputRequiredEvent)
        def handle_human_input_required(event: HumanInputRequiredEvent):
            self._broadcast_event(event)

        @crewai_event_bus.on(HumanInputCompletedEvent)
        def handle_human_input_completed(event: HumanInputCompletedEvent):
            self._broadcast_event(event)

        self._event_listeners_registered = True

    def add_websocket_connection(self, execution_id: str, websocket):
        """Add a WebSocket connection for an execution"""
        if execution_id not in self._websocket_connections:
            self._websocket_connections[execution_id] = set()
        self._websocket_connections[execution_id].add(websocket)

    def remove_websocket_connection(self, execution_id: str, websocket):
        """Remove a WebSocket connection"""
        if execution_id in self._websocket_connections:
            self._websocket_connections[execution_id].discard(websocket)
            if not self._websocket_connections[execution_id]:
                del self._websocket_connections[execution_id]

    def add_sse_connection(self, execution_id: str, queue):
        """Add an SSE connection for an execution"""
        if execution_id not in self._sse_connections:
            self._sse_connections[execution_id] = set()
        self._sse_connections[execution_id].add(queue)

    def remove_sse_connection(self, execution_id: str, queue):
        """Remove an SSE connection"""
        if execution_id in self._sse_connections:
            self._sse_connections[execution_id].discard(queue)
            if not self._sse_connections[execution_id]:
                del self._sse_connections[execution_id]

    def get_polling_events(self, execution_id: str, last_event_id: Optional[str] = None) -> List[Dict]:
        """Get events for polling clients"""
        if execution_id not in self._polling_events:
            return []

        events = self._polling_events[execution_id]
        
        if last_event_id:
            try:
                last_index = next(
                    i for i, event in enumerate(events)
                    if event.get("event_id") == last_event_id
                )
                return events[last_index + 1:]
            except StopIteration:
                pass

        return events

    def _broadcast_event(self, event):
        """Broadcast event to all relevant connections"""
        execution_id = getattr(event, 'execution_id', None)
        if not execution_id:
            return

        event_data = self._serialize_event(event)

        self._broadcast_websocket(execution_id, event_data)
        self._broadcast_sse(execution_id, event_data)
        self._store_polling_event(execution_id, event_data)

    def _serialize_event(self, event) -> Dict:
        """Serialize event to dictionary format"""
        event_dict = event.to_json()
        
        if not event_dict.get("event_id"):
            event_dict["event_id"] = str(uuid.uuid4())
        
        return event_dict

    def _broadcast_websocket(self, execution_id: str, event_data: Dict):
        """Broadcast event to WebSocket connections"""
        if execution_id not in self._websocket_connections:
            return

        connections_to_remove = set()
        for websocket in self._websocket_connections[execution_id]:
            try:
                asyncio.create_task(websocket.send_text(json.dumps(event_data)))
            except Exception:
                connections_to_remove.add(websocket)

        for websocket in connections_to_remove:
            self.remove_websocket_connection(execution_id, websocket)

    def _broadcast_sse(self, execution_id: str, event_data: Dict):
        """Broadcast event to SSE connections"""
        if execution_id not in self._sse_connections:
            return

        connections_to_remove = set()
        for queue in self._sse_connections[execution_id]:
            try:
                queue.put_nowait(event_data)
            except Exception:
                connections_to_remove.add(queue)

        for queue in connections_to_remove:
            self.remove_sse_connection(execution_id, queue)

    def _store_polling_event(self, execution_id: str, event_data: Dict):
        """Store event for polling clients"""
        if execution_id not in self._polling_events:
            self._polling_events[execution_id] = []

        self._polling_events[execution_id].append(event_data)
        
        if len(self._polling_events[execution_id]) > 100:
            self._polling_events[execution_id] = self._polling_events[execution_id][-100:]

    def cleanup_execution(self, execution_id: str):
        """Clean up all connections and events for an execution"""
        self._websocket_connections.pop(execution_id, None)
        self._sse_connections.pop(execution_id, None)
        self._polling_events.pop(execution_id, None)


event_stream_manager = EventStreamManager()
