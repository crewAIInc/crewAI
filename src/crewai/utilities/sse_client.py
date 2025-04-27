import json
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Union
from urllib.parse import urljoin

import requests
import sseclient

from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.base_events import BaseEvent


class SSEConnectionStartedEvent(BaseEvent):
    """Event emitted when an SSE connection is started"""
    type: str = "sse_connection_started"
    endpoint: str
    headers: Dict[str, str]


class SSEConnectionErrorEvent(BaseEvent):
    """Event emitted when an SSE connection encounters an error"""
    type: str = "sse_connection_error"
    endpoint: str
    error: str


class SSEMessageReceivedEvent(BaseEvent):
    """Event emitted when an SSE message is received"""
    type: str = "sse_message_received"
    endpoint: str
    event: str
    data: Any


class SSEClient:
    """Client for connecting to Server-Sent Events (SSE) endpoints"""

    def __init__(
        self,
        base_url: str,
        endpoint: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        """Initialize the SSE client.
        
        Args:
            base_url: Base URL for the SSE server.
            endpoint: Endpoint path to connect to (will be joined with base_url).
            headers: Headers to include in the SSE request.
            timeout: Connection timeout in seconds.
        """
        self.base_url = base_url
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._client: Optional[sseclient.SSEClient] = None
        self._response: Optional[requests.Response] = None

    def connect(self) -> None:
        """Establish a connection to the SSE server."""
        try:
            url = urljoin(self.base_url, self.endpoint)
            self.logger.info(f"Connecting to SSE server at {url}")
            
            crewai_event_bus.emit(
                self,
                event=SSEConnectionStartedEvent(
                    endpoint=url,
                    headers=self.headers
                )
            )
            
            self._response = requests.get(
                url,
                headers=self.headers,
                stream=True,
                timeout=self.timeout
            )
            if self._response is not None:
                self._response.raise_for_status()
                self._client = sseclient.SSEClient(self._response)
            
        except Exception as e:
            self.logger.error(f"Error connecting to SSE server: {str(e)}")
            crewai_event_bus.emit(
                self,
                event=SSEConnectionErrorEvent(
                    endpoint=urljoin(self.base_url, self.endpoint),
                    error=str(e)
                )
            )
            raise

    def listen(self, event_handlers: Optional[Mapping[str, Callable[[Any], None]]] = None) -> None:
        """Listen for SSE events and process them with registered handlers.
        
        Args:
            event_handlers: Dictionary mapping event types to handler functions.
        """
        if self._client is None:
            self.connect()
        
        event_handlers = event_handlers or {}
        
        try:
            if self._client is None:
                self.logger.error("SSE client is not initialized")
                return
                
            for event in self._client:
                event_type = event.event or "message"
                data = None
                
                try:
                    data = json.loads(event.data)
                except (json.JSONDecodeError, TypeError):
                    data = event.data
                
                crewai_event_bus.emit(
                    self,
                    event=SSEMessageReceivedEvent(
                        endpoint=urljoin(self.base_url, self.endpoint),
                        event=event_type,
                        data=data
                    )
                )
                
                handler = event_handlers.get(event_type)
                if handler:
                    handler(data)
                    
        except Exception as e:
            self.logger.error(f"Error processing SSE events: {str(e)}")
            crewai_event_bus.emit(
                self,
                event=SSEConnectionErrorEvent(
                    endpoint=urljoin(self.base_url, self.endpoint),
                    error=str(e)
                )
            )
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Close the SSE connection."""
        if self._response:
            self._response.close()
            self._response = None
            self._client = None
