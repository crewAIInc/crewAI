import json
import unittest
from unittest.mock import MagicMock, patch

import pytest
import requests
import sseclient

from crewai.utilities.sse_client import (
    SSEClient,
    SSEConnectionErrorEvent,
    SSEConnectionStartedEvent,
    SSEMessageReceivedEvent,
)


class TestSSEClient(unittest.TestCase):
    def setUp(self):
        self.base_url = "https://test.example.com"
        self.endpoint = "/events"
        self.headers = {"Authorization": "Bearer test-token"}
        self.sse_client = SSEClient(
            base_url=self.base_url,
            endpoint=self.endpoint,
            headers=self.headers
        )

    @patch("crewai.utilities.events.crewai_event_bus.emit")
    @patch("requests.get")
    @patch("sseclient.SSEClient")
    def test_connect_success(self, mock_sse_client, mock_get, mock_emit):
        mock_response = MagicMock()
        mock_get.return_value = mock_response
        
        self.sse_client.connect()
        
        mock_get.assert_called_once_with(
            "https://test.example.com/events",
            headers=self.headers,
            stream=True,
            timeout=30
        )
        mock_response.raise_for_status.assert_called_once()
        mock_sse_client.assert_called_once_with(mock_response)
        mock_emit.assert_called_once()
        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, SSEConnectionStartedEvent)
        assert event.endpoint == "https://test.example.com/events"
        assert event.headers == self.headers

    @patch("crewai.utilities.events.crewai_event_bus.emit")
    @patch("requests.get")
    def test_connect_error(self, mock_get, mock_emit):
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        with pytest.raises(requests.exceptions.RequestException):
            self.sse_client.connect()
        
        mock_emit.assert_called_once()
        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, SSEConnectionErrorEvent)
        assert event.endpoint == "https://test.example.com/events"
        assert "Connection error" in event.error

    @patch("crewai.utilities.events.crewai_event_bus.emit")
    @patch("requests.get")
    def test_listen_with_handlers(self, mock_get, mock_emit):
        mock_response = MagicMock()
        mock_get.return_value = mock_response
        
        mock_sse_client = MagicMock()
        mock_event1 = MagicMock(event="test_event", data='{"key": "value"}')
        mock_event2 = MagicMock(event="message", data="plain text")
        mock_sse_client.__iter__.return_value = [mock_event1, mock_event2]
        
        self.sse_client._client = mock_sse_client
        
        test_event_handler = MagicMock()
        message_handler = MagicMock()
        
        event_handlers = {
            "test_event": test_event_handler,
            "message": message_handler
        }
        self.sse_client.listen(event_handlers)
        
        test_event_handler.assert_called_once_with({"key": "value"})
        message_handler.assert_called_once_with("plain text")
        
        assert mock_emit.call_count == 2
        event1 = mock_emit.call_args_list[0][1]["event"]
        event2 = mock_emit.call_args_list[1][1]["event"]
        
        assert isinstance(event1, SSEMessageReceivedEvent)
        assert event1.event == "test_event"
        assert event1.data == {"key": "value"}
        
        assert isinstance(event2, SSEMessageReceivedEvent)
        assert event2.event == "message"
        assert event2.data == "plain text"
