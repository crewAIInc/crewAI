import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from crewai.tools import BaseTool, Tool
from crewai.tools.mcp_connector import MCPToolConnector


class TestMCPToolConnector(unittest.TestCase):
    def setUp(self):
        self.mock_tool = MagicMock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
        self.mock_tool.description = "A test tool"
        self.mock_tool.args_schema = MagicMock()
        self.mock_tool.args_schema.model_json_schema.return_value = {
            "properties": {"input": {"type": "string"}}
        }
        self.mock_tool.run.return_value = "Tool result"
        
        self.connector = MCPToolConnector(tools=[self.mock_tool])

    @patch("crewai.cli.authentication.utils.TokenManager.get_access_token")
    @patch("crewai.tools.mcp_connector.SSEClient")
    def test_connect_success(self, mock_sse_client, mock_get_token):
        mock_get_token.return_value = "test-token"
        mock_sse = MagicMock()
        mock_sse_client.return_value = mock_sse
        
        self.connector.connect()
        
        mock_get_token.assert_called_once()
        mock_sse_client.assert_called_once_with(
            base_url="https://app.crewai.com",
            endpoint="/api/v1/tools/events",
            headers={
                "Authorization": "Bearer test-token",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=30
        )
        mock_sse.connect.assert_called_once()

    @patch("crewai.cli.authentication.utils.TokenManager.get_access_token")
    def test_connect_no_token(self, mock_get_token):
        mock_get_token.return_value = None
        
        with pytest.raises(ValueError, match="Authentication token not found"):
            self.connector.connect()

    @patch("crewai.cli.authentication.utils.TokenManager.get_access_token")
    @patch("crewai.tools.mcp_connector.SSEClient")
    def test_listen(self, mock_sse_client, mock_get_token):
        mock_get_token.return_value = "test-token"
        mock_sse = MagicMock()
        mock_sse_client.return_value = mock_sse
        
        self.connector._sse_client = mock_sse
        self.connector.listen()
        
        mock_sse.listen.assert_called_once()
        handlers = mock_sse.listen.call_args[0][0]
        assert "tool_request" in handlers
        assert "connection_closed" in handlers

    @patch("crewai.cli.authentication.utils.TokenManager.get_access_token")
    @patch("crewai.tools.mcp_connector.SSEClient")
    def test_handle_tool_request(self, mock_sse_client, mock_get_token):
        mock_get_token.return_value = "test-token"
        
        test_data = {
            "tool_name": "test_tool",
            "arguments": {"input": "test input"},
            "request_id": "123"
        }
        
        self.connector._handle_tool_request(test_data)
        
        self.mock_tool.run.assert_called_once_with(input="test input")

    def test_handle_tool_request_not_found(self):
        test_data = {
            "tool_name": "non_existent_tool",
            "arguments": {"input": "test input"},
            "request_id": "123"
        }
        
        self.connector._handle_tool_request(test_data)
        
        self.mock_tool.run.assert_not_called()
