import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from crewai.tools import BaseTool, MCPToolConnector, Tool


@pytest.mark.integration
class TestMCPToolsIntegration(unittest.TestCase):
    @pytest.mark.skipif(
        not os.environ.get("CREWAI_INTEGRATION_TEST"),
        reason="Integration test requires CREWAI_INTEGRATION_TEST=true"
    )
    @patch("crewai.tools.mcp_connector.SSEClient")
    def test_mcp_tool_connector_integration(self, mock_sse_client):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
            
        calculator_tool = Tool(
            name="calculator_add",
            description="Add two numbers",
            func=add
        )
        
        connector = MCPToolConnector(tools=[calculator_tool])
        
        mock_sse = MagicMock()
        mock_sse_client.return_value = mock_sse
        
        connector.connect()
        
        tool_request_data = {
            "tool_name": "calculator_add",
            "arguments": {"a": 5, "b": 7},
            "request_id": "test-request-1"
        }
        
        connector._handle_tool_request(tool_request_data)
