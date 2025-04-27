import logging
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.cli.authentication.utils import TokenManager
from crewai.tools import BaseTool
from crewai.utilities.sse_client import SSEClient


class MCPToolConnector:
    """Connects tools to the Management Control Plane (MCP) via SSE."""

    MCP_BASE_URL = "https://app.crewai.com"
    SSE_ENDPOINT = "/api/v1/tools/events"
    
    def __init__(
        self, 
        tools: Optional[List[BaseTool]] = None,
        timeout: int = 30
    ):
        """Initialize the MCP Tool Connector.
        
        Args:
            tools: List of tools to connect to the MCP.
            timeout: Connection timeout in seconds.
        """
        self.tools = tools or []
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.token_manager = TokenManager()
        self._sse_client = None

    def connect(self) -> None:
        """Connect to the MCP SSE server for tools."""
        token = self.token_manager.get_token()
        if not token:
            self.logger.error("Authentication token not found. Please log in first.")
            raise ValueError("Authentication token not found. Please log in first.")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Requested-With": "XMLHttpRequest",
        }
        
        tool_data = {}
        for tool in self.tools:
            tool_data[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.args_schema.model_json_schema() if hasattr(tool.args_schema, "model_json_schema") else {},
            }
            
        
        self._sse_client = SSEClient(
            base_url=self.MCP_BASE_URL,
            endpoint=self.SSE_ENDPOINT,
            headers=headers,
            timeout=self.timeout
        )
        
        try:
            self._sse_client.connect()
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP SSE server: {str(e)}")
            raise

    def listen(self) -> None:
        """Listen for tool events from the MCP SSE server."""
        if not self._sse_client:
            self.connect()
            
        event_handlers = {
            "tool_request": self._handle_tool_request,
            "connection_closed": self._handle_connection_closed,
        }
        
        try:
            self._sse_client.listen(event_handlers)
        except Exception as e:
            self.logger.error(f"Error listening to MCP SSE events: {str(e)}")
            raise

    def _handle_tool_request(self, data: Dict[str, Any]) -> None:
        """Handle a tool request event from the MCP SSE server."""
        try:
            tool_name = data.get("tool_name")
            arguments = data.get("arguments", {})
            request_id = data.get("request_id")
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                self.logger.error(f"Tool '{tool_name}' not found")
                return
                
            result = tool.run(**arguments)
            
            
        except Exception as e:
            self.logger.error(f"Error handling tool request: {str(e)}")

    def _handle_connection_closed(self, data: Any) -> None:
        """Handle a connection closed event from the MCP SSE server."""
        self.logger.info("MCP SSE connection closed")

    def close(self) -> None:
        """Close the MCP SSE connection."""
        if self._sse_client:
            self._sse_client.close()
            self._sse_client = None
