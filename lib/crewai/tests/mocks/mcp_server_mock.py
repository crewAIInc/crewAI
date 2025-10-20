"""Mock MCP server implementation for testing."""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock


class MockMCPTool:
    """Mock MCP tool for testing."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class MockMCPServer:
    """Mock MCP server for testing various scenarios."""

    def __init__(self, server_url: str, tools: List[MockMCPTool] = None, behavior: str = "normal"):
        self.server_url = server_url
        self.tools = tools or []
        self.behavior = behavior
        self.call_count = 0
        self.initialize_count = 0
        self.list_tools_count = 0

    def add_tool(self, name: str, description: str, input_schema: Dict[str, Any] = None):
        """Add a tool to the mock server."""
        tool = MockMCPTool(name, description, input_schema)
        self.tools.append(tool)
        return tool

    async def simulate_initialize(self):
        """Simulate MCP session initialization."""
        self.initialize_count += 1

        if self.behavior == "slow_init":
            await asyncio.sleep(15)  # Exceed connection timeout
        elif self.behavior == "init_error":
            raise Exception("Initialization failed")
        elif self.behavior == "auth_error":
            raise Exception("Authentication failed")

    async def simulate_list_tools(self):
        """Simulate MCP tools listing."""
        self.list_tools_count += 1

        if self.behavior == "slow_list":
            await asyncio.sleep(20)  # Exceed discovery timeout
        elif self.behavior == "list_error":
            raise Exception("Failed to list tools")
        elif self.behavior == "json_error":
            raise Exception("JSON parsing error in list_tools")

        mock_result = Mock()
        mock_result.tools = self.tools
        return mock_result

    async def simulate_call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Simulate MCP tool execution."""
        self.call_count += 1

        if self.behavior == "slow_execution":
            await asyncio.sleep(35)  # Exceed execution timeout
        elif self.behavior == "execution_error":
            raise Exception("Tool execution failed")
        elif self.behavior == "tool_not_found":
            raise Exception(f"Tool {tool_name} not found")

        # Find the tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool and self.behavior == "normal":
            raise Exception(f"Tool {tool_name} not found")

        # Create mock successful response
        mock_result = Mock()
        mock_result.content = [Mock(text=f"Result from {tool_name} with args: {arguments}")]
        return mock_result


class MockMCPServerFactory:
    """Factory for creating various types of mock MCP servers."""

    @staticmethod
    def create_working_server(server_url: str) -> MockMCPServer:
        """Create a mock server that works normally."""
        server = MockMCPServer(server_url, behavior="normal")
        server.add_tool("search_tool", "Search for information")
        server.add_tool("analysis_tool", "Analyze data")
        return server

    @staticmethod
    def create_slow_server(server_url: str, slow_operation: str = "init") -> MockMCPServer:
        """Create a mock server that is slow for testing timeouts."""
        behavior_map = {
            "init": "slow_init",
            "list": "slow_list",
            "execution": "slow_execution"
        }

        server = MockMCPServer(server_url, behavior=behavior_map.get(slow_operation, "slow_init"))
        server.add_tool("slow_tool", "A slow tool")
        return server

    @staticmethod
    def create_failing_server(server_url: str, failure_type: str = "connection") -> MockMCPServer:
        """Create a mock server that fails in various ways."""
        behavior_map = {
            "connection": "init_error",
            "auth": "auth_error",
            "list": "list_error",
            "json": "json_error",
            "execution": "execution_error",
            "tool_missing": "tool_not_found"
        }

        server = MockMCPServer(server_url, behavior=behavior_map.get(failure_type, "init_error"))
        if failure_type != "tool_missing":
            server.add_tool("failing_tool", "A tool that fails")
        return server

    @staticmethod
    def create_exa_like_server(server_url: str) -> MockMCPServer:
        """Create a mock server that mimics the Exa MCP server."""
        server = MockMCPServer(server_url, behavior="normal")
        server.add_tool(
            "web_search_exa",
            "Search the web using Exa AI - performs real-time web searches and can scrape content from specific URLs",
            {"type": "object", "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}}}
        )
        server.add_tool(
            "get_code_context_exa",
            "Search and get relevant context for any programming task. Exa-code has the highest quality context",
            {"type": "object", "properties": {"query": {"type": "string"}, "language": {"type": "string"}}}
        )
        return server

    @staticmethod
    def create_weather_like_server(server_url: str) -> MockMCPServer:
        """Create a mock server that mimics a weather MCP server."""
        server = MockMCPServer(server_url, behavior="normal")
        server.add_tool(
            "get_current_weather",
            "Get current weather conditions for a location",
            {"type": "object", "properties": {"location": {"type": "string"}}}
        )
        server.add_tool(
            "get_forecast",
            "Get weather forecast for the next 5 days",
            {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}
        )
        server.add_tool(
            "get_alerts",
            "Get active weather alerts for a region",
            {"type": "object", "properties": {"region": {"type": "string"}}}
        )
        return server


class MCPServerContextManager:
    """Context manager for mock MCP servers."""

    def __init__(self, mock_server: MockMCPServer):
        self.mock_server = mock_server

    async def __aenter__(self):
        return (None, None, None)  # read, write, cleanup

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MCPSessionContextManager:
    """Context manager for mock MCP sessions."""

    def __init__(self, mock_server: MockMCPServer):
        self.mock_server = mock_server

    async def __aenter__(self):
        return MockMCPSession(self.mock_server)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self, mock_server: MockMCPServer):
        self.mock_server = mock_server

    async def initialize(self):
        """Mock session initialization."""
        await self.mock_server.simulate_initialize()

    async def list_tools(self):
        """Mock tools listing."""
        return await self.mock_server.simulate_list_tools()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Mock tool execution."""
        return await self.mock_server.simulate_call_tool(tool_name, arguments)


def mock_streamablehttp_client(server_url: str, mock_server: MockMCPServer):
    """Create a mock streamable HTTP client for testing."""
    return MCPServerContextManager(mock_server)


def mock_client_session(read, write, mock_server: MockMCPServer):
    """Create a mock client session for testing."""
    return MCPSessionContextManager(mock_server)


# Convenience functions for common test scenarios

def create_successful_exa_mock():
    """Create a successful Exa-like mock server."""
    return MockMCPServerFactory.create_exa_like_server("https://mcp.exa.ai/mcp")


def create_failing_connection_mock():
    """Create a mock server that fails to connect."""
    return MockMCPServerFactory.create_failing_server("https://failing.com/mcp", "connection")


def create_timeout_mock():
    """Create a mock server that times out."""
    return MockMCPServerFactory.create_slow_server("https://slow.com/mcp", "init")


def create_mixed_servers_scenario():
    """Create a mixed scenario with working and failing servers."""
    return {
        "working": MockMCPServerFactory.create_working_server("https://working.com/mcp"),
        "failing": MockMCPServerFactory.create_failing_server("https://failing.com/mcp"),
        "slow": MockMCPServerFactory.create_slow_server("https://slow.com/mcp"),
        "auth_fail": MockMCPServerFactory.create_failing_server("https://auth-fail.com/mcp", "auth")
    }


# Pytest fixtures for common mock scenarios

@pytest.fixture
def mock_exa_server():
    """Provide mock Exa server for tests."""
    return create_successful_exa_mock()


@pytest.fixture
def mock_failing_server():
    """Provide mock failing server for tests."""
    return create_failing_connection_mock()


@pytest.fixture
def mock_slow_server():
    """Provide mock slow server for tests."""
    return create_timeout_mock()


@pytest.fixture
def mixed_mock_servers():
    """Provide mixed mock servers scenario."""
    return create_mixed_servers_scenario()
