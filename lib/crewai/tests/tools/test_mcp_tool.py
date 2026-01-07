"""Tests for MCPTool class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.mcp.config import MCPServerHTTP, MCPServerSSE, MCPServerStdio
from crewai.tools import BaseTool, MCPTool


@pytest.fixture
def mock_tool_definitions():
    """Create mock MCP tool definitions (as returned by list_tools)."""
    return [
        {
            "name": "search",
            "description": "Search for information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    ]


class TestMCPToolFromServer:
    """Tests for MCPTool.from_server() method."""

    def test_from_server_with_url(self, mock_tool_definitions):
        """Test from_server with an HTTP URL."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("https://api.example.com/mcp")

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)

            tool_names = [tool.name for tool in tools]
            assert "api_example_com_search" in tool_names
            assert "api_example_com_read_file" in tool_names

    def test_from_server_with_headers(self, mock_tool_definitions):
        """Test from_server with HTTP headers."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server(
                "https://api.example.com/mcp",
                headers={"Authorization": "Bearer token"},
            )

            assert len(tools) == 2
            mock_client_class.assert_called_once()

    def test_from_server_with_mcp_server_http_config(self, mock_tool_definitions):
        """Test from_server with MCPServerHTTP configuration."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            config = MCPServerHTTP(
                url="https://api.example.com/mcp",
                headers={"Authorization": "Bearer token"},
            )
            tools = MCPTool.from_server(config)

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_from_server_with_mcp_server_sse_config(self, mock_tool_definitions):
        """Test from_server with MCPServerSSE configuration."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            config = MCPServerSSE(
                url="https://api.example.com/mcp/sse",
                headers={"Authorization": "Bearer token"},
            )
            tools = MCPTool.from_server(config)

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_from_server_with_mcp_server_stdio_config(self, mock_tool_definitions):
        """Test from_server with MCPServerStdio configuration."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            config = MCPServerStdio(
                command="python",
                args=["server.py"],
                env={"API_KEY": "test_key"},
            )
            tools = MCPTool.from_server(config)

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_from_server_with_tool_filter(self, mock_tool_definitions):
        """Test from_server with tool filter."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            def filter_search_only(tools):
                return [t for t in tools if "search" in t["name"]]

            config = MCPServerHTTP(
                url="https://api.example.com/mcp",
                tool_filter=filter_search_only,
            )
            tools = MCPTool.from_server(config)

            assert len(tools) == 1
            assert "search" in tools[0].name

    def test_from_server_with_npx_server_name(self, mock_tool_definitions):
        """Test from_server with npm package name (uses npx)."""
        with (
            patch("crewai.mcp.MCPClient") as mock_client_class,
            patch("shutil.which") as mock_which,
        ):
            mock_which.return_value = "/usr/bin/npx"
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("@anthropic/mcp-server-filesystem")

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)

    def test_from_server_with_uvx_server_name(self, mock_tool_definitions):
        """Test from_server with Python package name (uses uvx)."""
        with (
            patch("crewai.mcp.MCPClient") as mock_client_class,
            patch("shutil.which") as mock_which,
        ):
            def which_side_effect(cmd):
                if cmd == "uvx":
                    return "/usr/bin/uvx"
                return None

            mock_which.side_effect = which_side_effect
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("ibmcloud-mcp-server")

            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)


class TestMCPToolResolveServerConfig:
    """Tests for MCPTool._resolve_server_config() method."""

    def test_resolve_http_url(self):
        """Test resolving HTTP URL to MCPServerHTTP config."""
        config = MCPTool._resolve_server_config("https://api.example.com/mcp")
        assert isinstance(config, MCPServerHTTP)
        assert config.url == "https://api.example.com/mcp"

    def test_resolve_http_url_with_headers(self):
        """Test resolving HTTP URL with headers."""
        config = MCPTool._resolve_server_config(
            "https://api.example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert isinstance(config, MCPServerHTTP)
        assert config.headers == {"Authorization": "Bearer token"}

    def test_resolve_mcp_server_config_passthrough(self):
        """Test that MCPServerConfig objects are passed through."""
        original_config = MCPServerHTTP(url="https://api.example.com/mcp")
        config = MCPTool._resolve_server_config(original_config)
        assert config is original_config

    def test_resolve_invalid_type_raises_error(self):
        """Test that invalid server type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid server type"):
            MCPTool._resolve_server_config(123)


class TestMCPToolCreateStdioConfig:
    """Tests for MCPTool._create_stdio_config() method."""

    def test_create_stdio_config_npm_package(self):
        """Test creating stdio config for npm package."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/npx"
            config = MCPTool._create_stdio_config("@anthropic/mcp-server-filesystem")
            assert isinstance(config, MCPServerStdio)
            assert config.command == "npx"
            assert config.args == ["-y", "@anthropic/mcp-server-filesystem"]

    def test_create_stdio_config_python_package(self):
        """Test creating stdio config for Python package."""
        with patch("shutil.which") as mock_which:
            def which_side_effect(cmd):
                if cmd == "uvx":
                    return "/usr/bin/uvx"
                return None

            mock_which.side_effect = which_side_effect
            config = MCPTool._create_stdio_config("ibmcloud-mcp-server")
            assert isinstance(config, MCPServerStdio)
            assert config.command == "uvx"
            assert config.args == ["ibmcloud-mcp-server"]

    def test_create_stdio_config_with_env(self):
        """Test creating stdio config with environment variables."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/npx"
            config = MCPTool._create_stdio_config(
                "@anthropic/mcp-server-filesystem",
                env={"HOME": "/home/user"},
            )
            assert config.env == {"HOME": "/home/user"}

    def test_create_stdio_config_no_runner_raises_error(self):
        """Test that missing npx/uvx raises ValueError."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            with pytest.raises(ValueError, match="Neither npx nor uvx found"):
                MCPTool._create_stdio_config("some-server")


class TestMCPToolExtractServerName:
    """Tests for MCPTool._extract_server_name() method."""

    def test_extract_server_name_from_stdio_npm(self):
        """Test extracting server name from npm package."""
        config = MCPServerStdio(
            command="npx",
            args=["-y", "@anthropic/mcp-server-filesystem"],
        )
        name = MCPTool._extract_server_name(config)
        assert name == "mcp-server-filesystem"

    def test_extract_server_name_from_stdio_simple(self):
        """Test extracting server name from simple package."""
        config = MCPServerStdio(
            command="uvx",
            args=["ibmcloud-mcp-server"],
        )
        name = MCPTool._extract_server_name(config)
        assert name == "ibmcloud_mcp_server"

    def test_extract_server_name_from_http(self):
        """Test extracting server name from HTTP URL."""
        config = MCPServerHTTP(url="https://api.example.com/mcp")
        name = MCPTool._extract_server_name(config)
        assert name == "api_example_com"

    def test_extract_server_name_from_sse(self):
        """Test extracting server name from SSE URL."""
        config = MCPServerSSE(url="https://api.example.com/mcp/sse")
        name = MCPTool._extract_server_name(config)
        assert name == "api_example_com"


class TestMCPToolJsonSchemaConversion:
    """Tests for MCPTool JSON schema to Pydantic conversion."""

    def test_json_schema_to_pydantic_basic(self):
        """Test converting basic JSON schema to Pydantic model."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        model = MCPTool._json_schema_to_pydantic("search", schema)

        assert model.__name__ == "SearchArgs"
        assert "query" in model.model_fields

    def test_json_schema_to_pydantic_with_optional(self):
        """Test converting JSON schema with optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["query"],
        }
        model = MCPTool._json_schema_to_pydantic("search", schema)

        assert "query" in model.model_fields
        assert "limit" in model.model_fields

    def test_json_type_to_python_string(self):
        """Test converting JSON string type to Python."""
        assert MCPTool._json_type_to_python({"type": "string"}) == str

    def test_json_type_to_python_integer(self):
        """Test converting JSON integer type to Python."""
        assert MCPTool._json_type_to_python({"type": "integer"}) == int

    def test_json_type_to_python_number(self):
        """Test converting JSON number type to Python."""
        assert MCPTool._json_type_to_python({"type": "number"}) == float

    def test_json_type_to_python_boolean(self):
        """Test converting JSON boolean type to Python."""
        assert MCPTool._json_type_to_python({"type": "boolean"}) == bool

    def test_json_type_to_python_array(self):
        """Test converting JSON array type to Python."""
        assert MCPTool._json_type_to_python({"type": "array"}) == list

    def test_json_type_to_python_object(self):
        """Test converting JSON object type to Python."""
        assert MCPTool._json_type_to_python({"type": "object"}) == dict

    def test_json_type_to_python_unknown(self):
        """Test converting unknown JSON type defaults to str."""
        assert MCPTool._json_type_to_python({"type": "unknown"}) == str


class TestMCPToolExecution:
    """Tests for MCP tool execution."""

    def test_tool_execution_sync(self, mock_tool_definitions):
        """Test tool execution in synchronous context."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value="search result")
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("https://api.example.com/mcp")
            assert len(tools) == 2

            search_tool = next(t for t in tools if "search" in t.name)
            result = search_tool.run(query="test query")

            assert result == "search result"

    @pytest.mark.asyncio
    async def test_tool_execution_async(self, mock_tool_definitions):
        """Test tool execution in async context."""
        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client.call_tool = AsyncMock(return_value="search result")
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("https://api.example.com/mcp")
            assert len(tools) == 2

            search_tool = next(t for t in tools if "search" in t.name)
            result = search_tool.run(query="test query")

            assert result == "search result"


class TestMCPToolIntegrationWithAgent:
    """Tests for MCPTool integration with Agent."""

    def test_mcp_tool_with_agent(self, mock_tool_definitions):
        """Test using MCPTool.from_server() tools with an Agent."""
        from crewai.agent.core import Agent

        with patch("crewai.mcp.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            tools = MCPTool.from_server("https://api.example.com/mcp")

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                tools=tools,
            )

            assert len(agent.tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in agent.tools)
