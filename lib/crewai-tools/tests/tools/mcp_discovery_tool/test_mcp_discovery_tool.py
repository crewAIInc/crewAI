"""Tests for the MCP Discovery Tool."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool import (
    MCPDiscoveryConstraints,
    MCPDiscoveryResult,
    MCPDiscoveryTool,
    MCPDiscoveryToolSchema,
    MCPServerMetrics,
    MCPServerRecommendation,
)


@pytest.fixture
def mock_api_response() -> dict:
    """Create a mock API response for testing."""
    return {
        "recommendations": [
            {
                "server": "sqlite-server",
                "npm_package": "@modelcontextprotocol/server-sqlite",
                "install_command": "npx -y @modelcontextprotocol/server-sqlite",
                "confidence": 0.38,
                "description": "SQLite database server for MCP.",
                "capabilities": ["sqlite", "sql", "database", "embedded"],
                "metrics": {
                    "avg_latency_ms": 50.0,
                    "uptime_pct": 99.9,
                    "last_checked": "2026-01-17T10:30:00Z",
                },
                "docs_url": "https://modelcontextprotocol.io/docs/servers/sqlite",
                "github_url": "https://github.com/modelcontextprotocol/servers",
            },
            {
                "server": "postgres-server",
                "npm_package": "@modelcontextprotocol/server-postgres",
                "install_command": "npx -y @modelcontextprotocol/server-postgres",
                "confidence": 0.33,
                "description": "PostgreSQL database server for MCP.",
                "capabilities": ["postgres", "sql", "database", "queries"],
                "metrics": {
                    "avg_latency_ms": None,
                    "uptime_pct": None,
                    "last_checked": None,
                },
                "docs_url": "https://modelcontextprotocol.io/docs/servers/postgres",
                "github_url": "https://github.com/modelcontextprotocol/servers",
            },
        ],
        "total_found": 2,
        "query_time_ms": 245,
    }


@pytest.fixture
def discovery_tool() -> MCPDiscoveryTool:
    """Create an MCPDiscoveryTool instance for testing."""
    return MCPDiscoveryTool()


class TestMCPDiscoveryToolSchema:
    """Tests for MCPDiscoveryToolSchema."""

    def test_schema_with_required_fields(self) -> None:
        """Test schema with only required fields."""
        schema = MCPDiscoveryToolSchema(need="database server")
        assert schema.need == "database server"
        assert schema.constraints is None
        assert schema.limit == 5

    def test_schema_with_all_fields(self) -> None:
        """Test schema with all fields."""
        constraints = MCPDiscoveryConstraints(
            max_latency_ms=200,
            required_features=["auth", "realtime"],
            exclude_servers=["deprecated-server"],
        )
        schema = MCPDiscoveryToolSchema(
            need="database with authentication",
            constraints=constraints,
            limit=3,
        )
        assert schema.need == "database with authentication"
        assert schema.constraints is not None
        assert schema.constraints.max_latency_ms == 200
        assert schema.constraints.required_features == ["auth", "realtime"]
        assert schema.constraints.exclude_servers == ["deprecated-server"]
        assert schema.limit == 3

    def test_schema_limit_validation(self) -> None:
        """Test that limit is validated to be between 1 and 10."""
        with pytest.raises(ValueError):
            MCPDiscoveryToolSchema(need="test", limit=0)

        with pytest.raises(ValueError):
            MCPDiscoveryToolSchema(need="test", limit=11)


class TestMCPDiscoveryConstraints:
    """Tests for MCPDiscoveryConstraints."""

    def test_empty_constraints(self) -> None:
        """Test creating empty constraints."""
        constraints = MCPDiscoveryConstraints()
        assert constraints.max_latency_ms is None
        assert constraints.required_features is None
        assert constraints.exclude_servers is None

    def test_full_constraints(self) -> None:
        """Test creating constraints with all fields."""
        constraints = MCPDiscoveryConstraints(
            max_latency_ms=100,
            required_features=["feature1", "feature2"],
            exclude_servers=["server1", "server2"],
        )
        assert constraints.max_latency_ms == 100
        assert constraints.required_features == ["feature1", "feature2"]
        assert constraints.exclude_servers == ["server1", "server2"]


class TestMCPDiscoveryTool:
    """Tests for MCPDiscoveryTool."""

    def test_tool_initialization(self, discovery_tool: MCPDiscoveryTool) -> None:
        """Test tool initialization with default values."""
        assert discovery_tool.name == "Discover MCP Server"
        assert "MCP" in discovery_tool.description
        assert discovery_tool.base_url == "https://mcp-discovery-production.up.railway.app"
        assert discovery_tool.timeout == 30

    def test_build_request_payload_basic(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test building request payload with basic parameters."""
        payload = discovery_tool._build_request_payload(
            need="database server",
            constraints=None,
            limit=5,
        )
        assert payload == {"need": "database server", "limit": 5}

    def test_build_request_payload_with_constraints(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test building request payload with constraints."""
        constraints = MCPDiscoveryConstraints(
            max_latency_ms=200,
            required_features=["auth"],
            exclude_servers=["old-server"],
        )
        payload = discovery_tool._build_request_payload(
            need="database",
            constraints=constraints,
            limit=3,
        )
        assert payload["need"] == "database"
        assert payload["limit"] == 3
        assert "constraints" in payload
        assert payload["constraints"]["max_latency_ms"] == 200
        assert payload["constraints"]["required_features"] == ["auth"]
        assert payload["constraints"]["exclude_servers"] == ["old-server"]

    def test_build_request_payload_partial_constraints(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test building request payload with partial constraints."""
        constraints = MCPDiscoveryConstraints(max_latency_ms=100)
        payload = discovery_tool._build_request_payload(
            need="test",
            constraints=constraints,
            limit=5,
        )
        assert payload["constraints"] == {"max_latency_ms": 100}

    def test_process_recommendations(
        self, discovery_tool: MCPDiscoveryTool, mock_api_response: dict
    ) -> None:
        """Test processing recommendations from API response."""
        recommendations = discovery_tool._process_recommendations(
            mock_api_response["recommendations"]
        )
        assert len(recommendations) == 2

        first_rec = recommendations[0]
        assert first_rec["server"] == "sqlite-server"
        assert first_rec["npm_package"] == "@modelcontextprotocol/server-sqlite"
        assert first_rec["confidence"] == 0.38
        assert first_rec["capabilities"] == ["sqlite", "sql", "database", "embedded"]
        assert first_rec["metrics"]["avg_latency_ms"] == 50.0
        assert first_rec["metrics"]["uptime_pct"] == 99.9

        second_rec = recommendations[1]
        assert second_rec["server"] == "postgres-server"
        assert second_rec["metrics"]["avg_latency_ms"] is None

    def test_process_recommendations_with_malformed_data(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test processing recommendations with malformed data."""
        malformed_recommendations = [
            {"server": "valid-server", "confidence": 0.5},
            None,
            {"invalid": "data"},
        ]
        recommendations = discovery_tool._process_recommendations(
            malformed_recommendations
        )
        assert len(recommendations) >= 1
        assert recommendations[0]["server"] == "valid-server"

    def test_format_result_with_recommendations(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test formatting results with recommendations."""
        result: MCPDiscoveryResult = {
            "recommendations": [
                {
                    "server": "test-server",
                    "npm_package": "@test/server",
                    "install_command": "npx -y @test/server",
                    "confidence": 0.85,
                    "description": "A test server",
                    "capabilities": ["test", "demo"],
                    "metrics": {
                        "avg_latency_ms": 100.0,
                        "uptime_pct": 99.5,
                        "last_checked": "2026-01-17T10:00:00Z",
                    },
                    "docs_url": "https://example.com/docs",
                    "github_url": "https://github.com/test/server",
                }
            ],
            "total_found": 1,
            "query_time_ms": 150,
        }
        formatted = discovery_tool._format_result(result)
        assert "Found 1 MCP server(s)" in formatted
        assert "test-server" in formatted
        assert "85% confidence" in formatted
        assert "A test server" in formatted
        assert "test, demo" in formatted
        assert "npx -y @test/server" in formatted
        assert "100.0ms" in formatted
        assert "99.5%" in formatted

    def test_format_result_empty(self, discovery_tool: MCPDiscoveryTool) -> None:
        """Test formatting results with no recommendations."""
        result: MCPDiscoveryResult = {
            "recommendations": [],
            "total_found": 0,
            "query_time_ms": 50,
        }
        formatted = discovery_tool._format_result(result)
        assert "No MCP servers found" in formatted

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_make_api_request_success(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = discovery_tool._make_api_request({"need": "database", "limit": 5})

        assert result == mock_api_response
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {"need": "database", "limit": 5}
        assert call_args[1]["timeout"] == 30

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_make_api_request_with_api_key(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test API request with API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"MCP_DISCOVERY_API_KEY": "test-key"}):
            discovery_tool._make_api_request({"need": "test", "limit": 5})

        call_args = mock_post.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_make_api_request_empty_response(
        self, mock_post: MagicMock, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test API request with empty response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            discovery_tool._make_api_request({"need": "test", "limit": 5})

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_make_api_request_network_error(
        self, mock_post: MagicMock, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test API request with network error."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(requests.exceptions.ConnectionError):
            discovery_tool._make_api_request({"need": "test", "limit": 5})

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_make_api_request_json_decode_error(
        self, mock_post: MagicMock, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test API request with JSON decode error."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"invalid json"
        mock_post.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            discovery_tool._make_api_request({"need": "test", "limit": 5})

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_run_success(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test successful _run execution."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = discovery_tool._run(need="database server")

        assert "sqlite-server" in result
        assert "postgres-server" in result
        assert "Found 2 MCP server(s)" in result

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_run_with_constraints(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test _run with constraints."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = discovery_tool._run(
            need="database",
            constraints={"max_latency_ms": 100, "required_features": ["sql"]},
            limit=3,
        )

        assert "sqlite-server" in result
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["constraints"]["max_latency_ms"] == 100
        assert payload["constraints"]["required_features"] == ["sql"]
        assert payload["limit"] == 3

    def test_run_missing_need_parameter(
        self, discovery_tool: MCPDiscoveryTool
    ) -> None:
        """Test _run with missing need parameter."""
        with pytest.raises(ValueError, match="'need' parameter is required"):
            discovery_tool._run()

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_discover_method(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test the discover convenience method."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = discovery_tool.discover(
            need="database",
            constraints=MCPDiscoveryConstraints(max_latency_ms=200),
            limit=3,
        )

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "total_found" in result
        assert "query_time_ms" in result
        assert len(result["recommendations"]) == 2
        assert result["total_found"] == 2

    @patch("crewai_tools.tools.mcp_discovery_tool.mcp_discovery_tool.requests.post")
    def test_discover_returns_structured_data(
        self,
        mock_post: MagicMock,
        discovery_tool: MCPDiscoveryTool,
        mock_api_response: dict,
    ) -> None:
        """Test that discover returns properly structured data."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = discovery_tool.discover(need="database")

        first_rec = result["recommendations"][0]
        assert "server" in first_rec
        assert "npm_package" in first_rec
        assert "install_command" in first_rec
        assert "confidence" in first_rec
        assert "description" in first_rec
        assert "capabilities" in first_rec
        assert "metrics" in first_rec
        assert "docs_url" in first_rec
        assert "github_url" in first_rec


class TestMCPDiscoveryToolIntegration:
    """Integration tests for MCPDiscoveryTool (requires network)."""

    @pytest.mark.skip(reason="Integration test - requires network access")
    def test_real_api_call(self) -> None:
        """Test actual API call to MCP Discovery service."""
        tool = MCPDiscoveryTool()
        result = tool._run(need="database", limit=3)
        assert "MCP server" in result or "No MCP servers found" in result
