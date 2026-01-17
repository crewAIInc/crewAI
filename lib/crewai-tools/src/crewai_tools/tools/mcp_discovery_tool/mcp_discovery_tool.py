"""MCP Discovery Tool for CrewAI agents.

This tool enables agents to dynamically discover MCP servers based on
natural language queries using the MCP Discovery API.
"""

import json
import logging
import os
from typing import Any, TypedDict

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)


class MCPServerMetrics(TypedDict, total=False):
    """Performance metrics for an MCP server."""

    avg_latency_ms: float | None
    uptime_pct: float | None
    last_checked: str | None


class MCPServerRecommendation(TypedDict, total=False):
    """A recommended MCP server from the discovery API."""

    server: str
    npm_package: str
    install_command: str
    confidence: float
    description: str
    capabilities: list[str]
    metrics: MCPServerMetrics
    docs_url: str
    github_url: str


class MCPDiscoveryResult(TypedDict):
    """Result from the MCP Discovery API."""

    recommendations: list[MCPServerRecommendation]
    total_found: int
    query_time_ms: int


class MCPDiscoveryConstraints(BaseModel):
    """Constraints for MCP server discovery."""

    max_latency_ms: int | None = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds",
    )
    required_features: list[str] | None = Field(
        default=None,
        description="List of required features/capabilities",
    )
    exclude_servers: list[str] | None = Field(
        default=None,
        description="List of server names to exclude from results",
    )


class MCPDiscoveryToolSchema(BaseModel):
    """Input schema for MCPDiscoveryTool."""

    need: str = Field(
        ...,
        description=(
            "Natural language description of what you need. "
            "For example: 'database with authentication', 'email automation', "
            "'file storage', 'web scraping'"
        ),
    )
    constraints: MCPDiscoveryConstraints | None = Field(
        default=None,
        description="Optional constraints to filter results",
    )
    limit: int = Field(
        default=5,
        description="Maximum number of recommendations to return (1-10)",
        ge=1,
        le=10,
    )


class MCPDiscoveryTool(BaseTool):
    """Tool for discovering MCP servers dynamically.

    This tool uses the MCP Discovery API to find MCP servers that match
    a natural language description of what the agent needs. It enables
    agents to dynamically discover and select the best MCP servers for
    their tasks without requiring pre-configuration.

    Example:
        ```python
        from crewai import Agent
        from crewai_tools import MCPDiscoveryTool

        discovery_tool = MCPDiscoveryTool()

        agent = Agent(
            role='Researcher',
            tools=[discovery_tool],
            goal='Research and analyze data'
        )

        # The agent can now discover MCP servers dynamically:
        # discover_mcp_server(need="database with authentication")
        # Returns: Supabase MCP server with installation instructions
        ```

    Attributes:
        name: The name of the tool.
        description: A description of what the tool does.
        args_schema: The Pydantic model for input validation.
        base_url: The base URL for the MCP Discovery API.
        timeout: Request timeout in seconds.
    """

    name: str = "Discover MCP Server"
    description: str = (
        "Discover MCP (Model Context Protocol) servers that match your needs. "
        "Use this tool to find the best MCP server for any task using natural "
        "language. Returns server recommendations with installation instructions, "
        "capabilities, and performance metrics."
    )
    args_schema: type[BaseModel] = MCPDiscoveryToolSchema
    base_url: str = "https://mcp-discovery-production.up.railway.app"
    timeout: int = 30
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MCP_DISCOVERY_API_KEY",
                description="API key for MCP Discovery (optional for free tier)",
                required=False,
            ),
        ]
    )

    def _build_request_payload(
        self,
        need: str,
        constraints: MCPDiscoveryConstraints | None,
        limit: int,
    ) -> dict[str, Any]:
        """Build the request payload for the discovery API.

        Args:
            need: Natural language description of what is needed.
            constraints: Optional constraints to filter results.
            limit: Maximum number of recommendations.

        Returns:
            Dictionary containing the request payload.
        """
        payload: dict[str, Any] = {
            "need": need,
            "limit": limit,
        }

        if constraints:
            constraints_dict: dict[str, Any] = {}
            if constraints.max_latency_ms is not None:
                constraints_dict["max_latency_ms"] = constraints.max_latency_ms
            if constraints.required_features:
                constraints_dict["required_features"] = constraints.required_features
            if constraints.exclude_servers:
                constraints_dict["exclude_servers"] = constraints.exclude_servers
            if constraints_dict:
                payload["constraints"] = constraints_dict

        return payload

    def _make_api_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a request to the MCP Discovery API.

        Args:
            payload: The request payload.

        Returns:
            The API response as a dictionary.

        Raises:
            ValueError: If the API returns an empty response.
            requests.exceptions.RequestException: If the request fails.
        """
        url = f"{self.base_url}/api/v1/discover"
        headers = {
            "Content-Type": "application/json",
        }

        api_key = os.environ.get("MCP_DISCOVERY_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = None
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                logger.error("Empty response from MCP Discovery API")
                raise ValueError("Empty response from MCP Discovery API")
            return results
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to MCP Discovery API: {e}"
            if response is not None and hasattr(response, "content"):
                error_msg += (
                    f"\nResponse content: "
                    f"{response.content.decode('utf-8', errors='replace')}"
                )
            logger.error(error_msg)
            raise
        except json.JSONDecodeError as e:
            if response is not None and hasattr(response, "content"):
                logger.error(f"Error decoding JSON response: {e}")
                logger.error(
                    f"Response content: "
                    f"{response.content.decode('utf-8', errors='replace')}"
                )
            else:
                logger.error(
                    f"Error decoding JSON response: {e} (No response content available)"
                )
            raise

    def _process_single_recommendation(
        self, rec: dict[str, Any]
    ) -> MCPServerRecommendation | None:
        """Process a single recommendation from the API.

        Args:
            rec: Raw recommendation dictionary from the API.

        Returns:
            Processed MCPServerRecommendation or None if malformed.
        """
        try:
            metrics_data = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            metrics: MCPServerMetrics = {
                "avg_latency_ms": metrics_data.get("avg_latency_ms"),
                "uptime_pct": metrics_data.get("uptime_pct"),
                "last_checked": metrics_data.get("last_checked"),
            }

            recommendation: MCPServerRecommendation = {
                "server": rec.get("server", ""),
                "npm_package": rec.get("npm_package", ""),
                "install_command": rec.get("install_command", ""),
                "confidence": rec.get("confidence", 0.0),
                "description": rec.get("description", ""),
                "capabilities": rec.get("capabilities", []),
                "metrics": metrics,
                "docs_url": rec.get("docs_url", ""),
                "github_url": rec.get("github_url", ""),
            }
            return recommendation
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Skipping malformed recommendation: {rec}, error: {e}")
            return None

    def _process_recommendations(
        self, recommendations: list[dict[str, Any]]
    ) -> list[MCPServerRecommendation]:
        """Process and validate server recommendations.

        Args:
            recommendations: Raw recommendations from the API.

        Returns:
            List of processed MCPServerRecommendation objects.
        """
        processed: list[MCPServerRecommendation] = []
        for rec in recommendations:
            result = self._process_single_recommendation(rec)
            if result is not None:
                processed.append(result)
        return processed

    def _format_result(self, result: MCPDiscoveryResult) -> str:
        """Format the discovery result as a human-readable string.

        Args:
            result: The discovery result to format.

        Returns:
            A formatted string representation of the result.
        """
        if not result["recommendations"]:
            return "No MCP servers found matching your requirements."

        lines = [
            f"Found {result['total_found']} MCP server(s) "
            f"(query took {result['query_time_ms']}ms):\n"
        ]

        for i, rec in enumerate(result["recommendations"], 1):
            confidence_pct = rec.get("confidence", 0) * 100
            lines.append(f"{i}. {rec.get('server', 'Unknown')} ({confidence_pct:.0f}% confidence)")
            lines.append(f"   Description: {rec.get('description', 'N/A')}")
            lines.append(f"   Capabilities: {', '.join(rec.get('capabilities', []))}")
            lines.append(f"   Install: {rec.get('install_command', 'N/A')}")
            lines.append(f"   NPM Package: {rec.get('npm_package', 'N/A')}")

            metrics = rec.get("metrics", {})
            if metrics.get("avg_latency_ms") is not None:
                lines.append(f"   Avg Latency: {metrics['avg_latency_ms']}ms")
            if metrics.get("uptime_pct") is not None:
                lines.append(f"   Uptime: {metrics['uptime_pct']}%")

            if rec.get("docs_url"):
                lines.append(f"   Docs: {rec['docs_url']}")
            if rec.get("github_url"):
                lines.append(f"   GitHub: {rec['github_url']}")
            lines.append("")

        return "\n".join(lines)

    def _run(self, **kwargs: Any) -> str:
        """Execute the MCP discovery operation.

        Args:
            **kwargs: Keyword arguments matching MCPDiscoveryToolSchema.

        Returns:
            A formatted string with discovered MCP servers.

        Raises:
            ValueError: If required parameters are missing.
        """
        need: str | None = kwargs.get("need")
        if not need:
            raise ValueError("'need' parameter is required")

        constraints_data = kwargs.get("constraints")
        constraints: MCPDiscoveryConstraints | None = None
        if constraints_data:
            if isinstance(constraints_data, dict):
                constraints = MCPDiscoveryConstraints(**constraints_data)
            elif isinstance(constraints_data, MCPDiscoveryConstraints):
                constraints = constraints_data

        limit: int = kwargs.get("limit", 5)

        payload = self._build_request_payload(need, constraints, limit)
        response = self._make_api_request(payload)

        recommendations = self._process_recommendations(
            response.get("recommendations", [])
        )

        result: MCPDiscoveryResult = {
            "recommendations": recommendations,
            "total_found": response.get("total_found", len(recommendations)),
            "query_time_ms": response.get("query_time_ms", 0),
        }

        return self._format_result(result)

    def discover(
        self,
        need: str,
        constraints: MCPDiscoveryConstraints | None = None,
        limit: int = 5,
    ) -> MCPDiscoveryResult:
        """Discover MCP servers matching the given requirements.

        This is a convenience method that returns structured data instead
        of a formatted string.

        Args:
            need: Natural language description of what is needed.
            constraints: Optional constraints to filter results.
            limit: Maximum number of recommendations (1-10).

        Returns:
            MCPDiscoveryResult containing server recommendations.

        Example:
            ```python
            tool = MCPDiscoveryTool()
            result = tool.discover(
                need="database with authentication",
                constraints=MCPDiscoveryConstraints(
                    max_latency_ms=200,
                    required_features=["auth", "realtime"]
                ),
                limit=3
            )
            for rec in result["recommendations"]:
                print(f"{rec['server']}: {rec['description']}")
            ```
        """
        payload = self._build_request_payload(need, constraints, limit)
        response = self._make_api_request(payload)

        recommendations = self._process_recommendations(
            response.get("recommendations", [])
        )

        return {
            "recommendations": recommendations,
            "total_found": response.get("total_found", len(recommendations)),
            "query_time_ms": response.get("query_time_ms", 0),
        }
