"""Merge Agent Handler tools wrapper for CrewAI."""

import json
import logging
from typing import Any
from uuid import uuid4

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, create_model
import requests
import typing_extensions as te


logger = logging.getLogger(__name__)


class MergeAgentHandlerToolError(Exception):
    """Base exception for Merge Agent Handler tool errors."""



class MergeAgentHandlerTool(BaseTool):
    """
    Wrapper for Merge Agent Handler tools.

    This tool allows CrewAI agents to execute tools from Merge Agent Handler,
    which provides secure access to third-party integrations via the Model Context Protocol (MCP).

    Agent Handler manages authentication, permissions, and monitoring of all tool interactions.
    """

    tool_pack_id: str = Field(
        ..., description="UUID of the Agent Handler Tool Pack to use"
    )
    registered_user_id: str = Field(
        ..., description="UUID or origin_id of the registered user"
    )
    tool_name: str = Field(..., description="Name of the specific tool to execute")
    base_url: str = Field(
        default="https://ah-api.merge.dev",
        description="Base URL for Agent Handler API",
    )
    session_id: str | None = Field(
        default=None, description="MCP session ID (generated if not provided)"
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AGENT_HANDLER_API_KEY",
                description="Production API key for Agent Handler services",
                required=True,
            ),
        ]
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize session ID if not provided."""
        super().model_post_init(__context)
        if self.session_id is None:
            self.session_id = str(uuid4())

    def _get_api_key(self) -> str:
        """Get the API key from environment variables."""
        import os

        api_key = os.environ.get("AGENT_HANDLER_API_KEY")
        if not api_key:
            raise MergeAgentHandlerToolError(
                "AGENT_HANDLER_API_KEY environment variable is required. "
                "Set it with: export AGENT_HANDLER_API_KEY='your-key-here'"
            )
        return api_key

    def _make_mcp_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a JSON-RPC 2.0 MCP request to Agent Handler."""
        url = f"{self.base_url}/api/v1/tool-packs/{self.tool_pack_id}/registered-users/{self.registered_user_id}/mcp"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_api_key()}",
            "Mcp-Session-Id": self.session_id or str(uuid4()),
        }

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": str(uuid4()),
        }

        if params:
            payload["params"] = params

        # Log the full payload for debugging
        logger.debug(f"MCP Request to {url}: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            # Handle JSON-RPC error responses
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                error_code = result["error"].get("code", -1)
                logger.error(
                    f"Agent Handler API error (code {error_code}): {error_msg}"
                )
                raise MergeAgentHandlerToolError(f"API Error: {error_msg}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call Agent Handler API: {e!s}")
            raise MergeAgentHandlerToolError(
                f"Failed to communicate with Agent Handler API: {e!s}"
            ) from e

    def _run(self, **kwargs: Any) -> Any:
        """Execute the Agent Handler tool with the given arguments."""
        try:
            # Log what we're about to send
            logger.info(f"Executing {self.tool_name} with arguments: {kwargs}")

            # Make the tool call via MCP
            result = self._make_mcp_request(
                method="tools/call",
                params={"name": self.tool_name, "arguments": kwargs},
            )

            # Extract the actual result from the MCP response
            if "result" in result and "content" in result["result"]:
                content = result["result"]["content"]
                if content and len(content) > 0:
                    # Parse the text content (it's JSON-encoded)
                    text_content = content[0].get("text", "")
                    try:
                        return json.loads(text_content)
                    except json.JSONDecodeError:
                        return text_content

            return result

        except MergeAgentHandlerToolError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing tool {self.tool_name}: {e!s}")
            raise MergeAgentHandlerToolError(f"Tool execution failed: {e!s}") from e

    @classmethod
    def from_tool_name(
        cls,
        tool_name: str,
        tool_pack_id: str,
        registered_user_id: str,
        base_url: str = "https://ah-api.merge.dev",
        **kwargs: Any,
    ) -> te.Self:
        """
        Create a MergeAgentHandlerTool from a tool name.

        Args:
            tool_name: Name of the tool (e.g., "linear__create_issue")
            tool_pack_id: UUID of the Tool Pack
            registered_user_id: UUID of the registered user
            base_url: Base URL for Agent Handler API (defaults to production)
            **kwargs: Additional arguments to pass to the tool

        Returns:
            MergeAgentHandlerTool instance ready to use

        Example:
            >>> tool = MergeAgentHandlerTool.from_tool_name(
            ...     tool_name="linear__create_issue",
            ...     tool_pack_id="134e0111-0f67-44f6-98f0-597000290bb3",
            ...     registered_user_id="91b2b905-e866-40c8-8be2-efe53827a0aa"
            ... )
        """
        # Create an empty args schema model (proper BaseModel subclass)
        empty_args_schema = create_model(f"{tool_name.replace('__', '_').title()}Args")

        # Initialize session and get tool schema
        instance = cls(
            name=tool_name,
            description=f"Execute {tool_name} via Agent Handler",
            tool_pack_id=tool_pack_id,
            registered_user_id=registered_user_id,
            tool_name=tool_name,
            base_url=base_url,
            args_schema=empty_args_schema,  # Empty schema that properly inherits from BaseModel
            **kwargs,
        )

        # Try to fetch the actual tool schema from Agent Handler
        try:
            result = instance._make_mcp_request(method="tools/list")
            if "result" in result and "tools" in result["result"]:
                tools = result["result"]["tools"]
                tool_schema = next(
                    (t for t in tools if t.get("name") == tool_name), None
                )

                if tool_schema:
                    instance.description = tool_schema.get(
                        "description", instance.description
                    )

                    # Convert parameters schema to Pydantic model
                    if "parameters" in tool_schema:
                        try:
                            params = tool_schema["parameters"]
                            if params.get("type") == "object" and "properties" in params:
                                # Build field definitions for Pydantic
                                fields = {}
                                properties = params["properties"]
                                required = params.get("required", [])

                                for field_name, field_schema in properties.items():
                                    field_type = Any  # Default type
                                    field_default = ...  # Required by default

                                    # Map JSON schema types to Python types
                                    json_type = field_schema.get("type", "string")
                                    if json_type == "string":
                                        field_type = str
                                    elif json_type == "integer":
                                        field_type = int
                                    elif json_type == "number":
                                        field_type = float
                                    elif json_type == "boolean":
                                        field_type = bool
                                    elif json_type == "array":
                                        field_type = list[Any]
                                    elif json_type == "object":
                                        field_type = dict[str, Any]

                                    # Make field optional if not required
                                    if field_name not in required:
                                        field_type = field_type | None
                                        field_default = None

                                    field_description = field_schema.get("description")
                                    if field_description:
                                        fields[field_name] = (
                                            field_type,
                                            Field(
                                                default=field_default,
                                                description=field_description,
                                            ),
                                        )
                                    else:
                                        fields[field_name] = (field_type, field_default)

                                # Create the Pydantic model
                                if fields:
                                    args_schema = create_model(
                                        f"{tool_name.replace('__', '_').title()}Args",
                                        **fields,
                                    )
                                    instance.args_schema = args_schema

                        except Exception as e:
                            logger.warning(
                                f"Failed to create args schema for {tool_name}: {e!s}"
                            )

        except Exception as e:
            logger.warning(
                f"Failed to fetch tool schema for {tool_name}, using defaults: {e!s}"
            )

        return instance

    @classmethod
    def from_tool_pack(
        cls,
        tool_pack_id: str,
        registered_user_id: str,
        tool_names: list[str] | None = None,
        base_url: str = "https://ah-api.merge.dev",
        **kwargs: Any,
    ) -> list[te.Self]:
        """
        Create multiple MergeAgentHandlerTool instances from a Tool Pack.

        Args:
            tool_pack_id: UUID of the Tool Pack
            registered_user_id: UUID or origin_id of the registered user
            tool_names: Optional list of specific tool names to load. If None, loads all tools.
            base_url: Base URL for Agent Handler API (defaults to production)
            **kwargs: Additional arguments to pass to each tool

        Returns:
            List of MergeAgentHandlerTool instances

        Example:
            >>> tools = MergeAgentHandlerTool.from_tool_pack(
            ...     tool_pack_id="134e0111-0f67-44f6-98f0-597000290bb3",
            ...     registered_user_id="91b2b905-e866-40c8-8be2-efe53827a0aa",
            ...     tool_names=["linear__create_issue", "linear__get_issues"]
            ... )
        """
        # Create a temporary instance to fetch the tool list
        temp_instance = cls(
            name="temp",
            description="temp",
            tool_pack_id=tool_pack_id,
            registered_user_id=registered_user_id,
            tool_name="temp",
            base_url=base_url,
            args_schema=BaseModel,
        )

        try:
            # Fetch available tools
            result = temp_instance._make_mcp_request(method="tools/list")

            if "result" not in result or "tools" not in result["result"]:
                raise MergeAgentHandlerToolError(
                    "Failed to fetch tools from Agent Handler Tool Pack"
                )

            available_tools = result["result"]["tools"]

            # Filter tools if specific names were requested
            if tool_names:
                available_tools = [
                    t for t in available_tools if t.get("name") in tool_names
                ]

                # Check if all requested tools were found
                found_names = {t.get("name") for t in available_tools}
                missing_names = set(tool_names) - found_names
                if missing_names:
                    logger.warning(
                        f"The following tools were not found in the Tool Pack: {missing_names}"
                    )

            # Create tool instances
            tools = []
            for tool_schema in available_tools:
                tool_name = tool_schema.get("name")
                if not tool_name:
                    continue

                tool = cls.from_tool_name(
                    tool_name=tool_name,
                    tool_pack_id=tool_pack_id,
                    registered_user_id=registered_user_id,
                    base_url=base_url,
                    **kwargs,
                )
                tools.append(tool)

            return tools

        except MergeAgentHandlerToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to create tools from Tool Pack: {e!s}")
            raise MergeAgentHandlerToolError(f"Failed to load Tool Pack: {e!s}") from e
