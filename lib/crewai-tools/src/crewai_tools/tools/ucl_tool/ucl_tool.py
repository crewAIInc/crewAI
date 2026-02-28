"""UCL (Unified Context Layer) tools wrapper for crewAI."""

import typing as t

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, create_model
import typing_extensions as te


class UCLToolConfig(BaseModel):
    """Configuration for UCL Tool."""

    workspace_id: str = Field(..., description="UCL Workspace ID")
    api_key: str = Field(..., description="Workspace API Key")
    mcp_gateway_id: str = Field(..., description="Workspace MCP Gateway ID")
    base_url: str = Field(
        default="https://live.fastn.ai",
        description="Base URL for UCL API",
    )
    stage: str = Field(default="LIVE", description="API stage")


class UCLTool(BaseTool):
    """Wrapper for UCL (Unified Context Layer) tools."""

    ucl_action: t.Callable = Field(default=None, exclude=True)
    action_id: str = Field(default="", description="UCL Action ID")
    tool_name: str = Field(default="", description="UCL Tool Name")
    config: UCLToolConfig = Field(default=None, exclude=True)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="UCL_WORKSPACE_ID",
                description="UCL Workspace ID",
                required=True,
            ),
            EnvVar(
                name="UCL_API_KEY",
                description="UCL Workspace API Key",
                required=True,
            ),
            EnvVar(
                name="UCL_MCP_GATEWAY_ID",
                description="UCL MCP Gateway ID",
                required=True,
            ),
        ]
    )

    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Run the UCL action with given arguments."""
        if self.ucl_action is not None:
            return self.ucl_action(**kwargs)
        return {"error": "UCL action not configured"}

    @staticmethod
    def _get_headers(config: UCLToolConfig) -> dict[str, str]:
        """Get headers for UCL API requests."""
        return {
            "stage": config.stage,
            "x-fastn-space-id": config.workspace_id,
            "x-fastn-api-key": config.api_key,
            "x-fastn-space-agent-id": config.mcp_gateway_id,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _fetch_tools(
        config: UCLToolConfig,
        prompt: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, t.Any]]:
        """Fetch available tools from UCL API."""
        url = f"{config.base_url}/api/ucl/getTools"
        headers = UCLTool._get_headers(config)

        payload: dict[str, t.Any] = {"input": {"limit": limit}}
        if prompt:
            payload["input"]["prompt"] = prompt

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        return response.json()

    @staticmethod
    def _execute_tool(
        config: UCLToolConfig,
        action_id: str,
        tool_name: str,
        parameters: dict[str, t.Any],
    ) -> dict[str, t.Any]:
        """Execute a UCL tool."""
        url = f"{config.base_url}/api/ucl/executeTool"
        headers = UCLTool._get_headers(config)

        payload = {
            "input": {
                "actionId": action_id,
                "parameters": parameters,
                "toolName": tool_name,
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        if "body" in result:
            return result["body"]
        elif "rawBody" in result:
            return {"raw": result["rawBody"]}
        return result

    @staticmethod
    def _json_schema_to_pydantic_model(
        schema: dict[str, t.Any],
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """Convert JSON schema to Pydantic model."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        field_definitions: dict[str, t.Any] = {}

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for field_name, field_schema in properties.items():
            field_type = type_mapping.get(field_schema.get("type", "string"), str)
            field_description = field_schema.get("description", "")

            if field_name in required:
                field_definitions[field_name] = (
                    field_type,
                    Field(..., description=field_description),
                )
            else:
                field_definitions[field_name] = (
                    t.Optional[field_type],
                    Field(default=None, description=field_description),
                )

        return create_model(model_name, **field_definitions)

    @classmethod
    def from_config(
        cls,
        workspace_id: str,
        api_key: str,
        mcp_gateway_id: str,
        base_url: str = "https://live.fastn.ai",
        stage: str = "LIVE",
    ) -> UCLToolConfig:
        """Create a UCL configuration object."""
        return UCLToolConfig(
            workspace_id=workspace_id,
            api_key=api_key,
            mcp_gateway_id=mcp_gateway_id,
            base_url=base_url,
            stage=stage,
        )

    @classmethod
    def from_action(
        cls,
        config: UCLToolConfig,
        action_id: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> te.Self:
        """Create a UCLTool from a specific action."""

        def execute_action(**params: t.Any) -> dict[str, t.Any]:
            """Execute the UCL action."""
            return cls._execute_tool(
                config=config,
                action_id=action_id,
                tool_name=tool_name,
                parameters=params,
            )

        execute_action.__name__ = tool_name
        execute_action.__doc__ = description

        args_schema = cls._json_schema_to_pydantic_model(
            input_schema,
            model_name=f"{tool_name.replace('-', '_').replace(' ', '_')}Schema",
        )

        return cls(
            name=tool_name,
            description=description,
            args_schema=args_schema,
            ucl_action=execute_action,
            action_id=action_id,
            tool_name=tool_name,
            config=config,
            **kwargs,
        )

    @classmethod
    def get_tools(
        cls,
        config: UCLToolConfig,
        prompt: str | None = None,
        limit: int = 10,
        **kwargs: t.Any,
    ) -> list[te.Self]:
        """
        Fetch and create tools from UCL API.

        Args:
            config: UCL configuration object
            prompt: Optional prompt to filter tools by keywords
            limit: Maximum number of tools to fetch (default: 10)
            **kwargs: Additional arguments to pass to tool creation

        Returns:
            List of UCLTool instances
        """
        tools_data = cls._fetch_tools(config, prompt=prompt, limit=limit)
        tools: list[te.Self] = []

        for tool_data in tools_data:
            action_id = tool_data.get("actionId", "")
            function_data = tool_data.get("function", {})

            tool_name = function_data.get("name", "")
            description = function_data.get("description", "No description available")
            input_schema = function_data.get("inputSchema", {"properties": {}})

            if not tool_name or not action_id:
                continue

            tool = cls.from_action(
                config=config,
                action_id=action_id,
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                **kwargs,
            )
            tools.append(tool)

        return tools
