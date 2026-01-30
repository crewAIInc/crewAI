"""MCPServer for CrewAI."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
import logging
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp import StdioServerParameters
    import mcp.types
    from mcpadapt.core import MCPAdapt, ToolAdapter


try:
    from typing import ForwardRef, Union

    import jsonref
    from mcp import StdioServerParameters
    import mcp.types
    from mcpadapt.core import MCPAdapt, ToolAdapter
    from pydantic import BaseModel, Field, create_model

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


JSON_TYPE_MAPPING: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _resolve_refs_to_dict(obj: Any) -> Any:
    """Recursively convert JsonRef objects to regular dicts."""
    if isinstance(obj, dict):
        return {k: _resolve_refs_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs_to_dict(item) for item in obj]
    return obj


def _resolve_all_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve all $ref references in a JSON schema using jsonref.

    This function fully resolves all JSON Schema $ref references, including
    internal references that point to paths within the schema itself
    (e.g., '#/properties/geometry/anyOf/0/items').

    Args:
        schema: The JSON schema with potential $ref references.

    Returns:
        A new schema dict with all $ref references resolved and inlined.
    """
    if not MCP_AVAILABLE:
        return schema

    resolved = jsonref.replace_refs(schema, lazy_load=False)
    result = _resolve_refs_to_dict(resolved)
    if "$defs" in result:
        del result["$defs"]
    return result


def _create_model_from_schema(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> type[BaseModel]:
    """Create a Pydantic model from a JSON schema definition.

    This is a simplified version that handles common JSON schema patterns
    without passing problematic extra fields to Pydantic's Field().

    Args:
        schema: The JSON schema definition.
        model_name: The name for the created model.

    Returns:
        A Pydantic BaseModel class.
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP dependencies not available")

    created_models: dict[str, type[BaseModel]] = {}
    forward_refs: dict[str, ForwardRef] = {}

    def process_schema(name: str, schema_def: dict[str, Any]) -> type[BaseModel]:
        if name in created_models:
            return created_models[name]

        if name not in forward_refs:
            forward_refs[name] = ForwardRef(name)

        fields: dict[str, Any] = {}
        properties = schema_def.get("properties", {})
        required = set(schema_def.get("required", []))

        for field_name, field_schema in properties.items():
            field_type, default = get_field_type(field_name, field_schema, required)
            fields[field_name] = (
                field_type,
                Field(
                    default=default,
                    description=field_schema.get("description", ""),
                ),
            )

        model: type[BaseModel] = create_model(
            schema_def.get("title", name),
            __doc__=schema_def.get("description", ""),
            **fields,
        )

        created_models[name] = model
        return model

    def get_field_type(
        field_name: str, field_schema: dict[str, Any], required: set[str]
    ) -> tuple[Any, Any]:
        if "$ref" in field_schema:
            ref_parts = field_schema["$ref"].lstrip("#/").split("/")
            ref_name = ref_parts[-1]

            if ref_name not in created_models:
                ref_schema = schema
                for part in ref_parts:
                    ref_schema = ref_schema.get(part, {})
                process_schema(ref_name, ref_schema)

            field_type = created_models[ref_name]
            is_required = field_name in required
            return (
                field_type | None if not is_required else field_type,
                None if not is_required else ...,
            )

        if "anyOf" in field_schema:
            is_nullable = any(
                opt.get("type") == "null" for opt in field_schema["anyOf"]
            )
            types: list[type[Any]] = []

            for option in field_schema["anyOf"]:
                if "type" in option and option["type"] != "null":
                    types.append(JSON_TYPE_MAPPING.get(option["type"], Any))
                elif "enum" in option:
                    types.append(str)
                elif "$ref" in option:
                    ref_parts = option["$ref"].lstrip("#/").split("/")
                    ref_name = ref_parts[-1]

                    if ref_name not in created_models:
                        ref_schema = schema
                        for part in ref_parts:
                            ref_schema = ref_schema.get(part, {})
                        process_schema(ref_name, ref_schema)

                    types.append(created_models[ref_name])

            if len(types) == 0:
                field_type = Any
            elif len(types) == 1:
                field_type = types[0]
            else:
                field_type = Union[tuple(types)]  # noqa: UP007

            default = field_schema.get("default")
            is_required = field_name in required and default is None

            if is_nullable and not is_required:
                field_type = field_type | None

            return field_type, ... if is_required else default

        if field_schema.get("type") == "array" and "items" in field_schema:
            item_type, _ = get_field_type("item", field_schema["items"], set())
            field_type = list[item_type]
        else:
            json_type = field_schema.get("type", "string")

            if isinstance(json_type, list):
                types = []
                for t in json_type:
                    if t != "null":
                        mapped_type = JSON_TYPE_MAPPING.get(t, Any)
                        types.append(mapped_type)

                if len(types) == 0:
                    field_type = Any
                elif len(types) == 1:
                    field_type = types[0]
                else:
                    field_type = Union[tuple(types)]  # noqa: UP007
            else:
                field_type = JSON_TYPE_MAPPING.get(json_type, Any)

        default = field_schema.get("default")
        is_required = field_name in required and default is None

        if not is_required:
            field_type = field_type | None
            default = default if default is not None else None
        else:
            default = ...

        return field_type, default

    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            process_schema(def_name, def_schema)

    return process_schema(model_name, schema)


class CrewAIAdapterWithSchemaFix(ToolAdapter):
    """Custom CrewAI adapter that properly handles complex JSON schemas.

    This adapter extends mcpadapt's ToolAdapter to fix issues with complex
    JSON schemas that contain internal $ref references (e.g., Mapbox MCP server).
    It fully resolves all $ref references before creating Pydantic models,
    preventing KeyError exceptions during JSON schema generation.
    """

    def adapt(
        self,
        func: Callable[[dict[str, Any] | None], mcp.types.CallToolResult],
        mcp_tool: mcp.types.Tool,
    ) -> BaseTool:
        """Adapt a MCP tool to a CrewAI tool with proper schema handling.

        Args:
            func: The function to adapt.
            mcp_tool: The MCP tool to adapt.

        Returns:
            A CrewAI tool.
        """
        resolved_schema = _resolve_all_refs(mcp_tool.inputSchema)
        tool_input_model = _create_model_from_schema(resolved_schema)

        class CrewAIMCPTool(BaseTool):
            name: str = mcp_tool.name
            description: str = mcp_tool.description or ""
            args_schema: type[BaseModel] = tool_input_model

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                filtered_kwargs: dict[str, Any] = {}
                schema_properties = resolved_schema.get("properties", {})

                for key, value in kwargs.items():
                    if value is None and key in schema_properties:
                        prop_schema = schema_properties[key]
                        if isinstance(prop_schema.get("type"), list):
                            if "null" in prop_schema["type"]:
                                filtered_kwargs[key] = value
                        elif "anyOf" in prop_schema:
                            if any(
                                opt.get("type") == "null"
                                for opt in prop_schema["anyOf"]
                            ):
                                filtered_kwargs[key] = value
                    else:
                        filtered_kwargs[key] = value

                result = func(filtered_kwargs)
                return (
                    result.content[0].text
                    if len(result.content) == 1
                    else str(
                        [
                            content.text
                            for content in result.content
                            if hasattr(content, "text")
                        ]
                    )
                )

            def _generate_description(self) -> None:
                try:
                    args_schema = {
                        k: v
                        for k, v in jsonref.replace_refs(
                            self.args_schema.model_json_schema()
                        ).items()
                        if k != "$defs"
                    }
                except Exception:
                    args_schema = resolved_schema
                self.description = f"Tool Name: {self.name}\nTool Arguments: {args_schema}\nTool Description: {self.description}"

        return CrewAIMCPTool()

    async def async_adapt(
        self,
        afunc: Callable[
            [dict[str, Any] | None], Coroutine[Any, Any, mcp.types.CallToolResult]
        ],
        mcp_tool: mcp.types.Tool,
    ) -> Any:
        raise NotImplementedError("async is not supported by the CrewAI framework.")


class MCPServerAdapter:
    """Manages the lifecycle of an MCP server and make its tools available to CrewAI.

    Note: tools can only be accessed after the server has been started with the
        `start()` method.

    Attributes:
        tools: The CrewAI tools available from the MCP server.

    Usage:
        # context manager + stdio
        with MCPServerAdapter(...) as tools:
            # tools is now available

        # context manager + sse
        with MCPServerAdapter({"url": "http://localhost:8000/sse"}) as tools:
            # tools is now available

        # context manager with filtered tools
        with MCPServerAdapter(..., "tool1", "tool2") as filtered_tools:
            # only tool1 and tool2 are available

        # context manager with custom connect timeout (60 seconds)
        with MCPServerAdapter(..., connect_timeout=60) as tools:
            # tools is now available with longer timeout

        # manually stop mcp server
        try:
            mcp_server = MCPServerAdapter(...)
            tools = mcp_server.tools  # all tools

            # or with filtered tools and custom timeout
            mcp_server = MCPServerAdapter(..., "tool1", "tool2", connect_timeout=45)
            filtered_tools = mcp_server.tools  # only tool1 and tool2
            ...
        finally:
            mcp_server.stop()

        # Best practice is ensure cleanup is done after use.
        mcp_server.stop() # run after crew().kickoff()
    """

    def __init__(
        self,
        serverparams: StdioServerParameters | dict[str, Any],
        *tool_names: str,
        connect_timeout: int = 30,
    ) -> None:
        """Initialize the MCP Server.

        Args:
            serverparams: The parameters for the MCP server it supports either a
                `StdioServerParameters` or a `dict` respectively for STDIO and SSE.
            *tool_names: Optional names of tools to filter. If provided, only tools with
                matching names will be available.
            connect_timeout: Connection timeout in seconds to the MCP server (default is 30s).

        """
        super().__init__()
        self._adapter = None
        self._tools = None
        self._tool_names = list(tool_names) if tool_names else None

        if not MCP_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'mcp' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "mcp crewai-tools[mcp]"], check=True)  # noqa: S607

                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install mcp package") from e
            else:
                raise ImportError(
                    "`mcp` package not found, please run `uv add crewai-tools[mcp]`"
                )

        try:
            self._serverparams = serverparams
            self._adapter = MCPAdapt(
                self._serverparams, CrewAIAdapterWithSchemaFix(), connect_timeout
            )
            self.start()

        except Exception as e:
            if self._adapter is not None:
                try:
                    self.stop()
                except Exception as stop_e:
                    logger.error(f"Error during stop cleanup: {stop_e}")
            raise RuntimeError(f"Failed to initialize MCP Adapter: {e}") from e

    def start(self):
        """Start the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()

    def stop(self):
        """Stop the MCP server."""
        self._adapter.__exit__(None, None, None)

    @property
    def tools(self) -> ToolCollection[BaseTool]:
        """The CrewAI tools available from the MCP server.

        Raises:
            ValueError: If the MCP server is not started.

        Returns:
            The CrewAI tools available from the MCP server.
        """
        if self._tools is None:
            raise ValueError(
                "MCP server not started, run `mcp_server.start()` first before accessing `tools`"
            )

        tools_collection = ToolCollection(self._tools)
        if self._tool_names:
            return tools_collection.filter_by_names(self._tool_names)
        return tools_collection

    def __enter__(self):
        """Enter the context manager. Note that `__init__()` already starts the MCP server.
        So tools should already be available.
        """
        return self.tools

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        return self._adapter.__exit__(exc_type, exc_value, traceback)
