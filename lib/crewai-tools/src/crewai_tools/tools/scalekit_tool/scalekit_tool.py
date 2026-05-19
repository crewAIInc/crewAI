"""Scalekit AgentKit tools wrapper for CrewAI.

Scalekit provides OAuth-based tool execution across 100+ connectors and
3,000+ tools (Gmail, Slack, GitHub, Notion, Salesforce, etc.). This
wrapper converts Scalekit tools into CrewAI ``BaseTool`` instances so
agents can call third-party APIs on behalf of authenticated users.
"""

import os
import typing as t

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel as PydanticBaseModel, Field, create_model
import typing_extensions as te


class ScalekitTool(BaseTool):
    """Wrapper for Scalekit AgentKit tools."""

    scalekit_action: t.Callable[..., t.Any]
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SCALEKIT_ENV_URL",
                description="Scalekit environment URL",
                required=True,
            ),
            EnvVar(
                name="SCALEKIT_CLIENT_ID",
                description="Scalekit client ID",
                required=True,
            ),
            EnvVar(
                name="SCALEKIT_CLIENT_SECRET",
                description="Scalekit client secret",
                required=True,
            ),
        ]
    )

    def _run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Run the Scalekit tool action with given arguments."""
        return self.scalekit_action(**kwargs)

    @staticmethod
    def _get_client() -> t.Any:
        """Create a ScalekitClient from environment variables."""
        from scalekit import ScalekitClient

        env_url = os.environ.get("SCALEKIT_ENV_URL")
        client_id = os.environ.get("SCALEKIT_CLIENT_ID")
        client_secret = os.environ.get("SCALEKIT_CLIENT_SECRET")

        if not all([env_url, client_id, client_secret]):
            raise ValueError(
                "Missing Scalekit credentials. Set SCALEKIT_ENV_URL, "
                "SCALEKIT_CLIENT_ID, and SCALEKIT_CLIENT_SECRET environment "
                "variables. Get these from Scalekit Dashboard \u2192 Developers \u2192 "
                "API Credentials."
            )

        return ScalekitClient(env_url, client_id, client_secret)

    @classmethod
    def from_tool(
        cls,
        tool_name: str,
        identifier: str,
        connected_account_id: t.Optional[str] = None,
        connection_name: t.Optional[str] = None,
        **kwargs: t.Any,
    ) -> te.Self:
        """Wrap a single Scalekit tool as a CrewAI tool.

        Args:
            tool_name: The Scalekit tool name, e.g. ``"gmail_create_draft"``.
            identifier: The connected account identifier (e.g. user email or
                an opaque user id).
            connected_account_id: Optional connected account ID. When omitted,
                the first matching account for the identifier is used.
            connection_name: Optional connection name for account resolution.
            **kwargs: Additional arguments forwarded to ``BaseTool``.

        Returns:
            A ``ScalekitTool`` instance ready for use with a CrewAI agent.
        """
        from scalekit.v1.tools.tools_pb2 import ScopedToolFilter
        from scalekit.actions.frameworks.util import extract_tool_metadata

        client = cls._get_client()

        try:
            result_tuple = client.tools.list_scoped_tools(
                identifier,
                filter=ScopedToolFilter(tool_names=[tool_name]),
            )
        except Exception as exc:
            raise ValueError(
                f"Tool '{tool_name}' not found for identifier "
                f"'{identifier}': {exc}"
            ) from exc
        response = result_tuple[0]

        target_tool = None
        target_ca_id = connected_account_id
        for scoped_tool in response.tools:
            name, _, _ = extract_tool_metadata(scoped_tool.tool)
            if name == tool_name:
                target_tool = scoped_tool.tool
                if not target_ca_id:
                    target_ca_id = scoped_tool.connected_account_id
                break

        if target_tool is None:
            available = [
                extract_tool_metadata(st.tool)[0] for st in response.tools
            ]
            raise ValueError(
                f"Tool '{tool_name}' not found for identifier "
                f"'{identifier}'. Available: {available[:10]}"
            )

        name, description, definition_dict = extract_tool_metadata(target_tool)
        input_schema = definition_dict.get("input_schema", {})
        args_schema = _json_schema_to_pydantic(name, input_schema)

        def execute(**arguments: t.Any) -> str:
            resp = client.actions.execute_tool(
                tool_input=arguments,
                tool_name=name,
                identifier=identifier,
                connected_account_id=target_ca_id,
                connection_name=connection_name,
            )
            result_data = resp.data if hasattr(resp, "data") else {}
            result_dict = dict(result_data) if result_data else {}
            execution_id = getattr(resp, "execution_id", None)
            if execution_id:
                result_dict["execution_id"] = execution_id
            return (
                str(result_dict)
                if result_dict
                else f"Tool {name} executed successfully"
            )

        execute.__name__ = name
        execute.__doc__ = description

        return cls(
            name=name,
            description=description,
            args_schema=args_schema,
            scalekit_action=execute,
            **kwargs,
        )

    @classmethod
    def from_connection(
        cls,
        *connection_names: str,
        identifier: str,
        providers: t.Optional[list[str]] = None,
        tool_names: t.Optional[list[str]] = None,
        **kwargs: t.Any,
    ) -> list[te.Self]:
        """Create tools for all actions available on the given connections.

        Args:
            *connection_names: Connection names as shown in the Scalekit
                dashboard (e.g. ``"gmail"``, ``"slack"``, ``"github-prod"``).
            identifier: The connected account identifier.
            providers: Optional provider name filter (e.g. ``["google"]``).
            tool_names: Optional list of specific tool names to include.
            **kwargs: Additional arguments forwarded to ``BaseTool``.

        Returns:
            A list of ``ScalekitTool`` instances, one per discovered tool.
        """
        if not connection_names and not providers and not tool_names:
            raise ValueError(
                "Provide at least one connection name, provider, or tool_names"
            )

        from scalekit.v1.tools.tools_pb2 import ScopedToolFilter
        from scalekit.actions.frameworks.util import extract_tool_metadata

        client = cls._get_client()

        scoped_filter = ScopedToolFilter(
            providers=list(providers or []),
            tool_names=list(tool_names or []),
            connection_names=list(connection_names),
        )

        result_tuple = client.tools.list_scoped_tools(
            identifier, filter=scoped_filter
        )
        response = result_tuple[0]

        tools: list[te.Self] = []
        for scoped_tool in response.tools:
            tool = scoped_tool.tool
            ca_id = scoped_tool.connected_account_id

            name, description, definition_dict = extract_tool_metadata(tool)
            input_schema = definition_dict.get("input_schema", {})
            args_schema = _json_schema_to_pydantic(name, input_schema)

            def _make_executor(
                t_name: str, t_ca_id: str, t_desc: str
            ) -> t.Callable[..., str]:
                def execute(**arguments: t.Any) -> str:
                    resp = client.actions.execute_tool(
                        tool_input=arguments,
                        tool_name=t_name,
                        identifier=identifier,
                        connected_account_id=t_ca_id,
                    )
                    result_data = resp.data if hasattr(resp, "data") else {}
                    result_dict = dict(result_data) if result_data else {}
                    execution_id = getattr(resp, "execution_id", None)
                    if execution_id:
                        result_dict["execution_id"] = execution_id
                    return (
                        str(result_dict)
                        if result_dict
                        else f"Tool {t_name} executed successfully"
                    )

                execute.__name__ = t_name
                execute.__doc__ = t_desc
                return execute

            tools.append(
                cls(
                    name=name,
                    description=description,
                    args_schema=args_schema,
                    scalekit_action=_make_executor(name, ca_id, description),
                    **kwargs,
                )
            )

        return tools


def _json_schema_to_pydantic(
    tool_name: str, schema: dict[str, t.Any]
) -> type[PydanticBaseModel]:
    """Convert a JSON Schema dict to a Pydantic model for ``args_schema``."""
    properties = schema.get("properties", {})
    required_list = schema.get("required", [])
    required = set(required_list) if isinstance(required_list, list) else set()

    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, t.Any] = {}
    for prop_name, prop_spec in properties.items():
        if not isinstance(prop_spec, dict):
            continue
        prop_type = prop_spec.get("type", "string")
        description = prop_spec.get("description", "")

        # Handle union types like ["string", "null"]
        if isinstance(prop_type, list):
            prop_type = next((pt for pt in prop_type if pt != "null"), "string")

        python_type = type_map.get(prop_type, t.Any)

        if prop_name in required:
            fields[prop_name] = (python_type, Field(description=description))
        else:
            fields[prop_name] = (
                t.Optional[python_type],
                Field(default=None, description=description),
            )

    safe_name = (
        tool_name.replace(".", "_").replace("-", "_").title().replace("_", "")
    )
    return (
        create_model(f"{safe_name}Schema", **fields)
        if fields
        else create_model(f"{safe_name}Schema")
    )
