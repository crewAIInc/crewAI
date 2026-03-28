from __future__ import annotations

from typing import Any, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.adapters.mcp_adapter import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection


_PROXY_ACTIONS = ("get_proxy", "rotate_ip", "check_status")
_BROWSER_ACTIONS = (
    "browser_create",
    "browser_go",
    "browser_click",
    "browser_type",
    "browser_see",
    "browser_wait",
    "browser_extract",
    "browser_save",
    "browser_profile_list",
    "browser_profile_delete",
    "browser_end",
)


class ProxiesSxProxyToolSchema(BaseModel):
    action: Literal["get_proxy", "rotate_ip", "check_status"] = Field(
        default="get_proxy",
        description="Proxy action to execute via the Proxies.sx MCP server.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments forwarded as-is to the selected action.",
    )


class ProxiesSxBrowserToolSchema(BaseModel):
    action: Literal[
        "browser_create",
        "browser_go",
        "browser_click",
        "browser_type",
        "browser_see",
        "browser_wait",
        "browser_extract",
        "browser_save",
        "browser_profile_list",
        "browser_profile_delete",
        "browser_end",
    ] = Field(
        default="browser_create",
        description="Browser action to execute via the Proxies.sx browser MCP server.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments forwarded as-is to the selected action.",
    )


class _ProxiesSxMCPToolBase(BaseTool):
    mcp_command: str = "npx"
    mcp_args: list[str] = Field(default_factory=list)
    connect_timeout: int = 30
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["mcp", "mcpadapt"]
    )

    _adapter: MCPServerAdapter | None = PrivateAttr(default=None)
    _tools: ToolCollection[BaseTool] | None = PrivateAttr(default=None)

    def _supported_actions(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _build_server_params(self) -> Any:
        try:
            from mcp import StdioServerParameters

            return StdioServerParameters(command=self.mcp_command, args=self.mcp_args)
        except Exception:
            return {"command": self.mcp_command, "args": self.mcp_args}

    def _get_tools(self) -> ToolCollection[BaseTool]:
        if self._tools is None:
            self._adapter = MCPServerAdapter(
                self._build_server_params(),
                *self._supported_actions(),
                connect_timeout=self.connect_timeout,
            )
            self._tools = self._adapter.tools
        return self._tools

    def _execute_action(self, action: str, arguments: dict[str, Any]) -> Any:
        tools = self._get_tools()
        try:
            tool = tools[action]
        except KeyError as error:
            available_actions = ", ".join(tool.name for tool in tools)
            raise ValueError(
                f"Action '{action}' is not available from the Proxies.sx MCP server. "
                f"Available actions: {available_actions or 'none'}"
            ) from error
        return tool.run(**arguments)

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.stop()
            self._adapter = None
            self._tools = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class ProxiesSxProxyTool(_ProxiesSxMCPToolBase):
    name: str = "Proxies.sx Mobile Proxy Tool"
    description: str = (
        "Use Proxies.sx real 4G/5G mobile proxy actions via MCP. "
        "Supported actions: get_proxy, rotate_ip, check_status."
    )
    args_schema: type[BaseModel] = ProxiesSxProxyToolSchema
    country: str | None = Field(
        default=None,
        description="Optional default country used when action='get_proxy'.",
    )
    mcp_args: list[str] = Field(default_factory=lambda: ["@proxies-sx/mcp-server"])

    def _supported_actions(self) -> tuple[str, ...]:
        return _PROXY_ACTIONS

    def _run(
        self,
        action: Literal["get_proxy", "rotate_ip", "check_status"] = "get_proxy",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        run_args = dict(arguments or {})
        if action == "get_proxy" and self.country and "country" not in run_args:
            run_args["country"] = self.country
        return self._execute_action(action=action, arguments=run_args)


class ProxiesSxBrowserTool(_ProxiesSxMCPToolBase):
    name: str = "Proxies.sx Browser MCP Tool"
    description: str = (
        "Use Proxies.sx antidetect browser actions via MCP. "
        "Supports browser_create, browser_go, browser_click, browser_type, "
        "browser_see, browser_wait, browser_extract, browser_save, "
        "browser_profile_list, browser_profile_delete, and browser_end."
    )
    args_schema: type[BaseModel] = ProxiesSxBrowserToolSchema
    mcp_args: list[str] = Field(default_factory=lambda: ["@proxies-sx/browser-mcp"])

    def _supported_actions(self) -> tuple[str, ...]:
        return _BROWSER_ACTIONS

    def _run(
        self,
        action: Literal[
            "browser_create",
            "browser_go",
            "browser_click",
            "browser_type",
            "browser_see",
            "browser_wait",
            "browser_extract",
            "browser_save",
            "browser_profile_list",
            "browser_profile_delete",
            "browser_end",
        ] = "browser_create",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        return self._execute_action(action=action, arguments=dict(arguments or {}))
