"""Webz.io news search tool backed by the hosted News Search MCP server."""

from __future__ import annotations

import logging
import os
import threading
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from crewai_tools.adapters.mcp_adapter import MCPServerAdapter


logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://news-search-mcp.webz.io/mcp"
MCP_TRANSPORT = "streamable-http"
MCP_TOOL_NAME = "news_search_by_webz"
AUTH_ENV_VAR = "WEBZ_API_TOKEN"
MCP_URL_ENV_VAR = "WEBZ_MCP_URL"


class WebzioNewsSearchToolSchema(BaseModel):
    """Fallback arguments, used until the live MCP schema has been fetched.

    Extra keys are allowed so that server-side filters still reach Webz.io when
    the tool is running on this fallback; the MCP server validates them.
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(..., description="The news search query string.")


class WebzioNewsSearchTool(BaseTool):
    """Search global news coverage with Webz.io over its hosted MCP server.

    The argument schema is fetched from the MCP server when the tool is built,
    so filters that Webz.io adds server-side become available to the agent
    without a crewai-tools release. Construction does not fail when the server
    is unreachable: the tool keeps :class:`WebzioNewsSearchToolSchema` and
    retries the connection on the first call, which is where a missing token or
    an unreachable server is reported.

    The MCP endpoint must be an absolute ``https://`` URL so the Bearer token
    is not sent in cleartext.

    The MCP session stays open for reuse. Close it with :meth:`stop`, or use the
    tool as a context manager.

    Attributes:
        api_token: The Webz.io API token.
        mcp_url: The News Search MCP endpoint.
        connect_timeout: Seconds allowed for the MCP connection.
    """

    name: str = "Webzio News Search"
    description: str = (
        "Search global news articles and blog posts with Webz.io. Returns "
        "matching articles with title, url, publication date, source, language "
        "and text. Supports filtering by language, country, published date, "
        "sentiment, site and more; pass only the filters you were asked for."
    )
    args_schema: type[BaseModel] = WebzioNewsSearchToolSchema
    api_token: str | None = Field(
        default_factory=lambda: os.getenv(AUTH_ENV_VAR),
        description=(
            "Webz.io API token. Falls back to the WEBZ_API_TOKEN environment "
            "variable when not provided."
        ),
        json_schema_extra={"required": False},
    )
    mcp_url: str = Field(
        default_factory=lambda: os.getenv(MCP_URL_ENV_VAR) or DEFAULT_MCP_URL,
        description=(
            "News Search MCP endpoint. Falls back to the WEBZ_MCP_URL "
            "environment variable, then to the hosted endpoint."
        ),
        json_schema_extra={"required": False},
    )
    connect_timeout: int = Field(
        default=30,
        description="Seconds to wait for the MCP server connection.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["mcp", "mcpadapt"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=AUTH_ENV_VAR,
                description="API token for the Webz.io News API",
                required=True,
            ),
            EnvVar(
                name=MCP_URL_ENV_VAR,
                description="Override for the Webz.io News Search MCP endpoint",
                required=False,
            ),
        ]
    )

    _adapter: MCPServerAdapter | None = PrivateAttr(default=None)
    _mcp_tool: BaseTool | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **kwargs: Any) -> None:
        """Build the tool and try to adopt the live MCP argument schema.

        Args:
            **kwargs: Field overrides, such as ``api_token`` or ``mcp_url``.
        """
        super().__init__(**kwargs)
        try:
            self._connect()
        except Exception as e:
            logger.debug(f"Deferring the Webz.io MCP connection: {e}")

    def _connect(self) -> BaseTool:
        """Open the MCP session, if needed, and adopt the live argument schema.

        Returns:
            The MCP-backed tool that runs the search.

        Raises:
            ValueError: If the API token is missing, the endpoint is not HTTPS,
                or if the server does not expose the news search tool.
        """
        with self._lock:
            if self._mcp_tool is not None:
                return self._mcp_tool

            token = (self.api_token or "").strip()
            if not token:
                raise ValueError(
                    f"Webz.io API token is missing. Set {AUTH_ENV_VAR} or pass "
                    f"api_token=... to {type(self).__name__}."
                )

            url = self.mcp_url.strip().rstrip("/")
            parsed_url = urlsplit(url)
            if parsed_url.scheme != "https" or not parsed_url.netloc:
                raise ValueError(
                    f"The Webz.io MCP endpoint must be an absolute https:// URL, "
                    f"got {url!r}."
                )

            adapter = MCPServerAdapter(
                {
                    "url": url,
                    "transport": MCP_TRANSPORT,
                    "headers": {"Authorization": f"Bearer {token}"},
                },
                MCP_TOOL_NAME,
                connect_timeout=self.connect_timeout,
            )
            try:
                mcp_tools = list(adapter.tools)
                if not mcp_tools:
                    raise ValueError(
                        f"The MCP server at {url} did not expose a "
                        f"{MCP_TOOL_NAME} tool."
                    )
            except Exception:
                adapter.stop()
                raise

            self._adapter = adapter
            self._mcp_tool = mcp_tools[0]
            self.args_schema = self._mcp_tool.args_schema
            return self._mcp_tool

    def _run(self, **kwargs: Any) -> str:
        """Run a news search against Webz.io.

        Args:
            **kwargs: Search arguments accepted by the live MCP schema, always
                including ``query``.

        Returns:
            The search results as returned by the MCP server.
        """
        return str(self._connect()._run(**kwargs))

    def stop(self) -> None:
        """Close the MCP session. A later call reconnects."""
        with self._lock:
            adapter, self._adapter, self._mcp_tool = self._adapter, None, None
        if adapter is not None:
            adapter.stop()

    def __enter__(self) -> WebzioNewsSearchTool:
        """Return the tool itself; the MCP session is already open.

        Returns:
            This tool.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the MCP session on leaving the context.

        Args:
            exc_type: The exception type raised in the block, if any.
            exc_value: The exception raised in the block, if any.
            traceback: The traceback of that exception, if any.
        """
        self.stop()
