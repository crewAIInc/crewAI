"""Typed tools for the hosted Baizhi Cloud Agent Toolkit MCP service."""

from contextvars import ContextVar
import ipaddress
import json
import logging
import os
import re
from typing import Any, ClassVar, Literal

import anyio
from crewai.tools import BaseTool, EnvVar
import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self


_ACTIVE_KEY: ContextVar[str | None] = ContextVar("crewai_baizhi_key", default=None)


class _SDKLogFilter(logging.Filter):
    """Sanitize SDK diagnostics only in the current Baizhi call context."""

    def filter(self, record: logging.LogRecord) -> bool:
        key = _ACTIVE_KEY.get()
        if key:
            record.msg = record.getMessage().replace(key, "[REDACTED]")
            record.args = ()
            # SDK validation tracebacks can contain complete remote responses.
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


for _logger_name in ("mcp.client.streamable_http", "client"):
    logging.getLogger(_logger_name).addFilter(_SDKLogFilter())


class _Input(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True
    )


class BaizhiSearchFilter(_Input):
    """Restrict sources with bare domain names or IP addresses, not URLs."""

    domains: list[str] | None = Field(
        default=None, description="Include bare domains or IPs."
    )
    exclude_domains: list[str] | None = Field(
        default=None, description="Exclude bare domains or IPs."
    )

    @field_validator("domains", "exclude_domains")
    @classmethod
    def validate_domains(cls, values: list[str] | None) -> list[str] | None:
        for value in values or []:
            try:
                ipaddress.ip_address(value)
                continue
            except ValueError:
                pass
            try:
                domain = value.encode("idna").decode("ascii")
            except UnicodeError:
                raise ValueError("Use bare domains or IPs, not URLs.") from None
            if (
                len(domain) > 253
                or "." not in domain
                or not all(
                    re.fullmatch(r"(?!-)[a-zA-Z0-9-]{1,63}(?<!-)", label)
                    for label in domain.split(".")
                )
            ):
                raise ValueError("Use bare domains or IPs, not URLs.")
        return values


class _PageInput(_Input):
    url: AnyHttpUrl = Field(description="Public HTTP(S) page, without credentials.")

    @field_validator("url")
    @classmethod
    def reject_url_credentials(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.username is not None or value.password is not None:
            raise ValueError("Do not send URLs with embedded credentials.")
        return value


class BaizhiSearchInput(_Input):
    """Search public web pages."""

    query: str = Field(
        min_length=1, description="Search terms; use filter for site restrictions."
    )
    count: int = Field(default=10, ge=1, le=50, description="Maximum result count.")
    filter: BaizhiSearchFilter | None = Field(
        default=None, description="Optional source restrictions."
    )
    need_summary: bool = Field(
        default=False, description="Return and record a summary for each result."
    )
    time_range: Literal["day", "week", "month", "year"] = "month"


class BaizhiScrapeInput(_PageInput):
    """Read one HTTP or HTTPS page."""

    return_format: Literal["markdown", "json"] = "markdown"
    accept_language: str | None = Field(
        default=None, description="Preferred page language, for example en-US."
    )
    download: Literal[False] = Field(
        default=False,
        description="Downloads are disabled for this research tool.",
    )


class BaizhiExtractInput(_PageInput):
    """Extract fields or follow an extraction instruction for one page."""

    fields: dict[str, Literal["string", "number", "boolean", "array"]] | None = Field(
        default=None, description="Field names mapped to output types."
    )
    instruction: str | None = Field(
        default=None,
        min_length=1,
        description="What to extract; required when fields is empty.",
    )
    accept_language: str | None = Field(
        default=None, description="Preferred page language."
    )
    download: Literal[False] = Field(
        default=False,
        description="Downloads are disabled for this research tool.",
    )

    @model_validator(mode="after")
    def require_extraction_target(self) -> Self:
        if not self.fields and not self.instruction:
            raise ValueError("Provide fields or instruction.")
        return self


class _BaizhiTool(BaseTool):
    """One bounded MCP call, with credentials outside the agent's arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True, hide_input_in_errors=True)
    _mcp_name: ClassVar[str]
    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(value) if (value := os.getenv("BAIZHI_API_KEY")) else None
        ),
        validate_default=True,
        exclude=True,
        repr=False,
        description="Baizhi Cloud API key; defaults to BAIZHI_API_KEY.",
    )
    timeout: float = Field(
        default=60,
        gt=0,
        allow_inf_nan=False,
        description="Total call deadline in seconds, including initialization.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BAIZHI_API_KEY",
                description="Your Baizhi Cloud API key.",
                required=True,
            )
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["mcp>=1.28.1,<2"])

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and (
            not value.get_secret_value().strip()
            or any(c.isspace() for c in value.get_secret_value())
        ):
            raise ValueError(
                "Provide a non-empty Baizhi Cloud API key without whitespace or a Bearer prefix."
            )
        return value

    def _redact(self, value: Any) -> Any:
        key = self.api_key.get_secret_value() if self.api_key else ""
        if isinstance(value, str):
            return value.replace(key, "[REDACTED]") if key else value
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {
                self._redact(name): self._redact(item) for name, item in value.items()
            }
        return value

    async def _run(self, **kwargs: Any) -> str:
        """Execute via the SDK; CrewAI also supports this coroutine in run()."""
        arguments = self.args_schema.model_validate(kwargs).model_dump(
            mode="json", exclude_none=True
        )
        if self.api_key is None:
            raise ValueError(
                "Set BAIZHI_API_KEY or provide api_key when constructing the tool."
            )
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            from mcp.types import TextContent
        except ImportError:
            raise ImportError(
                "Baizhi Cloud tools require MCP support. Install it with: uv add 'crewai-tools[mcp]'"
            ) from None
        log_context = _ACTIVE_KEY.set(self.api_key.get_secret_value())
        try:
            with anyio.fail_after(self.timeout):
                async with httpx.AsyncClient(
                    headers={
                        "Authorization": f"Bearer {self.api_key.get_secret_value()}"
                    },
                    timeout=self.timeout,
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    async with streamable_http_client(
                        "https://agent-toolkit.app.baizhi.cloud/mcp", http_client=client
                    ) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(self._mcp_name, arguments)
        except TimeoutError:
            raise TimeoutError(
                "Baizhi Cloud tool call timed out; its server-side outcome may be unknown."
            ) from None
        except Exception:
            # Do not expose provider bodies, request headers, or credentials in traces.
            raise RuntimeError(
                "Baizhi Cloud MCP request failed. Check credentials, credits, and service availability."
            ) from None
        finally:
            _ACTIVE_KEY.reset(log_context)
        if result.isError:
            raise RuntimeError(
                "Baizhi Cloud returned a tool error. Check the inputs and account credits."
            )
        if result.structuredContent is not None:
            return json.dumps(
                self._redact(result.structuredContent), ensure_ascii=False
            )
        if any(not isinstance(block, TextContent) for block in result.content):
            raise RuntimeError("Baizhi Cloud returned unsupported non-text content.")
        return "\n".join(
            self._redact(block.text)
            for block in result.content
            if isinstance(block, TextContent)
        )

    async def _arun(self, **kwargs: Any) -> str:
        """Support BaseTool.arun without a worker thread."""
        return await self._run(**kwargs)


class BaizhiSearchTool(_BaizhiTool):
    """Search the public web through Baizhi Cloud."""

    name: str = "Baizhi Web Search"
    description: str = "Search public web pages through Baizhi Cloud. Sends the query to the hosted service and may consume paid credits."
    args_schema: type[BaseModel] = BaizhiSearchInput
    _mcp_name: ClassVar[str] = "websearch_search"


class BaizhiScrapeTool(_BaizhiTool):
    """Read a web page through Baizhi Cloud."""

    name: str = "Baizhi Web Scrape"
    description: str = "Read the body of one HTTP or HTTPS page through Baizhi Cloud. Sends the URL to the hosted service and may consume paid credits."
    args_schema: type[BaseModel] = BaizhiScrapeInput
    _mcp_name: ClassVar[str] = "web_scrape"


class BaizhiExtractTool(_BaizhiTool):
    """Extract structured page information through Baizhi Cloud."""

    name: str = "Baizhi Web Extract"
    description: str = "Extract named fields or follow an extraction instruction for one web page through Baizhi Cloud. Sends inputs to the hosted service and may consume paid credits."
    args_schema: type[BaseModel] = BaizhiExtractInput
    _mcp_name: ClassVar[str] = "web_extract"
