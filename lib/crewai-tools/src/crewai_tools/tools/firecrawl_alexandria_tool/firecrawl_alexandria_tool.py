from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


try:
    from firecrawl import Firecrawl  # type: ignore[import-untyped]

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


class FirecrawlAlexandriaToolSchema(BaseModel):
    provider: str = Field(
        description="Alexandria provider id, for example 'sec-gov', or 'firecrawl' for find-tools",
    )
    capability: str = Field(
        description="Capability of that provider, for example 'filings/company'",
    )
    options: str | None = Field(
        default=None,
        description='JSON-encoded object string with the capability inputs, for example {"ticker": "AAPL"}',
    )


class FirecrawlAlexandriaTool(BaseTool):
    """Tool for running Firecrawl Alexandria data provider capabilities.

    Alexandria exposes structured data providers (company filings, public
    records, finance, sports and more) behind the Firecrawl API. Each call runs
    one provider capability and returns its data. The `firecrawl` provider's
    `find-tools` capability searches the catalog and lists each capability's
    options, so the same tool covers discovery and execution. To run this tool,
    you need a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key. Defaults to the
            `FIRECRAWL_API_KEY` environment variable.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl Alexandria tool"
    description: str = (
        "Run a Firecrawl Alexandria data provider capability and return its "
        "structured data. To find a capability and its options, first call it "
        "with provider='firecrawl', capability='find-tools' and options "
        '\'{"query": "<what you need>", "level": "tools", "expand": ["options"], "limit": 5}\'. '
        "Then call the provider and capability it returns, with options as a "
        "JSON-encoded object string using the option names it lists."
    )
    args_schema: type[BaseModel] = FirecrawlAlexandriaToolSchema
    api_key: str | None = None
    _firecrawl: Any = PrivateAttr(None)
    package_dependencies: list[str] = Field(default_factory=lambda: ["firecrawl-py"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="FIRECRAWL_API_KEY",
                description="API key for Firecrawl services",
                required=True,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self._initialize_firecrawl()

    def _initialize_firecrawl(self) -> None:
        try:
            from firecrawl import Firecrawl

            self._firecrawl = Firecrawl(api_key=self.api_key)
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "firecrawl-py>=4.43.0"], check=True)  # noqa: S607
                    from firecrawl import Firecrawl

                    self._firecrawl = Firecrawl(api_key=self.api_key)
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install firecrawl-py package") from e
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add 'firecrawl-py>=4.43.0'`"
                ) from None

    def _run(
        self,
        provider: str,
        capability: str,
        options: str | None = None,
    ) -> Any:
        if not self._firecrawl:
            raise RuntimeError("Firecrawl client not properly initialized")

        try:
            parsed_options = json.loads(options) if options else {}
        except json.JSONDecodeError as e:
            return ToolFailure(
                message=f"options must be a JSON object: {e}",
                code="invalid_options",
            )
        if not isinstance(parsed_options, dict):
            return ToolFailure(
                message="options must be a JSON object.",
                code="invalid_options",
            )

        from firecrawl.v2.utils.error_handler import (  # type: ignore[import-untyped]
            FirecrawlError,
        )

        try:
            response = self._firecrawl.scrape(
                alexandria={
                    "provider": provider,
                    "capability": capability,
                    "options": parsed_options,
                },
                integration="_crewai",
            )
        except FirecrawlError as e:
            status = e.status_code
            return ToolFailure(
                message=f"Firecrawl Alexandria could not run {provider}/{capability}: {e}",
                code=e.code or "firecrawl_error",
                retryable=status is not None and (status == 429 or status >= 500),
            )
        if not response.alexandria:
            return ToolFailure(
                message=f"Firecrawl Alexandria returned no result for {provider}/{capability}.",
                code="empty_response",
            )

        result = response.alexandria[0]
        if result.error is not None:
            status = result.error.status
            return ToolFailure(
                message=f"Firecrawl Alexandria could not run {provider}/{capability}: {result.error.message}",
                code=result.error.code,
                retryable=status is not None and (status == 429 or status >= 500),
            )
        return result.data


try:
    from firecrawl import Firecrawl  # noqa: F401

    if not getattr(FirecrawlAlexandriaTool, "_model_rebuilt", False):
        FirecrawlAlexandriaTool.model_rebuild()
        FirecrawlAlexandriaTool._model_rebuilt = True  # type: ignore[attr-defined]
except ImportError:
    pass
