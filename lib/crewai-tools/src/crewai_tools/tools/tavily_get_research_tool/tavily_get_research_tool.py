from __future__ import annotations

import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


load_dotenv()
try:
    from tavily import AsyncTavilyClient, TavilyClient  # type: ignore[import-untyped]

    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class TavilyGetResearchToolSchema(BaseModel):
    """Input schema for TavilyGetResearchTool."""

    request_id: str = Field(
        ...,
        description="Existing Tavily research request ID to fetch status and results for.",
    )


class TavilyGetResearchTool(BaseTool):
    """Tool that uses the Tavily Research status endpoint to retrieve results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _client: Any | None = PrivateAttr(default=None)
    _async_client: Any | None = PrivateAttr(default=None)
    name: str = "Tavily Get Research"
    description: str = (
        "A tool that retrieves the status and results of an existing Tavily "
        "research task by request ID. It returns Tavily responses as JSON."
    )
    args_schema: type[BaseModel] = TavilyGetResearchToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["tavily-python"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TAVILY_API_KEY",
                description="API key for Tavily research service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if TAVILY_AVAILABLE:
            api_key = os.getenv("TAVILY_API_KEY")
            self._client = TavilyClient(api_key=api_key)
            self._async_client = AsyncTavilyClient(api_key=api_key)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'tavily-python' package is required. 'click' and "
                    "'subprocess' are also needed to assist with installation "
                    "if the package is missing. Please install 'tavily-python' "
                    "manually (e.g., 'pip install tavily-python') and ensure "
                    "'click' and 'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'tavily-python' package, which is required "
                "for TavilyGetResearchTool. Would you like to install it?"
            ):
                try:
                    subprocess.run(["uv", "add", "tavily-python"], check=True)  # noqa: S607
                    raise ImportError(
                        "'tavily-python' has been installed. Please restart your "
                        "Python application to use the TavilyGetResearchTool."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'tavily-python' but failed: {e}. "
                        "Please install it manually to use the TavilyGetResearchTool."
                    ) from e
            else:
                raise ImportError(
                    "The 'tavily-python' package is required to use the "
                    "TavilyGetResearchTool. Please install it with: uv add tavily-python"
                )

    @staticmethod
    def _stringify_response(response: Any) -> str:
        if isinstance(response, str):
            return response
        return json.dumps(response, indent=2)

    def _run(self, request_id: str) -> str:
        """Synchronously retrieves Tavily research task status and results."""
        if not self._client:
            raise ValueError(
                "Tavily client is not initialized. Ensure 'tavily-python' is "
                "installed and API key is set."
            )

        return self._stringify_response(self._client.get_research(request_id))

    async def _arun(self, request_id: str) -> str:
        """Asynchronously retrieves Tavily research task status and results."""
        if not self._async_client:
            raise ValueError(
                "Tavily async client is not initialized. Ensure 'tavily-python' is "
                "installed and API key is set."
            )

        return self._stringify_response(
            await self._async_client.get_research(request_id)
        )
