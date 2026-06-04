from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from crewai_tools.security.safe_path import validate_url


try:
    from firecrawl import Firecrawl  # type: ignore[import-untyped]

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


class FirecrawlInteractToolSchema(BaseModel):
    prompt: str = Field(
        description="Natural-language description of the task for the Firecrawl agent to carry out by navigating and interacting with web pages"
    )
    urls: list[str] | None = Field(
        default=None,
        description="Optional list of URLs to start from or constrain the agent to",
    )


class FirecrawlInteractTool(BaseTool):
    """Tool for running an autonomous Firecrawl browser agent using the Firecrawl v2 API. To run this tool, you need to have a Firecrawl API key.

    The agent navigates and interacts with web pages to accomplish a natural-language
    task, then returns the result.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl v2 agent parameters.

    Default configuration options (Firecrawl v2 API):
        model (str): Agent model to use ("spark-1-mini" or "spark-1-pro"). Default: "spark-1-mini"
        max_credits (int): Maximum credits the agent may spend. Default: None (no cap)
        strict_constrain_to_urls (bool): Restrict the agent to the provided urls only. Default: None
        poll_interval (int): Seconds between status polls while the agent runs. Default: 2
        timeout (int): Overall timeout in seconds. Default: None
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web interact tool"
    description: str = (
        "Run an autonomous Firecrawl browser agent that navigates and interacts with "
        "web pages to accomplish a task, then returns the result"
    )
    args_schema: type[BaseModel] = FirecrawlInteractToolSchema
    api_key: str | None = None
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "spark-1-mini",
            "max_credits": None,
            "strict_constrain_to_urls": None,
            "poll_interval": 2,
            "timeout": None,
        }
    )

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
        try:
            from firecrawl import Firecrawl
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "firecrawl-py"], check=True)  # noqa: S607
                from firecrawl import (
                    Firecrawl,
                )
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                ) from None

        self._firecrawl = Firecrawl(api_key=api_key)

    def _run(self, prompt: str, urls: list[str] | None = None) -> Any:
        if not self._firecrawl:
            raise RuntimeError("Firecrawl client not properly initialized")

        validated_urls = [validate_url(u) for u in urls] if urls else None
        return self._firecrawl.agent(urls=validated_urls, prompt=prompt, **self.config)


try:
    from firecrawl import Firecrawl  # noqa: F401

    if not getattr(FirecrawlInteractTool, "_model_rebuilt", False):
        FirecrawlInteractTool.model_rebuild()
        FirecrawlInteractTool._model_rebuilt = True  # type: ignore[attr-defined]
except ImportError:
    pass
