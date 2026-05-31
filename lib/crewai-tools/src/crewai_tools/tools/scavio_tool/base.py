import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field

try:
    from scavio import AsyncScavioClient, ScavioClient

    SCAVIO_AVAILABLE = True
except ImportError:
    SCAVIO_AVAILABLE = False


class ScavioBaseTool(BaseTool):
    """Base class for Scavio API tools.

    Handles client initialization, API key resolution, and the
    missing-package auto-install prompt shared by all Scavio tools.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any | None = None
    async_client: Any | None = None
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SCAVIO_API_KEY"),
        description=(
            "The Scavio API key. If not provided, it will be loaded "
            "from the environment variable SCAVIO_API_KEY."
        ),
    )
    max_results: int = Field(
        default=5,
        description="The maximum number of results to return.",
    )

    package_dependencies: list[str] = Field(
        default_factory=lambda: ["scavio"],
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SCAVIO_API_KEY",
                description="API key for Scavio Search API",
                required=True,
            ),
        ],
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if SCAVIO_AVAILABLE:
            self.client = ScavioClient(api_key=self.api_key)
            self.async_client = AsyncScavioClient(api_key=self.api_key)
        else:
            try:
                import subprocess

                import click
            except ImportError as e:
                raise ImportError(
                    "The 'scavio' package is required. 'click' and 'subprocess' "
                    "are also needed to assist with installation if the package "
                    "is missing. Please install 'scavio' manually "
                    "(e.g., 'pip install scavio') and ensure 'click' and "
                    "'subprocess' are available."
                ) from e

            if click.confirm(
                "You are missing the 'scavio' package, which is required "
                "for Scavio tools. Would you like to install it?"
            ):
                try:
                    subprocess.run(  # noqa: S607, S603
                        ["uv", "add", "scavio"],
                        check=True,
                    )
                    raise ImportError(
                        "'scavio' has been installed. Please restart your "
                        "Python application to use Scavio tools."
                    )
                except subprocess.CalledProcessError as e:
                    raise ImportError(
                        f"Attempted to install 'scavio' but failed: {e}. "
                        f"Please install it manually to use Scavio tools."
                    ) from e
            else:
                raise ImportError(
                    "The 'scavio' package is required to use Scavio tools. "
                    "Please install it with: uv add scavio"
                )

    def _truncate_results(
        self, raw: dict[str, Any], key: str = "results"
    ) -> dict[str, Any]:
        """Truncate a list field in the response to max_results."""
        items = raw.get(key)
        if isinstance(items, list) and len(items) > self.max_results:
            raw[key] = items[: self.max_results]
        return raw

    def _format_response(self, raw: dict[str, Any]) -> str:
        """Serialize a response dict to a JSON string."""
        return json.dumps(raw, indent=2)
