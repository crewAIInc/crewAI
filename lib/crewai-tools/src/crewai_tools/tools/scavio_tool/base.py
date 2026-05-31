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

    Handles client initialization, API key resolution, and shared
    utilities for all Scavio tools.
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
        ge=1,
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
        if not SCAVIO_AVAILABLE:
            raise ImportError(
                "The 'scavio' package is required to use Scavio tools. "
                "Install it with: pip install scavio"
            )
        if not self.api_key:
            raise ValueError(
                "A Scavio API key is required. Provide it via the "
                "'api_key' parameter or set the SCAVIO_API_KEY "
                "environment variable."
            )
        self.client = ScavioClient(api_key=self.api_key)
        self.async_client = AsyncScavioClient(api_key=self.api_key)

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
