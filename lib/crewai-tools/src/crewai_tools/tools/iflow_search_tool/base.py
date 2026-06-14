from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field


try:
    from iflow_search import IFlowSearchClient  # type: ignore[import-untyped]

    IFLOW_AVAILABLE = True
except ImportError:
    IFLOW_AVAILABLE = False

_INTEGRATION_NAME = "crewai-tools"


class IFlowSearchToolBase(BaseTool):
    """Shared plumbing for the iFlow Search tool suite.

    Wraps the ``iflow-search`` SDK client. Subclasses implement ``_run`` by
    calling :meth:`_get_client` and normalizing the typed response into a
    JSON string for the agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any | None = Field(
        default=None,
        description="Optional pre-built IFlowSearchClient (useful for testing or custom HTTP setups).",
    )
    api_key: str | None = Field(
        default=None,
        description="iFlow Search API key. Falls back to the IFLOW_API_KEY environment variable.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional override for the iFlow API base URL.",
    )
    timeout: float | None = Field(
        default=None,
        description="Request timeout in seconds (SDK default when unset).",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["iflow-search"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="IFLOW_API_KEY",
                description="API key for the iFlow Search (心流搜索) service",
                required=True,
            ),
        ]
    )

    def _get_client(self) -> Any:
        """Return the injected client or lazily build one from configuration.

        Raises:
            ImportError: when the ``iflow-search`` package is not installed.
            ValueError: when no API key is available.
        """
        if self.client is not None:
            return self.client
        api_key = self.api_key or os.getenv("IFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "iFlow Search API key not found. Set the IFLOW_API_KEY environment "
                "variable or pass api_key= when constructing the tool."
            )
        if not IFLOW_AVAILABLE:
            raise ImportError(
                "The 'iflow-search' package is required to use the iFlow Search tools. "
                "Install with: pip install 'crewai-tools[iflow-search]' "
                "(or pip install iflow-search)."
            )
        self.client = IFlowSearchClient(
            api_key=api_key,
            source="crewai",
            integration_name=_INTEGRATION_NAME,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self.client
