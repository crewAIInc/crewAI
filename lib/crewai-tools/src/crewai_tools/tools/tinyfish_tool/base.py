"""Shared base class for TinyFish tools."""

from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field, PrivateAttr


try:
    from tinyfish import TinyFish

    _TINYFISH_AVAILABLE = True
    _TINYFISH_IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised in install-failure tests
    TinyFish = Any  # type: ignore[misc,assignment]
    _TINYFISH_AVAILABLE = False
    _TINYFISH_IMPORT_ERROR = str(exc)


_TINYFISH_API_KEY_ENV = "TINYFISH_API_KEY"
_TINYFISH_INTEGRATION_ENV = "TF_API_INTEGRATION"
_TINYFISH_INTEGRATION_TAG = "crewai-tools"


class TinyfishToolBase(BaseTool):
    """Shared configuration and client management for TinyFish tools.

    Concrete tools subclass this and implement ``_run`` using the
    cached client returned by :meth:`_get_client`.
    """

    api_key: str | None = Field(
        default_factory=lambda: os.getenv(_TINYFISH_API_KEY_ENV),
        description="TinyFish API key (overrides env var if provided).",
        json_schema_extra={"required": False},
    )

    package_dependencies: list[str] = Field(default_factory=lambda: ["tinyfish"])

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=_TINYFISH_API_KEY_ENV,
                description=(
                    "TinyFish Web Agent API key. "
                    "Get one at https://agent.tinyfish.ai/api-keys"
                ),
                required=True,
            ),
        ]
    )

    _client: TinyFish | None = PrivateAttr(default=None)

    def _get_client(self) -> tuple[TinyFish | None, str | None]:
        """Return (client, None) on success or (None, error_message) on failure."""
        if not _TINYFISH_AVAILABLE:
            return None, (
                "Error: the 'tinyfish' Python SDK is not installed. "
                "Install with `pip install tinyfish` (or "
                "`uv add crewai-tools --extra tinyfish`). "
                f"(Import error: {_TINYFISH_IMPORT_ERROR})"
            )
        if self._client is not None:
            return self._client, None
        key = self.api_key or os.environ.get(_TINYFISH_API_KEY_ENV)
        if not key:
            return None, (
                "Error: TINYFISH_API_KEY is not set. "
                "Pass api_key when instantiating the tool or set the "
                "TINYFISH_API_KEY environment variable. "
                "Get a key at https://agent.tinyfish.ai/api-keys"
            )
        os.environ.setdefault(_TINYFISH_INTEGRATION_ENV, _TINYFISH_INTEGRATION_TAG)
        self._client = TinyFish(api_key=key)
        return self._client, None

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return f"Error: {type(exc).__name__}: {exc}"
