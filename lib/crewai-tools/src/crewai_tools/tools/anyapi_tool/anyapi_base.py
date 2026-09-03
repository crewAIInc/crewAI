import os
from typing import Any, cast

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr


ANYAPI_BASE_URL = "https://api.getanyapi.com"

INSTALL_HINT = (
    "Missing optional dependency 'getanyapi'. Install it with:\n"
    "  pip install getanyapi\n"
    "or\n"
    "  uv add crewai-tools --extra getanyapi\n"
)

MISSING_KEY_HINT = (
    "An AnyAPI key is required. Pass api_key=... or set the ANYAPI_API_KEY environment "
    "variable. Create a key at https://getanyapi.com/dashboard; new accounts start with "
    "free trial credit."
)


def import_getanyapi() -> Any:
    """Import the AnyAPI SDK lazily so the base install stays light."""
    try:
        import getanyapi
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return getanyapi


class AnyApiToolBase(BaseTool):
    """Credential handling and client wiring shared by the AnyAPI tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = ANYAPI_BASE_URL
    package_dependencies: list[str] = Field(default=["getanyapi"])
    env_vars: list[EnvVar] = Field(
        default=[
            EnvVar(
                name="ANYAPI_API_KEY",
                description="API key for AnyAPI, created at https://getanyapi.com/dashboard",
                required=True,
            ),
        ]
    )

    # The key is never a model field: CrewAI serializes a tool's fields into
    # checkpoints and telemetry, and a secret must not travel with them. Only the
    # SDK client below holds it.
    _anyapi: Any = PrivateAttr(default=None)
    _client: Any = PrivateAttr(default=None)

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._anyapi = import_getanyapi()

        key = api_key or os.getenv("ANYAPI_API_KEY")
        if not key:
            raise ValueError(MISSING_KEY_HINT)

        self._client = self._anyapi.AnyAPI(api_key=key, base_url=self.base_url)

    def _as_json(self, payload: Any) -> str:
        """Serialize an SDK response the way the AnyAPI wire publishes it."""
        return cast(str, payload.model_dump_json(by_alias=True, exclude_none=True))
