from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr


logger = logging.getLogger(__name__)


class DaytonaBaseTool(BaseTool):
    """Shared base for tools that act on a Daytona sandbox.

    Lifecycle modes:
      - persistent=False (default): create a fresh sandbox per `_run` call and
        delete it when the call returns. Safer and stateless — nothing leaks if
        the agent forgets cleanup.
      - persistent=True: lazily create a single sandbox on first use, cache it
        on the instance, and register an atexit hook to delete it at process
        exit. Cheaper across many calls and lets files/state carry over.
      - sandbox_id=<existing>: attach to a sandbox the caller already owns.
        Never deleted by the tool.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_dependencies: list[str] = Field(default_factory=lambda: ["daytona"])

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("DAYTONA_API_KEY"),
        description="Daytona API key. Falls back to DAYTONA_API_KEY env var.",
        json_schema_extra={"required": False},
    )
    api_url: str | None = Field(
        default_factory=lambda: os.getenv("DAYTONA_API_URL"),
        description="Daytona API URL override. Falls back to DAYTONA_API_URL env var.",
        json_schema_extra={"required": False},
    )
    target: str | None = Field(
        default_factory=lambda: os.getenv("DAYTONA_TARGET"),
        description="Daytona target region. Falls back to DAYTONA_TARGET env var.",
        json_schema_extra={"required": False},
    )

    persistent: bool = Field(
        default=False,
        description=(
            "If True, reuse one sandbox across all calls to this tool instance "
            "and delete it at process exit. Default False creates and deletes a "
            "fresh sandbox per call."
        ),
    )
    sandbox_id: str | None = Field(
        default=None,
        description=(
            "Attach to an existing sandbox by id or name instead of creating a "
            "new one. The tool will never delete a sandbox it did not create."
        ),
    )
    create_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional kwargs forwarded to CreateSandboxFromSnapshotParams when "
            "creating a sandbox (e.g. language, snapshot, env_vars, labels)."
        ),
    )
    sandbox_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for sandbox create/delete operations.",
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="DAYTONA_API_KEY",
                description="API key for Daytona sandbox service",
                required=False,
            ),
            EnvVar(
                name="DAYTONA_API_URL",
                description="Daytona API base URL (optional)",
                required=False,
            ),
            EnvVar(
                name="DAYTONA_TARGET",
                description="Daytona target region (optional)",
                required=False,
            ),
        ]
    )

    _client: Any | None = PrivateAttr(default=None)
    _persistent_sandbox: Any | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cleanup_registered: bool = PrivateAttr(default=False)

    _sdk_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _import_sdk(cls) -> dict[str, Any]:
        if cls._sdk_cache:
            return cls._sdk_cache
        try:
            from daytona import (
                CreateSandboxFromSnapshotParams,
                Daytona,
                DaytonaConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'daytona' package is required for Daytona sandbox tools. "
                "Install it with: uv add daytona  (or) pip install daytona"
            ) from exc
        cls._sdk_cache = {
            "Daytona": Daytona,
            "DaytonaConfig": DaytonaConfig,
            "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
        }
        return cls._sdk_cache

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        sdk = self._import_sdk()
        config_kwargs: dict[str, Any] = {}
        if self.api_key:
            config_kwargs["api_key"] = self.api_key
        if self.api_url:
            config_kwargs["api_url"] = self.api_url
        if self.target:
            config_kwargs["target"] = self.target
        config = sdk["DaytonaConfig"](**config_kwargs) if config_kwargs else None
        self._client = sdk["Daytona"](config) if config else sdk["Daytona"]()
        return self._client

    def _build_create_params(self) -> Any | None:
        if not self.create_params:
            return None
        sdk = self._import_sdk()
        return sdk["CreateSandboxFromSnapshotParams"](**self.create_params)

    def _acquire_sandbox(self) -> tuple[Any, bool]:
        """Return (sandbox, should_delete_after_use)."""
        client = self._get_client()

        if self.sandbox_id:
            return client.get(self.sandbox_id), False

        if self.persistent:
            with self._lock:
                if self._persistent_sandbox is None:
                    self._persistent_sandbox = client.create(
                        self._build_create_params(),
                        timeout=self.sandbox_timeout,
                    )
                    if not self._cleanup_registered:
                        atexit.register(self.close)
                        self._cleanup_registered = True
                return self._persistent_sandbox, False

        sandbox = client.create(
            self._build_create_params(),
            timeout=self.sandbox_timeout,
        )
        return sandbox, True

    def _release_sandbox(self, sandbox: Any, should_delete: bool) -> None:
        if not should_delete:
            return
        try:
            sandbox.delete(timeout=self.sandbox_timeout)
        except Exception:
            logger.debug(
                "Best-effort sandbox cleanup failed after ephemeral use; "
                "the sandbox may need manual deletion.",
                exc_info=True,
            )

    def close(self) -> None:
        """Delete the cached persistent sandbox if one exists."""
        with self._lock:
            sandbox = self._persistent_sandbox
            self._persistent_sandbox = None
        if sandbox is None:
            return
        try:
            sandbox.delete(timeout=self.sandbox_timeout)
        except Exception:
            logger.debug(
                "Best-effort persistent sandbox cleanup failed at close(); "
                "the sandbox may need manual deletion.",
                exc_info=True,
            )
