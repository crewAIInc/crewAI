from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr


logger = logging.getLogger(__name__)


class OpenSandboxBaseTool(BaseTool):
    """Shared base for tools that act on an Open Sandbox sandbox.

    Lifecycle modes:
      - persistent=False (default): create a fresh sandbox per `_run` call and
        kill it when the call returns. Safer and stateless — nothing leaks if
        the agent forgets cleanup.
      - persistent=True: lazily create a single sandbox on first use, cache it
        on the instance, and register an atexit hook to kill it at process
        exit. Cheaper across many calls and lets files/state carry over.
      - sandbox_id=<existing>: attach to a sandbox the caller already owns.
        Never killed by the tool.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_dependencies: list[str] = Field(default_factory=lambda: ["opensandbox"])

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPEN_SANDBOX_API_KEY"),
        description="Open Sandbox API key. Falls back to OPEN_SANDBOX_API_KEY env var.",
        json_schema_extra={"required": False},
    )
    domain: str | None = Field(
        default_factory=lambda: os.getenv("OPEN_SANDBOX_DOMAIN"),
        description=(
            "Open Sandbox management API domain (e.g. 'api.opensandbox.io'). "
            "Falls back to OPEN_SANDBOX_DOMAIN env var."
        ),
        json_schema_extra={"required": False},
    )
    protocol: str | None = Field(
        default=None,
        description="Protocol for the management API ('http' or 'https').",
        json_schema_extra={"required": False},
    )

    persistent: bool = Field(
        default=False,
        description=(
            "If True, reuse one sandbox across all calls to this tool instance "
            "and kill it at process exit. Default False creates and kills a "
            "fresh sandbox per call."
        ),
    )
    sandbox_id: str | None = Field(
        default=None,
        description=(
            "Attach to an existing sandbox by id instead of creating a new one. "
            "The tool will never kill a sandbox it did not create."
        ),
    )
    create_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional kwargs forwarded to SandboxSync.create when creating a "
            "sandbox (e.g. image, env, resource, metadata, entrypoint)."
        ),
    )
    sandbox_timeout: float = Field(
        default=60.0,
        description=(
            "Timeout in seconds to wait for sandbox readiness on create/connect."
        ),
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPEN_SANDBOX_API_KEY",
                description="API key for Open Sandbox service",
                required=False,
            ),
            EnvVar(
                name="OPEN_SANDBOX_DOMAIN",
                description="Open Sandbox management API domain (optional)",
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
            from opensandbox.config.connection_sync import ConnectionConfigSync
            from opensandbox.models.execd import RunCommandOpts
            from opensandbox.models.filesystem import SearchEntry, WriteEntry
            from opensandbox.sync.sandbox import SandboxSync
        except ImportError as exc:
            raise ImportError(
                "The 'opensandbox' package is required for Open Sandbox tools. "
                "Install it with: uv add opensandbox  (or) pip install opensandbox"
            ) from exc
        cls._sdk_cache = {
            "SandboxSync": SandboxSync,
            "ConnectionConfigSync": ConnectionConfigSync,
            "RunCommandOpts": RunCommandOpts,
            "WriteEntry": WriteEntry,
            "SearchEntry": SearchEntry,
        }
        return cls._sdk_cache

    def _get_client(self) -> Any:
        """Return a cached ConnectionConfigSync built from this tool's fields.

        Open Sandbox has no separate "client" object — connection settings are
        carried by ConnectionConfigSync and passed into SandboxSync.create /
        SandboxSync.connect. We cache one config per tool instance.
        """
        if self._client is not None:
            return self._client
        sdk = self._import_sdk()
        config_kwargs: dict[str, Any] = {}
        if self.api_key:
            config_kwargs["api_key"] = self.api_key
        if self.domain:
            config_kwargs["domain"] = self.domain
        if self.protocol:
            config_kwargs["protocol"] = self.protocol
        self._client = sdk["ConnectionConfigSync"](**config_kwargs)
        return self._client

    def _build_create_kwargs(self) -> dict[str, Any]:
        return dict(self.create_params) if self.create_params else {}

    def _acquire_sandbox(self) -> tuple[Any, bool]:
        """Return (sandbox, should_kill_after_use)."""
        sdk = self._import_sdk()
        config = self._get_client()

        if self.sandbox_id:
            sandbox = sdk["SandboxSync"].connect(
                self.sandbox_id,
                connection_config=config,
                connect_timeout=_seconds_to_timedelta(self.sandbox_timeout),
            )
            return sandbox, False

        if self.persistent:
            with self._lock:
                if self._persistent_sandbox is None:
                    self._persistent_sandbox = sdk["SandboxSync"].create(
                        connection_config=config,
                        ready_timeout=_seconds_to_timedelta(self.sandbox_timeout),
                        **self._build_create_kwargs(),
                    )
                    if not self._cleanup_registered:
                        atexit.register(self.close)
                        self._cleanup_registered = True
                return self._persistent_sandbox, False

        sandbox = sdk["SandboxSync"].create(
            connection_config=config,
            ready_timeout=_seconds_to_timedelta(self.sandbox_timeout),
            **self._build_create_kwargs(),
        )
        return sandbox, True

    def _release_sandbox(self, sandbox: Any, should_kill: bool) -> None:
        if not should_kill:
            return
        try:
            sandbox.kill()
        except Exception:
            logger.debug(
                "Best-effort sandbox kill failed after ephemeral use; "
                "the sandbox may need manual termination.",
                exc_info=True,
            )
        try:
            sandbox.close()
        except Exception:
            logger.debug(
                "Best-effort sandbox local-resource close failed after ephemeral use.",
                exc_info=True,
            )

    def close(self) -> None:
        """Kill the cached persistent sandbox if one exists."""
        with self._lock:
            sandbox = self._persistent_sandbox
            self._persistent_sandbox = None
        if sandbox is None:
            return
        try:
            sandbox.kill()
        except Exception:
            logger.debug(
                "Best-effort persistent sandbox kill failed at close(); "
                "the sandbox may need manual termination.",
                exc_info=True,
            )
        try:
            sandbox.close()
        except Exception:
            logger.debug(
                "Best-effort persistent sandbox local-resource close failed at close().",
                exc_info=True,
            )


def _seconds_to_timedelta(seconds: float) -> Any:
    from datetime import timedelta

    return timedelta(seconds=seconds)
