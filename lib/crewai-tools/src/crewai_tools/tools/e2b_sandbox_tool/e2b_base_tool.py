from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr


logger = logging.getLogger(__name__)


class E2BBaseTool(BaseTool):
    """Shared base for tools that act on an E2B sandbox.

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

    package_dependencies: list[str] = Field(default_factory=lambda: ["e2b"])

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(val) if (val := os.getenv("E2B_API_KEY")) else None
        ),
        description="E2B API key. Falls back to E2B_API_KEY env var.",
        json_schema_extra={"required": False},
        repr=False,
    )
    domain: str | None = Field(
        default_factory=lambda: os.getenv("E2B_DOMAIN"),
        description="E2B API domain override. Falls back to E2B_DOMAIN env var.",
        json_schema_extra={"required": False},
    )

    template: str | None = Field(
        default=None,
        description=(
            "Optional template/snapshot name or id to create the sandbox from. "
            "Defaults to E2B's base template when omitted."
        ),
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
            "Attach to an existing sandbox by id instead of creating a new "
            "one. The tool will never kill a sandbox it did not create."
        ),
    )
    sandbox_timeout: int = Field(
        default=300,
        description=(
            "Idle timeout in seconds after which E2B auto-kills the sandbox. "
            "Applied at create time and when attaching via sandbox_id."
        ),
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Environment variables to set inside the sandbox at create time.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Metadata key-value pairs to attach to the sandbox at create time.",
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="E2B_API_KEY",
                description="API key for E2B sandbox service",
                required=False,
            ),
            EnvVar(
                name="E2B_DOMAIN",
                description="E2B API domain (optional)",
                required=False,
            ),
        ]
    )

    _persistent_sandbox: Any | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cleanup_registered: bool = PrivateAttr(default=False)

    _sdk_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _import_sandbox_class(cls) -> Any:
        """Return the Sandbox class used by this tool.

        Subclasses override this to swap in a different SDK (e.g. the code
        interpreter sandbox). The default uses plain `e2b.Sandbox`.
        """
        cached = cls._sdk_cache.get("e2b.Sandbox")
        if cached is not None:
            return cached
        try:
            from e2b import Sandbox  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'e2b' package is required for E2B sandbox tools. "
                "Install it with: uv add e2b  (or) pip install e2b"
            ) from exc
        cls._sdk_cache["e2b.Sandbox"] = Sandbox
        return Sandbox

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key.get_secret_value()
        if self.domain:
            kwargs["domain"] = self.domain
        if self.sandbox_timeout is not None:
            kwargs["timeout"] = self.sandbox_timeout
        return kwargs

    def _create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = self._connect_kwargs()
        if self.template is not None:
            kwargs["template"] = self.template
        if self.envs is not None:
            kwargs["envs"] = self.envs
        if self.metadata is not None:
            kwargs["metadata"] = self.metadata
        return kwargs

    def _acquire_sandbox(self) -> tuple[Any, bool]:
        """Return (sandbox, should_kill_after_use)."""
        sandbox_cls = self._import_sandbox_class()

        if self.sandbox_id:
            return (
                sandbox_cls.connect(self.sandbox_id, **self._connect_kwargs()),
                False,
            )

        if self.persistent:
            with self._lock:
                if self._persistent_sandbox is None:
                    self._persistent_sandbox = sandbox_cls.create(
                        **self._create_kwargs()
                    )
                    if not self._cleanup_registered:
                        atexit.register(self.close)
                        self._cleanup_registered = True
                return self._persistent_sandbox, False

        sandbox = sandbox_cls.create(**self._create_kwargs())
        return sandbox, True

    def _release_sandbox(self, sandbox: Any, should_kill: bool) -> None:
        if not should_kill:
            return
        try:
            sandbox.kill()
        except Exception:
            logger.debug(
                "Best-effort sandbox cleanup failed after ephemeral use; "
                "the sandbox may need manual termination.",
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
                "Best-effort persistent sandbox cleanup failed at close(); "
                "the sandbox may need manual termination.",
                exc_info=True,
            )
