from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, ClassVar

from crewai.tools import BaseTool
from pydantic import ConfigDict, Field, PrivateAttr


logger = logging.getLogger(__name__)


class BoxLiteBaseTool(BaseTool):
    """Shared base for tools that act on a BoxLite micro-VM box.

    BoxLite boots an OCI image inside a hardware-isolated micro-VM on the local
    host, so — unlike the E2B and Daytona sandbox tools — no API key or cloud
    account is required. The host must support micro-VMs: macOS 12+ on Apple
    Silicon, or Linux with KVM (``/dev/kvm``).

    Lifecycle modes (mirrors the E2B/Daytona sandbox tools):
      - ``persistent=False`` (default): create a fresh box per ``_run`` call and
        remove it when the call returns. Stateless and safe — nothing leaks if
        the agent forgets cleanup.
      - ``persistent=True``: lazily create one box on first use, cache it on the
        instance, and remove it at process exit. Cheaper across many calls and
        lets files or installed packages carry over between them.

    Boxes are addressed only through this tool instance; there is no attach to a
    box created elsewhere, because BoxLite's synchronous API does not expose
    one with the rich exec (cwd/env/timeout) these tools rely on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_dependencies: list[str] = Field(default_factory=lambda: ["boxlite[sync]"])

    image: str = Field(
        default="python:slim",
        description=(
            "OCI image the micro-VM boots from. Any Docker/OCI reference works "
            "(e.g. 'python:slim', 'ubuntu:24.04')."
        ),
    )
    cpus: int | None = Field(
        default=None,
        description="vCPUs for the box. Defaults to the BoxLite runtime default.",
    )
    memory_mib: int | None = Field(
        default=None,
        description="Memory in MiB for the box. Defaults to the BoxLite runtime default.",
    )
    persistent: bool = Field(
        default=False,
        description=(
            "If True, reuse one box across all calls to this tool instance and "
            "remove it at process exit. Default False creates and removes a "
            "fresh box per call."
        ),
    )

    _persistent_box: Any | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cleanup_registered: bool = PrivateAttr(default=False)

    _sdk_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _import_boxlite(cls) -> Any:
        """Return the ``boxlite`` module.

        Imported lazily and cached so that importing this tool (e.g. for spec
        generation or ``from crewai_tools import ...``) never requires the
        optional ``boxlite`` dependency to be installed.
        """
        cached = cls._sdk_cache.get("boxlite")
        if cached is not None:
            return cached
        try:
            import boxlite  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'boxlite' package is required for BoxLite sandbox tools. "
                "Install it with: uv add 'boxlite[sync]'  (or) "
                "pip install 'boxlite[sync]'"
            ) from exc
        cls._sdk_cache["boxlite"] = boxlite
        return boxlite

    def _new_box(self) -> Any:
        """Create and start a fresh box from this tool's configuration."""
        boxlite = self._import_boxlite()
        box = boxlite.SyncSimpleBox(
            image=self.image,
            cpus=self.cpus,
            memory_mib=self.memory_mib,
            auto_remove=True,
        )
        # SyncSimpleBox exposes its lifecycle only via the context-manager
        # protocol; enter it here and exit in _release_box/close so the box can
        # outlive a single ``with`` block (required for persistent mode).
        box.__enter__()
        return box

    def _acquire_box(self) -> tuple[Any, bool]:
        """Return ``(box, should_remove_after_use)``."""
        if self.persistent:
            with self._lock:
                if self._persistent_box is None:
                    self._persistent_box = self._new_box()
                    if not self._cleanup_registered:
                        atexit.register(self.close)
                        self._cleanup_registered = True
                return self._persistent_box, False
        return self._new_box(), True

    def _release_box(self, box: Any, should_remove: bool) -> None:
        if not should_remove:
            return
        try:
            box.__exit__(None, None, None)
        except Exception:
            logger.debug(
                "Best-effort box cleanup failed after ephemeral use; "
                "the micro-VM may need manual removal.",
                exc_info=True,
            )

    def close(self) -> None:
        """Remove the cached persistent box if one exists."""
        with self._lock:
            box = self._persistent_box
            self._persistent_box = None
        if box is None:
            return
        try:
            box.__exit__(None, None, None)
        except Exception:
            logger.debug(
                "Best-effort persistent box cleanup failed at close(); "
                "the micro-VM may need manual removal.",
                exc_info=True,
            )

    @staticmethod
    def _result_dict(result: Any) -> dict[str, Any]:
        """Normalize a BoxLite ExecResult into a plain JSON-friendly dict."""
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
