import atexit
import logging
from typing import Any, Optional
from pydantic import Field
import threading
from typing import ClassVar

from crewai.tools import BaseTool
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr


logger = logging.getLogger(__name__)


class K8sBaseTool(BaseTool):
    name: str = "K8sBaseTool"
    description: str = "Basic tool definition"

    warmpool: str = Field(description="Agent sandbox warmpool name")
    namespace: str = Field(default="default", description="K8s namespace")


    persistent: bool = Field(
        default=False,
        description=(
            "If True, reuse one sandbox across all calls to this tool instance "
            "and kill it at process exit. Default False creates and kills a "
            "fresh sandbox per call."
        ),
    )

    claim_name: Any | None = Field(default=None)
    _persistent_sandbox: Any | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cleanup_registered: bool = PrivateAttr(default=False)

    _sdk_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _import_sandbox_client_class(cls) -> Any:
        """Returns the Sandbox Client that is used to communicate with k8s."""
        cached = cls._sdk_cache.get("k8s_agent_sandbox.SandboxClient")
        if cached is not None:
            return cached
        try:
            from k8s_agent_sandbox import SandboxClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'k8s_agent_sandbox' package is required for k8s_agent_sandbox sandbox tools."
            ) from exc
        cls._sdk_cache["k8s_agent_sandbox.SandboxClient"] = SandboxClient
        return SandboxClient

    @classmethod
    def _import_sandbox_local_tunnel_connection_config_class(cls) -> Any:
        """Returns the Sandbox Client that is used to communicate with k8s."""
        cached = cls._sdk_cache.get("k8s_agent_sandbox.models.SandboxLocalTunnelConnectionConfig")
        if cached is not None:
            return cached
        try:
            from k8s_agent_sandbox.models import SandboxLocalTunnelConnectionConfig  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'k8s_agent_sandbox' package is required for k8s_agent_sandbox sandbox tools."
            ) from exc
        cls._sdk_cache["k8s_agent_sandbox.models.SandboxLocalTunnelConnectionConfig"] = SandboxLocalTunnelConnectionConfig
        return SandboxLocalTunnelConnectionConfig

    def _get_sandbox(self) -> tuple[Any, bool]:
        """Return (sandbox, should_kill_after_use)."""
        SandboxClient = self._import_sandbox_client_class()
        SandboxLocalTunnelConnectionConfig = self._import_sandbox_local_tunnel_connection_config_class()

        client = SandboxClient(
            connection_config=SandboxLocalTunnelConnectionConfig(
              router_namespace=self.namespace
            )
        )

        if self.claim_name:
            return client.get_sandbox(self.claim_name), False

        if self.persistent:
            with self._lock:
                if self._persistent_sandbox is None:
                    self._persistent_sandbox = client.create_sandbox(
                        warmpool=self.warmpool,
                        namespace=self.namespace,
                    )
                    if not self._cleanup_registered:
                        atexit.register(self.close)
                        self._cleanup_registered = True
                return self._persistent_sandbox, False

        sandbox = client.create_sandbox(
            warmpool=self.warmpool,
            namespace=self.namespace,
        )
        return sandbox, True

    def _release_sandbox(self, sandbox, should_terminate) -> None:
        if not should_terminate:
            return
        try:
            sandbox.terminate()
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
            sandbox.terminate()
        except Exception:
            logger.debug(
                "Best-effort persistent sandbox cleanup failed at close(); "
                "the sandbox may need manual termination.",
                exc_info=True,
            )
