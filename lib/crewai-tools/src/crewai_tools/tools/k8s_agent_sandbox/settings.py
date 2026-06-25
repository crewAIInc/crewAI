from dataclasses import (
    dataclass,
    field,
)

from k8s_agent_sandbox.models import (
    SandboxConnectionConfig,
    SandboxTracerConfig,
)
from k8s_agent_sandbox.sandbox_client import SandboxClient


@dataclass
class K8sAgentSandboxToolClientSettings:
    """
    The dataclass that stores data required to created an Agent Sandbox Client.
    It basically has the same arguments as the `k8s_agent_sandbox.SandboxClient`
    class.
    """

    connection_config: SandboxConnectionConfig | None = None
    tracer_config: SandboxTracerConfig | None = None
    cleanup: bool = False

    _client: SandboxClient | None = field(default=None, init=False, repr=False)

    @property
    def client(self) -> SandboxClient:
        if self._client is not None:
            return self._client

        self._client = SandboxClient(
            connection_config=self.connection_config,
            tracer_config=self.tracer_config,
            cleanup=self.cleanup,
        )
        return self._client


@dataclass
class K8sAgentSandboxToolSandboxSettings:
    """
    The dataclass that stores sandbox settings.

    Args:
        warmpool: Name of the warm pool where a sandbox should be created.
        namespace: Name of the namespace where a sandbox should be created.
        sandbox_timeout: The timeout in seconds after which the sandbox will be automatically killed.
    """

    warmpool: str
    namespace: str
    sandbox_timeout: int = 300
