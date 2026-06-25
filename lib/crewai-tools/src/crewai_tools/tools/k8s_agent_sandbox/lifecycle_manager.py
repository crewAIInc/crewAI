from threading import Lock
from abc import ABC, abstractmethod
import logging

from k8s_agent_sandbox.exceptions import SandboxNotFoundError
from k8s_agent_sandbox.sandbox import Sandbox


from .settings import (
    K8sAgentSandboxToolClientSettings,
    K8sAgentSandboxToolSandboxSettings,
)


logger = logging.getLogger(__name__)


class K8sAgentSandboxLifecycleManager(ABC):
    """
    The base lifecycle manager of the K8s Agent Sandbox.
    """

    def __init__(
        self,
        client_settings: K8sAgentSandboxToolClientSettings,
        sandbox_settings: K8sAgentSandboxToolSandboxSettings,
    ):
        self._sandbox_settings = sandbox_settings
        self._client_settings = client_settings or K8sAgentSandboxToolClientSettings()

        self._client = self._client_settings.client

        self._lock = Lock()
        self._closed = False

        self._sandbox: Sandbox | None = None
        self._sandbox_acquired: bool = False

    def acquire_sandbox(self) -> Sandbox:
        """
        Acquires a sandbox based on this implementation and returns it.
        In order tto be acquired again by someone else, it has to be
        released first by the :meth:`release_sandbox` method.
        """
        if self._closed:
            raise RuntimeError("Attempt to acquire a sandbox from a closed helper.")
        self._lock.acquire()
        self._sandbox_acquired = True
        return self._acquire_sandbox()

    def release_sandbox(self) -> None:
        """
        Releases a sandbox that is previously acquired.
        """
        if self._closed:
            return
        if not self._sandbox_acquired:
            return
        try:
            self._release_sandbox()
        finally:
            self._lock.release()

    def close(self) -> None:
        """Closes the lifecycle manager."""
        if self._closed:
            return

        self._close()

    @abstractmethod
    def _acquire_sandbox(self) -> Sandbox:
        pass

    @abstractmethod
    def _release_sandbox(self) -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        pass

    def _terminate_sandbox(self) -> None:
        if self._sandbox is None:
            return

        self._sandbox.terminate()
        self._sandbox = None

    def _create_sandbox(self) -> Sandbox:
        return self._client.create_sandbox(
            warmpool=self._sandbox_settings.warmpool,
            namespace=self._sandbox_settings.namespace,
            shutdown_after_seconds=self._sandbox_settings.sandbox_timeout,
        )


class EphemeralModeK8sAgentSandboxLifecycleManager(K8sAgentSandboxLifecycleManager):
    """
    Lifecycle manager that creates new sandbox on each call of the `acquire_sandbox`
    method and terminates it on `release_sandbox`.
    """

    def _acquire_sandbox(self) -> Sandbox:
        self._sandbox = self._create_sandbox()
        return self._sandbox

    def _release_sandbox(self) -> None:
        self._terminate_sandbox()

    def _close(self) -> None:
        self._terminate_sandbox()


class AttachModeK8sAgentSandboxLifecycleManager(K8sAgentSandboxLifecycleManager):
    """
    Lifecycle manager that attaches to existing sandbox by its claim name without creating a new sandbox.
    Sandbox is also not terminated on the `release_sandbox`.
    """

    def __init__(
        self,
        client_settings: K8sAgentSandboxToolClientSettings,
        sandbox_settings: K8sAgentSandboxToolSandboxSettings,
        claim_name: str,
    ):
        super().__init__(client_settings, sandbox_settings)
        self._claim_name = claim_name

    def _acquire_sandbox(self) -> Sandbox:
        try:
            self._sandbox = self._client.get_sandbox(
                self._claim_name,
                namespace=self._sandbox_settings.namespace,
            )
        except SandboxNotFoundError:
            self._sandbox = None

        if self._sandbox is not None:
            return self._sandbox

        raise SandboxNotFoundError(
            f"A sandbox with sandbox claim '{self._claim_name}' "
            "is expected to exist, but cannot be found."
        )

    def _release_sandbox(self) -> None:
        pass

    def _close(self) -> None:
        pass


class PersistentModeK8sAgentSandboxLifecycleManager(K8sAgentSandboxLifecycleManager):
    """
    Lifecycle manager that manages a sandbox which remains the same across all calls
    of the`acquire_sandbox` and `release_sandbox`.
    """

    def _acquire_sandbox(self) -> Sandbox:
        if self._sandbox is not None:
            return self._sandbox

        self._sandbox = self._create_sandbox()
        return self._sandbox

    def _release_sandbox(self) -> None:
        pass

    def _close(self) -> None:
        self._terminate_sandbox()
