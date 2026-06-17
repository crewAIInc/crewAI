import atexit
import typing
from typing_extensions import Self

from .settings import (
    K8sAgentSandboxToolClientSettings,
    K8sAgentSandboxToolSandboxSettings,
)
from .lifecycle_manager import (
    K8sAgentSandboxLifecycleManager,
    EphemeralModeK8sAgentSandboxLifecycleManager,
    AttachModeK8sAgentSandboxLifecycleManager,
    PersistentModeK8sAgentSandboxLifecycleManager,
)

if typing.TYPE_CHECKING:
    # Avoiding a circular dependency.
    from .base_tool import K8sAgentSandboxBaseTool



class K8sAgentSandboxToolset:
    """
    The toolset is responsible for sharing the settings among many K8s Agent Sandbox tools.
    The tools that are added to a same toolset will share the sandbox settings and the sandbox itself.

    It is recommended to use the factory method :meth:`create` insteat of the constructor.

    Args:
        lifecycle_manager: The instanse of a sandbox lifecycle manager that is responsible for
            managing a sandbox.
        cleanup_on_exit: When True, registers its :meth:`close` method at the `atexit` module
            to be called when the programm exits.
    """
    def __init__(
        self,
        lifecycle_manager: K8sAgentSandboxLifecycleManager,
        cleanup_on_exit: bool = True,
    ):
        self.lifecycle_manager = lifecycle_manager

        if cleanup_on_exit:
            atexit.register(self.close)



        self._all_tools: dict[str, 'K8sAgentSandboxBaseTool'] = {}

    def add_tool(self, tool: 'K8sAgentSandboxBaseTool'):
        name = tool.name
        if name in self._all_tools:
            raise ValueError(f"The tool '{name}' is already in the toolset.")

        self._all_tools[name] = tool


    @property
    def tools(self) -> list['K8sAgentSandboxBaseTool']:
        return list(self._all_tools.values())

    def close(self):
        self.lifecycle_manager.close()


    @classmethod
    def create(
        cls,
        sandbox_settings: K8sAgentSandboxToolSandboxSettings,
        client_settings: K8sAgentSandboxToolClientSettings | None = None,
        persistent: bool = False,
        claim_name: str | None = None,
        cleanup_on_exit: bool = True,
    ) -> Self:

        """
        Create a toolset by using Agent Sandbox client and sandbox settings.

        Args:
            sandbox_settings: Settings for the sandbox instanse that will be
                managed by this toolset.
            client_settings: Settings for K8s Agent Sandbox client. If None,
                the default settings with Local Tunnel Mode is used.
            claim_name:  Attach to an existing sandbox by its claim_name instead of
                creating a new one. The tool will never kill a sandbox it did not
                create. Mutually exclusive with the `persistent` argument.
            persistent: If True, reuses one sandbox across all calls of the tools in this toolset,
                and this sandbox can be killed by closing the toolset. Mutually exclusive with
                the `claim_name` argument.
            cleanup_on_exit: Same as in the constructor.

        """
        if persistent and claim_name is not None:
            raise ValueError("The persistent and attach modes are mutually exclusive.")

        client_settings = client_settings or K8sAgentSandboxToolClientSettings()


        if claim_name is not None:
            lifecycle_manager = AttachModeK8sAgentSandboxLifecycleManager(
                client_settings,
                sandbox_settings,
                claim_name,
            )

        elif persistent:
            lifecycle_manager = PersistentModeK8sAgentSandboxLifecycleManager(
                client_settings,
                sandbox_settings,
            )

        else:
            lifecycle_manager = EphemeralModeK8sAgentSandboxLifecycleManager(
                client_settings,
                sandbox_settings,
            )

        return cls(
            lifecycle_manager,
            cleanup_on_exit=cleanup_on_exit,
        )

