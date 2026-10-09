from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr


logger = logging.getLogger(__name__)


class SmolBaseTool(BaseTool):
    """Shared lifecycle for a local or cloud Smol Machines microVM.

    Each tool call creates and deletes its own VM by default. ``persistent=True``
    keeps a VM across calls to this tool, while ``machine_id`` attaches to an
    existing VM and never deletes it. Share an owned VM across tools by passing
    the first tool's ``active_machine_id`` to the others.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_dependencies: list[str] = Field(default_factory=lambda: ["smolmachines"])
    target: Literal["local", "cloud"] = "local"
    image: str = "python:3.12-alpine"
    network: bool = Field(
        default=True,
        description="Allow guest egress (also needed for cold image pulls on cloud).",
    )
    api_key: str | None = Field(default=None, repr=False)
    base_url: str | None = None
    persistent: bool = False
    machine_id: str | None = None
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SMOL_CLOUD_TOKEN",
                description="Smol Machines Cloud token (cloud target only)",
                required=False,
            )
        ]
    )

    _machine: Any = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cleanup_registered: bool = PrivateAttr(default=False)

    @staticmethod
    def _sdk() -> Any:
        try:
            import smol
        except ImportError as exc:
            raise ImportError(
                "Smol Machines sandbox tools require smolmachines; "
                "install 'crewai-tools[smol]' or 'smolmachines'."
            ) from exc
        return smol

    def _connection(self, sdk: Any) -> Any:
        return sdk.ConnectOptions(
            target=self.target, api_key=self.api_key, base_url=self.base_url
        )

    def _create_machine(self, sdk: Any) -> Any:
        return sdk.Machine.create(
            sdk.MachineConfig(
                image=self.image, network=self.network, persistent=self.persistent
            ),
            self._connection(sdk),
        )

    def _acquire_machine(self) -> tuple[Any, bool]:
        sdk = self._sdk()
        if self.machine_id:
            # The caller owns this machine. Connecting never grants cleanup rights.
            machine = sdk.Machine.connect(self.machine_id, self._connection(sdk))
            if machine.state() == "stopped":
                machine.start()
            machine.wait_until_ready()
            return machine, False

        if self.persistent:
            with self._lock:
                if self._machine is None:
                    self._machine = self._create_machine(sdk)
                    if not self._cleanup_registered:
                        atexit.register(self._cleanup_on_exit)
                        self._cleanup_registered = True
                return self._machine, False

        return self._create_machine(sdk), True

    def _cleanup_on_exit(self) -> None:
        try:
            self.close()
        except Exception:
            logger.warning(
                "Could not delete persistent Smol Machines VM", exc_info=True
            )

    @staticmethod
    def _release_machine(machine: Any, delete: bool) -> None:
        if delete:
            machine.delete()

    def close(self) -> None:
        """Delete the VM created by persistent mode; attached VMs remain owned by their caller."""
        if self.machine_id:
            return
        with self._lock:
            if self._machine is not None:
                self._machine.delete()
                self._machine = None

    @property
    def active_machine_id(self) -> str | None:
        """ID for sharing the persistent VM with another tool."""
        if self.machine_id:
            return self.machine_id
        with self._lock:
            return self._machine.id if self._machine is not None else None


def execution_result(result: Any) -> dict[str, Any]:
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }
