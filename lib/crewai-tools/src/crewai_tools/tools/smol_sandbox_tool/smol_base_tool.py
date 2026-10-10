from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
import logging
from pathlib import Path
import threading
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr


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
        description="Allow guest egress for network requests and package installs.",
    )
    api_key: SecretStr | None = Field(default=None, repr=False)
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
    _pending_cleanup: list[Any] = PrivateAttr(default_factory=list)
    _deleting_count: int = PrivateAttr(default=0)
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
            target=self.target,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
        )

    def _create_machine(self, sdk: Any) -> Any:
        if self.target == "local" and not self.network:
            image = self.image
            local_path = image.startswith(("/", "./", "../")) or image.endswith(
                (".tar", ".tar.gz", ".tgz")
            )
            windows_path = (
                len(image) >= 3
                and image[0].isalpha()
                and image[1] == ":"
                and image[2] in ("/", "\\")
            )
            if (local_path or windows_path) and not Path(image).exists():
                raise ValueError(f"Local offline image source does not exist: {image}")
        return sdk.Machine.create(
            sdk.MachineConfig(
                image=self.image,
                resources=sdk.ResourceSpec(network=self.network),
                # Tool reuse is in-process; SDK persistence would retain local
                # disk state if this process exits before close() can delete it.
                persistent=False,
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

    @contextmanager
    def _machine_session(self) -> Iterator[Any]:
        machine, delete = self._acquire_machine()
        try:
            yield machine
        except BaseException:
            self._release_machine(machine, delete, suppress_errors=True)
            raise
        else:
            self._release_machine(machine, delete, suppress_errors=False)

    def _cleanup_on_exit(self) -> None:
        try:
            self._close_owned(unregister=False)
        except Exception:
            logger.warning("Could not delete Smol Machines VM", exc_info=True)

    def _release_machine(
        self, machine: Any, delete: bool, *, suppress_errors: bool
    ) -> None:
        if not delete:
            return
        try:
            machine.delete()
        except Exception:
            with self._lock:
                self._pending_cleanup.append(machine)
                if not self._cleanup_registered:
                    atexit.register(self._cleanup_on_exit)
                    self._cleanup_registered = True
            if not suppress_errors:
                raise
            logger.warning(
                "Could not delete Smol Machines VM %s after tool failure; close() will retry",
                getattr(machine, "id", "unknown"),
                exc_info=True,
            )

    def close(self) -> None:
        """Delete owned VMs, retrying failed deletions on the next close."""
        self._close_owned(unregister=True)

    def _close_owned(self, *, unregister: bool) -> None:
        if self.machine_id:
            return
        with self._lock:
            if self._machine is not None:
                self._pending_cleanup.append(self._machine)
                self._machine = None
            to_delete = self._pending_cleanup
            self._pending_cleanup = []
            self._deleting_count += len(to_delete)
        failed: list[Any] = []
        first_error: Exception | None = None
        for machine in to_delete:
            try:
                machine.delete()
            except Exception as exc:  # noqa: PERF203 - VM deletion dwarfs loop overhead
                failed.append(machine)
                if first_error is None:
                    first_error = exc
        with self._lock:
            self._pending_cleanup.extend(failed)
            self._deleting_count -= len(to_delete)
            # A concurrent run or close may overlap a slow delete.
            if (
                unregister
                and self._machine is None
                and not self._pending_cleanup
                and self._deleting_count == 0
                and self._cleanup_registered
            ):
                atexit.unregister(self._cleanup_on_exit)
                self._cleanup_registered = False
        if first_error is not None:
            raise first_error

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
