from typing import Callable, Literal
import base64
import shlex
from typing import Any
import posixpath
import logging
from textwrap import dedent

from pydantic import (
    Field,
    BaseModel,
    model_validator,
)
from k8s_agent_sandbox.models import FileEntry
from k8s_agent_sandbox.sandbox import Sandbox

from crewai_tools.tools.k8s_agent_sandbox.base_tool import (
    DEFAULT_TOOL_TIMEOUT_SEC,
    K8sAgentSandboxBaseTool,
    create_timeout_tracker,
)


logger = logging.getLogger(__name__)

FileAction = Literal[
    "read", "write", "append", "list", "delete", "mkdir", "info", "exists"
]


class K8sAgentSandboxFileToolSchema(BaseModel):
    action: FileAction = Field(
        ...,
        description=dedent("""
            The filesystem action to perform:"
              - read: returns file contents.
              - write: create or replace a file with content.
              - append: append content to an existing file — use this for
                writing large files in chunks to avoid hitting tool-call
                size limits.
              - list: lists a directory.
              - delete: removes a file/directory.
              - mkdir: creates a directory.
              - exists: returns a boolean for whether the path exists.
        """),
    )
    path: str = Field(
        ...,
        description="The path inside the sandbox which is relative to a sandbox's working directory.",
    )
    content: str | None = Field(
        default=None,
        description=(
            "Content to write or append. If omitted for 'write', an empty file "
            "is created. For files larger than a few KB, prefer one 'write' "
            "with empty content followed by multiple 'append' calls of ~4KB "
            "each to stay within tool-call payload limits."
        ),
    )
    binary: bool = Field(
        default=False,
        description=(
            "For 'write'/'append': treat content as base64 and upload raw "
            "bytes. For 'read': return contents as base64 instead of decoded "
            "utf-8."
        ),
    )

    timeout: int | None = Field(
        default=DEFAULT_TOOL_TIMEOUT_SEC,
        description="Maximum seconds to wait for the action to finish.",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> "K8sAgentSandboxFileToolSchema":
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        return self


class K8sAgentSandboxFileTool(K8sAgentSandboxBaseTool):
    """Read, write, and manage files inside an K8s agent sandbox.

    Notes:
      - Most useful with `persistent=True` or an explicit `sandbox_id`. With
        the default ephemeral mode, files disappear when this tool call
        finishes.
    """

    name: str = "K8s Agent Sandbox Files Tool"
    description: str = (
        "Perform filesystem operations inside an K8s agent sandbox: read a file, "
        "write content to a path, append content to an existing file, list a "
        "directory, delete a path, make a directory, fetch file metadata, or "
        "check whether a path exists.")
    args_schema: type[BaseModel] = K8sAgentSandboxFileToolSchema

    def _run_with_sandbox(
        self,
        sandbox: Sandbox,
        action: FileAction,
        path: str,
        timeout: int,
        binary: bool,
        content: str | None = None,
    ) -> dict[str, Any]:

        if action == "read":
            return self._read(sandbox, path, binary=binary, timeout=timeout)
        if action == "write":
            return self._write(
                sandbox,
                path,
                content or "",
                binary=binary,
                timeout_tracker=create_timeout_tracker(timeout),
            )
        if action == "append":
            return self._append(
                sandbox,
                path,
                content or "",
                binary=binary,
                timeout_tracker=create_timeout_tracker(timeout),
            )
        if action == "list":
            return self._list(sandbox, path, timeout=timeout)
        if action == "delete":
            return self._delete(sandbox, path, timeout=timeout)
        if action == "mkdir":
            self._mkdir(sandbox, path, timeout=timeout)
            return {"status": "created", "path": path}
        if action == "info":
            return self._info(sandbox, path)
        if action == "exists":
            result = sandbox.files.exists(path, timeout=timeout)
            return {"path": path, "exists": result}

        raise ValueError(f"Unknown action: {action}")

    def _read(
        self,
        sandbox: Sandbox,
        path: str,
        *,
        binary: bool,
        timeout: int,
    ) -> dict[str, Any]:

        data: bytes = sandbox.files.read(path, timeout=timeout)
        if binary:
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
        try:
            return {"path": path, "encoding": "utf-8", "content": data.decode("utf-8")}
        except UnicodeDecodeError:
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "note": "File was not valid utf-8; returned as base64.",
            }

    def _write(
        self,
        sandbox: Sandbox,
        path: str,
        content: str,
        *,
        binary: bool,
        timeout_tracker: Callable[[], int],
    ) -> dict[str, Any]:

        payload = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path, timeout=timeout_tracker())
        sandbox.files.write(path, payload, timeout=timeout_tracker())
        return {"status": "written", "path": path, "bytes": len(payload)}

    def _append(
        self,
        sandbox: Sandbox,
        path: str,
        content: str,
        *,
        binary: bool,
        timeout_tracker: Callable[[], int],
    ) -> dict[str, Any]:
        chunk: bytes = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path, timeout=timeout_tracker())

        existing = sandbox.files.read(path, timeout=timeout_tracker())
        payload = existing + chunk
        sandbox.files.write(path, payload, timeout=timeout_tracker())
        return {
            "status": "appended",
            "path": path,
            "appended_bytes": len(chunk),
            "total_bytes": len(payload),
        }

    def _list(self, sandbox: Sandbox, path: str, *, timeout: int) -> dict[str, Any]:
        entries = sandbox.files.list(path, timeout=timeout)
        return {
            "path": path,
            "entries": [self._entry_to_dict(e) for e in entries],
        }

    def _delete(self, sandbox: Sandbox, path: str, *, timeout: int) -> dict[str, Any]:
        # TODO: Fall back to deleting with shell command.
        # Use normal file delete API when it is available in SDK.

        path = posixpath.normpath(path)

        if posixpath.dirname(path) == "/":
            raise RuntimeError(f"The path {path} cannot be deleted.")

        command = f"rm -r {shlex.quote(path)}"
        try:
            result = sandbox.commands.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during the run of the deletion command '{command}'. Error: {e}."
            )

        if result.exit_code != 0:
            raise RuntimeError(
                f"Cannot delete directory {path}. Error: {result.stderr}."
            )

        return {"status": "deleted", "path": path}

    def _mkdir(self, sandbox: Sandbox, path: str, *, timeout: int):
        try:
            result = sandbox.commands.run(
                f"mkdir -p {shlex.quote(path)}",
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during the creation of a directory {path}. Error: {e}."
            )

        if result.exit_code != 0:
            raise RuntimeError(
                f"Cannot create directory {path}. Error: {result.stderr}."
            )

    def _ensure_parent_dir(self, sandbox: Sandbox, path: str, timeout: int):
        parent = posixpath.dirname(path)
        if not parent or parent in ("/", "."):
            return

        return self._mkdir(sandbox, parent, timeout=timeout)

    @staticmethod
    def _entry_to_dict(entry: FileEntry) -> dict[str, Any]:

        fields = (
            "name",
            "type",
            "size",
            "mod_time",
        )
        result: dict[str, Any] = {}
        for field in fields:
            value = getattr(entry, field, None)
            result[field] = value
        return result
