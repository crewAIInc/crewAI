from __future__ import annotations

import base64
from builtins import type as type_
import logging
import posixpath
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.open_sandbox_tool.open_sandbox_base_tool import (
    OpenSandboxBaseTool,
)


logger = logging.getLogger(__name__)


FileAction = Literal["read", "write", "append", "list", "delete", "mkdir", "info"]


class OpenSandboxFileToolSchema(BaseModel):
    action: FileAction = Field(
        ...,
        description=(
            "The filesystem action to perform: 'read' (returns file contents), "
            "'write' (create or replace a file with content), 'append' (append "
            "content to an existing file — use this for writing large files in "
            "chunks to avoid hitting tool-call size limits), 'list' (lists a "
            "directory), 'delete' (removes a file/dir), 'mkdir' (creates a "
            "directory), 'info' (returns file metadata)."
        ),
    )
    path: str = Field(..., description="Absolute path inside the sandbox.")
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
            "For 'write': treat content as base64 and upload raw bytes. "
            "For 'read': return contents as base64 instead of decoded utf-8."
        ),
    )
    recursive: bool = Field(
        default=False,
        description="For action='delete': remove a directory recursively.",
    )
    mode: int = Field(
        default=755,
        description="For action='mkdir': Unix file mode as an integer (default 755).",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> OpenSandboxFileToolSchema:
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        return self


class OpenSandboxFileTool(OpenSandboxBaseTool):
    """Read, write, and manage files inside an Open Sandbox sandbox.

    Notes:
      - Most useful with `persistent=True` or an explicit `sandbox_id`. With the
        default ephemeral mode, files disappear when this tool call finishes.
    """

    name: str = "Open Sandbox File"
    description: str = (
        "Perform filesystem operations inside an Open Sandbox sandbox: read a "
        "file, write content to a path, append content to an existing file, "
        "list a directory, delete a path, make a directory, or fetch file "
        "metadata. For files larger than a few KB, create the file with "
        "action='write' and empty content, then send the body via multiple "
        "'append' calls of ~4KB each to stay within tool-call payload limits."
    )
    args_schema: type_[BaseModel] = OpenSandboxFileToolSchema

    def _run(
        self,
        action: FileAction,
        path: str,
        content: str | None = None,
        binary: bool = False,
        recursive: bool = False,
        mode: int = 755,
    ) -> Any:
        sandbox, should_kill = self._acquire_sandbox()
        try:
            if action == "read":
                return self._read(sandbox, path, binary=binary)
            if action == "write":
                return self._write(sandbox, path, content or "", binary=binary)
            if action == "append":
                return self._append(sandbox, path, content or "", binary=binary)
            if action == "list":
                return self._list(sandbox, path)
            if action == "delete":
                return self._delete(sandbox, path, recursive=recursive)
            if action == "mkdir":
                return self._mkdir(sandbox, path, mode=mode)
            if action == "info":
                return self._info(sandbox, path)
            raise ValueError(f"Unknown action: {action}")
        finally:
            self._release_sandbox(sandbox, should_kill)

    def _read(self, sandbox: Any, path: str, *, binary: bool) -> dict[str, Any]:
        if binary:
            data: bytes = sandbox.files.read_bytes(path)
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
        try:
            text: str = sandbox.files.read_file(path)
            return {"path": path, "encoding": "utf-8", "content": text}
        except UnicodeDecodeError:
            data = sandbox.files.read_bytes(path)
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "note": "File was not valid utf-8; returned as base64.",
            }

    def _write(
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        payload = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path)
        sandbox.files.write_file(path, payload)
        return {"status": "written", "path": path, "bytes": len(payload)}

    def _append(
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        chunk = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path)
        try:
            existing: bytes = sandbox.files.read_bytes(path)
        except Exception:
            existing = b""
        payload = existing + chunk
        sandbox.files.write_file(path, payload)
        return {
            "status": "appended",
            "path": path,
            "appended_bytes": len(chunk),
            "total_bytes": len(payload),
        }

    def _ensure_parent_dir(self, sandbox: Any, path: str) -> None:
        """Make sure the parent directory of `path` exists.

        Best-effort mkdir of the parent; any error (e.g. already exists) is
        swallowed because create_directories may not be idempotent.
        """
        parent = posixpath.dirname(path)
        if not parent or parent in ("/", "."):
            return
        sdk = self._import_sdk()
        try:
            sandbox.files.create_directories([sdk["WriteEntry"](path=parent, mode=755)])
        except Exception:
            logger.debug(
                "Best-effort parent-directory create failed for %s; "
                "assuming it already exists and proceeding with the write.",
                parent,
                exc_info=True,
            )

    def _mkdir(self, sandbox: Any, path: str, *, mode: int) -> dict[str, Any]:
        sdk = self._import_sdk()
        sandbox.files.create_directories([sdk["WriteEntry"](path=path, mode=mode)])
        return {"status": "created", "path": path, "mode": mode}

    def _delete(self, sandbox: Any, path: str, *, recursive: bool) -> dict[str, Any]:
        if recursive:
            sandbox.files.delete_directories([path])
        else:
            sandbox.files.delete_files([path])
        return {"status": "deleted", "path": path}

    def _list(self, sandbox: Any, path: str) -> dict[str, Any]:
        sdk = self._import_sdk()
        entries = sandbox.files.search(sdk["SearchEntry"](path=path, pattern="*"))
        return {
            "path": path,
            "entries": [self._entry_info_to_dict(entry) for entry in entries],
        }

    def _info(self, sandbox: Any, path: str) -> dict[str, Any]:
        info_map = sandbox.files.get_file_info([path])
        info = info_map.get(path) if hasattr(info_map, "get") else None
        if info is None:
            return {"path": path, "found": False}
        return self._entry_info_to_dict(info)

    @staticmethod
    def _entry_info_to_dict(info: Any) -> dict[str, Any]:
        fields = (
            "path",
            "mode",
            "owner",
            "group",
            "size",
            "modified_at",
            "created_at",
        )
        out: dict[str, Any] = {}
        for field in fields:
            value = getattr(info, field, None)
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            out[field] = value
        return out
