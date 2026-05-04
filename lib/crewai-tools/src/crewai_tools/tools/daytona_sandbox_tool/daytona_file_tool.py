from __future__ import annotations

import base64
from builtins import type as type_
import logging
import posixpath
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.daytona_sandbox_tool.daytona_base_tool import DaytonaBaseTool


logger = logging.getLogger(__name__)


FileAction = Literal["read", "write", "append", "list", "delete", "mkdir", "info"]


class DaytonaFileToolSchema(BaseModel):
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
        description="For action='delete': remove directories recursively.",
    )
    mode: str = Field(
        default="0755",
        description="For action='mkdir': octal permission string (default 0755).",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> DaytonaFileToolSchema:
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        return self


class DaytonaFileTool(DaytonaBaseTool):
    """Read, write, and manage files inside a Daytona sandbox.

    Notes:
      - Most useful with `persistent=True` or an explicit `sandbox_id`. With the
        default ephemeral mode, files disappear when this tool call finishes.
    """

    name: str = "Daytona Sandbox Files"
    description: str = (
        "Perform filesystem operations inside a Daytona sandbox: read a file, "
        "write content to a path, append content to an existing file, list a "
        "directory, delete a path, make a directory, or fetch file metadata. "
        "For files larger than a few KB, create the file with action='write' "
        "and empty content, then send the body via multiple 'append' calls of "
        "~4KB each to stay within tool-call payload limits."
    )
    args_schema: type_[BaseModel] = DaytonaFileToolSchema

    def _run(
        self,
        action: FileAction,
        path: str,
        content: str | None = None,
        binary: bool = False,
        recursive: bool = False,
        mode: str = "0755",
    ) -> Any:
        sandbox, should_delete = self._acquire_sandbox()
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
                sandbox.fs.delete_file(path, recursive=recursive)
                return {"status": "deleted", "path": path}
            if action == "mkdir":
                sandbox.fs.create_folder(path, mode)
                return {"status": "created", "path": path, "mode": mode}
            if action == "info":
                return self._info(sandbox, path)
            raise ValueError(f"Unknown action: {action}")
        finally:
            self._release_sandbox(sandbox, should_delete)

    def _read(self, sandbox: Any, path: str, *, binary: bool) -> dict[str, Any]:
        data: bytes = sandbox.fs.download_file(path)
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
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        payload = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path)
        sandbox.fs.upload_file(payload, path)
        return {"status": "written", "path": path, "bytes": len(payload)}

    def _append(
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        chunk = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path)
        try:
            existing: bytes = sandbox.fs.download_file(path)
        except Exception:
            existing = b""
        payload = existing + chunk
        sandbox.fs.upload_file(payload, path)
        return {
            "status": "appended",
            "path": path,
            "appended_bytes": len(chunk),
            "total_bytes": len(payload),
        }

    @staticmethod
    def _ensure_parent_dir(sandbox: Any, path: str) -> None:
        """Make sure the parent directory of `path` exists.

        Daytona's upload returns 400 if the parent directory is missing. We
        best-effort mkdir the parent; any error (e.g. already exists) is
        swallowed because `create_folder` is not idempotent on the server.
        """
        parent = posixpath.dirname(path)
        if not parent or parent in ("/", "."):
            return
        try:
            sandbox.fs.create_folder(parent, "0755")
        except Exception:
            logger.debug(
                "Best-effort parent-directory create failed for %s; "
                "assuming it already exists and proceeding with the write.",
                parent,
                exc_info=True,
            )

    def _list(self, sandbox: Any, path: str) -> dict[str, Any]:
        entries = sandbox.fs.list_files(path)
        return {
            "path": path,
            "entries": [self._file_info_to_dict(entry) for entry in entries],
        }

    def _info(self, sandbox: Any, path: str) -> dict[str, Any]:
        return self._file_info_to_dict(sandbox.fs.get_file_info(path))

    @staticmethod
    def _file_info_to_dict(info: Any) -> dict[str, Any]:
        fields = (
            "name",
            "size",
            "mode",
            "permissions",
            "is_dir",
            "mod_time",
            "owner",
            "group",
        )
        return {field: getattr(info, field, None) for field in fields}
