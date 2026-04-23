from __future__ import annotations

import base64
from builtins import type as type_
import logging
import posixpath
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.e2b_sandbox_tool.e2b_base_tool import E2BBaseTool


logger = logging.getLogger(__name__)


FileAction = Literal[
    "read", "write", "append", "list", "delete", "mkdir", "info", "exists"
]


class E2BFileToolSchema(BaseModel):
    action: FileAction = Field(
        ...,
        description=(
            "The filesystem action to perform: 'read' (returns file contents), "
            "'write' (create or replace a file with content), 'append' (append "
            "content to an existing file — use this for writing large files in "
            "chunks to avoid hitting tool-call size limits), 'list' (lists a "
            "directory), 'delete' (removes a file/dir), 'mkdir' (creates a "
            "directory), 'info' (returns file metadata), 'exists' (returns a "
            "boolean for whether the path exists)."
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
            "For 'write'/'append': treat content as base64 and upload raw "
            "bytes. For 'read': return contents as base64 instead of decoded "
            "utf-8."
        ),
    )
    depth: int = Field(
        default=1,
        description="For action='list': how many levels deep to recurse (default 1).",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> E2BFileToolSchema:
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        return self


class E2BFileTool(E2BBaseTool):
    """Read, write, and manage files inside an E2B sandbox.

    Notes:
      - Most useful with `persistent=True` or an explicit `sandbox_id`. With
        the default ephemeral mode, files disappear when this tool call
        finishes.
    """

    name: str = "E2B Sandbox Files"
    description: str = (
        "Perform filesystem operations inside an E2B sandbox: read a file, "
        "write content to a path, append content to an existing file, list a "
        "directory, delete a path, make a directory, fetch file metadata, or "
        "check whether a path exists. For files larger than a few KB, create "
        "the file with action='write' and empty content, then send the body "
        "via multiple 'append' calls of ~4KB each to stay within tool-call "
        "payload limits."
    )
    args_schema: type_[BaseModel] = E2BFileToolSchema

    def _run(
        self,
        action: FileAction,
        path: str,
        content: str | None = None,
        binary: bool = False,
        depth: int = 1,
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
                return self._list(sandbox, path, depth=depth)
            if action == "delete":
                sandbox.files.remove(path)
                return {"status": "deleted", "path": path}
            if action == "mkdir":
                created = sandbox.files.make_dir(path)
                return {"status": "created", "path": path, "created": bool(created)}
            if action == "info":
                return self._info(sandbox, path)
            if action == "exists":
                return {"path": path, "exists": bool(sandbox.files.exists(path))}
            raise ValueError(f"Unknown action: {action}")
        finally:
            self._release_sandbox(sandbox, should_kill)

    def _read(self, sandbox: Any, path: str, *, binary: bool) -> dict[str, Any]:
        if binary:
            data: bytes = sandbox.files.read(path, format="bytes")
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
        try:
            content: str = sandbox.files.read(path)
            return {"path": path, "encoding": "utf-8", "content": content}
        except UnicodeDecodeError:
            data = sandbox.files.read(path, format="bytes")
            return {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "note": "File was not valid utf-8; returned as base64.",
            }

    def _write(
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        payload: str | bytes = base64.b64decode(content) if binary else content
        self._ensure_parent_dir(sandbox, path)
        sandbox.files.write(path, payload)
        size = (
            len(payload)
            if isinstance(payload, (bytes, bytearray))
            else len(payload.encode("utf-8"))
        )
        return {"status": "written", "path": path, "bytes": size}

    def _append(
        self, sandbox: Any, path: str, content: str, *, binary: bool
    ) -> dict[str, Any]:
        chunk: bytes = base64.b64decode(content) if binary else content.encode("utf-8")
        self._ensure_parent_dir(sandbox, path)
        try:
            existing: bytes = sandbox.files.read(path, format="bytes")
        except Exception:
            existing = b""
        payload = existing + chunk
        sandbox.files.write(path, payload)
        return {
            "status": "appended",
            "path": path,
            "appended_bytes": len(chunk),
            "total_bytes": len(payload),
        }

    @staticmethod
    def _ensure_parent_dir(sandbox: Any, path: str) -> None:
        parent = posixpath.dirname(path)
        if not parent or parent in ("/", "."):
            return
        try:
            sandbox.files.make_dir(parent)
        except Exception:
            logger.debug(
                "Best-effort parent-directory create failed for %s; "
                "assuming it already exists and proceeding with the write.",
                parent,
                exc_info=True,
            )

    def _list(self, sandbox: Any, path: str, *, depth: int) -> dict[str, Any]:
        entries = sandbox.files.list(path, depth=depth)
        return {
            "path": path,
            "entries": [self._entry_to_dict(e) for e in entries],
        }

    def _info(self, sandbox: Any, path: str) -> dict[str, Any]:
        return self._entry_to_dict(sandbox.files.get_info(path))

    @staticmethod
    def _entry_to_dict(entry: Any) -> dict[str, Any]:
        fields = (
            "name",
            "path",
            "type",
            "size",
            "mode",
            "permissions",
            "owner",
            "group",
            "modified_time",
            "symlink_target",
        )
        result: dict[str, Any] = {}
        for field in fields:
            value = getattr(entry, field, None)
            if value is not None and field == "modified_time":
                result[field] = (
                    value.isoformat() if hasattr(value, "isoformat") else str(value)
                )
            else:
                result[field] = value
        return result
