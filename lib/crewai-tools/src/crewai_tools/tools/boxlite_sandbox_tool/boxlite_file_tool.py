from __future__ import annotations

import base64
from builtins import type as type_
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.boxlite_sandbox_tool.boxlite_base_tool import BoxLiteBaseTool


FileAction = Literal[
    "read", "write", "append", "list", "delete", "mkdir", "info", "exists"
]


class BoxLiteFileToolSchema(BaseModel):
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
    path: str = Field(..., description="Absolute path inside the box.")
    content: str | None = Field(
        default=None,
        description=(
            "Content to write or append. If omitted for 'write', an empty file "
            "is created. For files larger than a few KB, prefer one 'write' with "
            "empty content followed by multiple 'append' calls of ~4KB each to "
            "stay within tool-call payload limits."
        ),
    )
    binary: bool = Field(
        default=False,
        description=(
            "For 'write'/'append': treat content as base64 and write the decoded "
            "raw bytes. For 'read': return contents as base64 instead of decoded "
            "utf-8."
        ),
    )
    depth: int = Field(
        default=1,
        description="For action='list': how many levels deep to recurse (default 1).",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> BoxLiteFileToolSchema:
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        return self


class BoxLiteFileTool(BoxLiteBaseTool):
    """Read, write, and manage files inside a BoxLite micro-VM.

    BoxLite exposes command execution rather than a files API, so these
    operations run over the box's shell, with content transferred as base64 so
    binary data and arbitrary text survive intact. Most useful with
    ``persistent=True``; with the default ephemeral mode the box — and its
    files — disappear when the call finishes.
    """

    name: str = "BoxLite Sandbox Files"
    description: str = (
        "Perform filesystem operations inside a BoxLite micro-VM: read a file, "
        "write content to a path, append content to an existing file, list a "
        "directory, delete a path, make a directory, fetch file metadata, or "
        "check whether a path exists. For files larger than a few KB, create the "
        "file with action='write' and empty content, then send the body via "
        "multiple 'append' calls of ~4KB each to stay within tool-call payload "
        "limits."
    )
    args_schema: type_[BaseModel] = BoxLiteFileToolSchema

    def _run(
        self,
        action: FileAction,
        path: str,
        content: str | None = None,
        binary: bool = False,
        depth: int = 1,
    ) -> Any:
        box, should_remove = self._acquire_box()
        try:
            if action == "read":
                return self._read(box, path, binary=binary)
            if action == "write":
                return self._write(
                    box, path, content or "", binary=binary, append=False
                )
            if action == "append":
                return self._write(box, path, content or "", binary=binary, append=True)
            if action == "list":
                return self._list(box, path, depth=depth)
            if action == "delete":
                return self._simple(box, "deleted", path, 'rm -rf -- "$1"')
            if action == "mkdir":
                return self._simple(box, "created", path, 'mkdir -p -- "$1"')
            if action == "info":
                return self._info(box, path)
            if action == "exists":
                return self._exists(box, path)
            raise ValueError(f"Unknown action: {action}")
        finally:
            self._release_box(box, should_remove)

    @staticmethod
    def _sh(box: Any, script: str, *sh_args: str) -> Any:
        # $0 is a placeholder name ("sh"); the caller's values arrive as the
        # positional parameters "$1", "$2", ... so nothing is re-parsed by the
        # shell and paths/content need no escaping.
        return box.exec("/bin/sh", "-c", script, "sh", *sh_args)

    def _read(self, box: Any, path: str, *, binary: bool) -> dict[str, Any]:
        if binary:
            result = self._sh(box, 'base64 -- "$1"', path)
            if result.exit_code != 0:
                return {"path": path, "error": (result.stderr or "read failed").strip()}
            return {
                "path": path,
                "encoding": "base64",
                "content": result.stdout.strip(),
            }
        result = self._sh(box, 'cat -- "$1"', path)
        if result.exit_code != 0:
            return {"path": path, "error": (result.stderr or "read failed").strip()}
        return {"path": path, "encoding": "utf-8", "content": result.stdout}

    def _write(
        self, box: Any, path: str, content: str, *, binary: bool, append: bool
    ) -> dict[str, Any]:
        payload = (
            content
            if binary
            else base64.b64encode(content.encode("utf-8")).decode("ascii")
        )
        redirect = ">>" if append else ">"
        script = (
            'mkdir -p -- "$(dirname -- "$1")" && '
            f'printf %s "$2" | base64 -d {redirect} "$1"'
        )
        result = self._sh(box, script, path, payload)
        if result.exit_code != 0:
            return {"path": path, "error": (result.stderr or "write failed").strip()}
        try:
            written = len(base64.b64decode(payload))
        except (ValueError, TypeError):
            written = 0
        return {
            "status": "appended" if append else "written",
            "path": path,
            "bytes": written,
        }

    def _list(self, box: Any, path: str, *, depth: int) -> dict[str, Any]:
        result = self._sh(
            box, 'find -- "$1" -maxdepth "$2" -mindepth 1', path, str(depth)
        )
        if result.exit_code != 0:
            return {"path": path, "error": (result.stderr or "list failed").strip()}
        entries = [line for line in result.stdout.splitlines() if line]
        return {"path": path, "entries": entries}

    def _info(self, box: Any, path: str) -> dict[str, Any]:
        result = self._sh(box, 'ls -ldn -- "$1"', path)
        if result.exit_code != 0:
            return {"path": path, "error": (result.stderr or "info failed").strip()}
        return {"path": path, "info": result.stdout.strip()}

    def _exists(self, box: Any, path: str) -> dict[str, Any]:
        result = self._sh(box, '[ -e "$1" ] && echo true || echo false', path)
        return {"path": path, "exists": result.stdout.strip() == "true"}

    def _simple(self, box: Any, status: str, path: str, script: str) -> dict[str, Any]:
        result = self._sh(box, script, path)
        if result.exit_code != 0:
            return {
                "path": path,
                "error": (result.stderr or "operation failed").strip(),
            }
        return {"status": status, "path": path}
