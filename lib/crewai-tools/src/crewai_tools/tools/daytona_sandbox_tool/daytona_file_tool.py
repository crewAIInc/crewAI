from __future__ import annotations

import base64
from builtins import type as type_
import logging
import posixpath
import shlex
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.daytona_sandbox_tool.daytona_base_tool import DaytonaBaseTool


logger = logging.getLogger(__name__)


FileAction = Literal[
    "read",
    "write",
    "append",
    "list",
    "delete",
    "mkdir",
    "info",
    "exists",
    "move",
    "find",
    "search",
    "chmod",
    "replace",
]


class DaytonaFileToolSchema(BaseModel):
    action: FileAction = Field(
        ...,
        description=(
            "The filesystem action to perform: "
            "'read' (returns file contents); "
            "'write' (create or replace a file with content); "
            "'append' (append content to an existing file — use this for "
            "writing large files in chunks to avoid hitting tool-call size "
            "limits); "
            "'list' (lists a directory); "
            "'delete' (removes a file/dir); "
            "'mkdir' (creates a directory); "
            "'info' (returns file metadata); "
            "'exists' (returns whether a path exists); "
            "'move' (rename or relocate a file/dir; requires 'destination'); "
            "'find' (grep file CONTENTS recursively; requires 'pattern'); "
            "'search' (find files by NAME pattern; requires 'pattern'); "
            "'chmod' (change permissions/owner/group; pass at least one of "
            "'mode', 'owner', 'group'); "
            "'replace' (find-and-replace text across files; requires "
            "'paths', 'pattern', and 'replacement')."
        ),
    )
    path: str | None = Field(
        default=None,
        description=(
            "Absolute path inside the sandbox. Required for all actions "
            "except 'replace' (which uses 'paths' instead)."
        ),
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
            "For 'write': treat content as base64 and upload raw bytes. "
            "For 'read': return contents as base64 instead of decoded utf-8."
        ),
    )
    recursive: bool = Field(
        default=False,
        description="For action='delete': remove directories recursively.",
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Octal permission string. For 'mkdir' it sets the new directory "
            "permissions (defaults to '0755' if omitted). For 'chmod' it sets "
            "the target's mode (e.g. '755' to make a script executable). "
            "Ignored for other actions."
        ),
    )
    destination: str | None = Field(
        default=None,
        description="For action='move': absolute destination path.",
    )
    pattern: str | None = Field(
        default=None,
        description=(
            "For 'find': substring matched against file CONTENTS. "
            "For 'search': glob-style pattern matched against file NAMES "
            "(e.g. '*.py'). "
            "For 'replace': text to replace inside files."
        ),
    )
    replacement: str | None = Field(
        default=None,
        description="For action='replace': replacement text for 'pattern'.",
    )
    paths: list[str] | None = Field(
        default=None,
        description=(
            "For action='replace': list of absolute file paths in which to "
            "replace 'pattern' with 'replacement'."
        ),
    )
    owner: str | None = Field(
        default=None,
        description="For action='chmod': new file owner (user name).",
    )
    group: str | None = Field(
        default=None,
        description="For action='chmod': new file group.",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> DaytonaFileToolSchema:
        if self.action != "replace" and not self.path:
            raise ValueError(f"action={self.action!r} requires 'path'.")
        if self.action == "append" and self.content is None:
            raise ValueError(
                "action='append' requires 'content'. Pass the chunk to append "
                "in the 'content' field."
            )
        if self.action == "move" and not self.destination:
            raise ValueError("action='move' requires 'destination'.")
        if self.action == "find" and not self.pattern:
            raise ValueError(
                "action='find' requires 'pattern' (text to search for inside files)."
            )
        if self.action == "search" and not self.pattern:
            raise ValueError("action='search' requires 'pattern' (glob, e.g. '*.py').")
        if self.action == "chmod" and not (self.mode or self.owner or self.group):
            raise ValueError(
                "action='chmod' requires at least one of 'mode', 'owner', or 'group'."
            )
        if self.action == "replace":
            if not self.paths:
                raise ValueError(
                    "action='replace' requires 'paths' (list of file paths)."
                )
            if not self.pattern:
                raise ValueError("action='replace' requires 'pattern'.")
            if self.replacement is None:
                raise ValueError("action='replace' requires 'replacement'.")
        return self


class DaytonaFileTool(DaytonaBaseTool):
    """Read, write, and manage files inside a Daytona sandbox.

    Notes:
      - Most useful with `persistent=True` or an explicit `sandbox_id`. With the
        default ephemeral mode, files disappear when this tool call finishes.
    """

    name: str = "Daytona Sandbox Files"
    description: str = (
        "Perform filesystem operations inside a Daytona sandbox: read, "
        "write, append, list, delete, mkdir, info, exists, move, find "
        "(content grep), search (filename glob), chmod (permissions/owner/"
        "group), and replace (bulk find-and-replace across files). "
        "For files larger than a few KB, create the file with action='write' "
        "and empty content, then send the body via multiple 'append' calls of "
        "~4KB each to stay within tool-call payload limits."
    )
    args_schema: type_[BaseModel] = DaytonaFileToolSchema

    def _run(
        self,
        action: FileAction,
        path: str | None = None,
        content: str | None = None,
        binary: bool = False,
        recursive: bool = False,
        mode: str | None = None,
        destination: str | None = None,
        pattern: str | None = None,
        replacement: str | None = None,
        paths: list[str] | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> Any:
        sandbox, should_delete = self._acquire_sandbox()
        try:
            if action == "read":
                if path is None:
                    raise ValueError("action='read' requires 'path'")
                return self._read(sandbox, path, binary=binary)
            if action == "write":
                if path is None:
                    raise ValueError("action='write' requires 'path'")
                return self._write(sandbox, path, content or "", binary=binary)
            if action == "append":
                if path is None:
                    raise ValueError("action='append' requires 'path'")
                return self._append(sandbox, path, content or "", binary=binary)
            if action == "list":
                if path is None:
                    raise ValueError("action='list' requires 'path'")
                return self._list(sandbox, path)
            if action == "delete":
                if path is None:
                    raise ValueError("action='delete' requires 'path'")
                sandbox.fs.delete_file(path, recursive=recursive)
                return {"status": "deleted", "path": path}
            if action == "mkdir":
                if path is None:
                    raise ValueError("action='mkdir' requires 'path'")
                mkdir_mode = mode or "0755"
                sandbox.fs.create_folder(path, mkdir_mode)
                return {"status": "created", "path": path, "mode": mkdir_mode}
            if action == "info":
                if path is None:
                    raise ValueError("action='info' requires 'path'")
                return self._info(sandbox, path)
            if action == "exists":
                if path is None:
                    raise ValueError("action='exists' requires 'path'")
                return self._exists(sandbox, path)
            if action == "move":
                if path is None or destination is None:
                    raise ValueError("action='move' requires 'path' and 'destination'")
                sandbox.fs.move_files(path, destination)
                return {"status": "moved", "from": path, "to": destination}
            if action == "find":
                if path is None or pattern is None:
                    raise ValueError("action='find' requires 'path' and 'pattern'")
                return self._find(sandbox, path, pattern)
            if action == "search":
                if path is None or pattern is None:
                    raise ValueError("action='search' requires 'path' and 'pattern'")
                return self._search(sandbox, path, pattern)
            if action == "chmod":
                if path is None:
                    raise ValueError("action='chmod' requires 'path'")
                return self._chmod(sandbox, path, mode=mode, owner=owner, group=group)
            if action == "replace":
                if paths is None or pattern is None or replacement is None:
                    raise ValueError(
                        "action='replace' requires 'paths', 'pattern', and "
                        "'replacement'"
                    )
                return self._replace(sandbox, paths, pattern, replacement)
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

        # Server-side `cat >>` keeps this O(chunk_size) per call. The naive
        # download-concat-reupload alternative is O(N^2) in total transfer.
        # /tmp/ is on the sandbox's ephemeral filesystem, not the host.
        temp_path = f"/tmp/.crewai-append-{uuid.uuid4().hex}"  # noqa: S108
        sandbox.fs.upload_file(chunk, temp_path)

        quoted_temp = shlex.quote(temp_path)
        quoted_target = shlex.quote(path)
        response = sandbox.process.exec(
            f"cat {quoted_temp} >> {quoted_target}; "
            f"rc=$?; rm -f {quoted_temp}; exit $rc"
        )

        exit_code = getattr(response, "exit_code", 0)
        if exit_code != 0:
            try:
                sandbox.fs.delete_file(temp_path)
            except Exception:
                logger.debug(
                    "Best-effort temp-file cleanup failed after append "
                    "error; the file may need manual deletion.",
                    exc_info=True,
                )
            raise RuntimeError(
                f"append failed: exit_code={exit_code}, "
                f"output={getattr(response, 'result', '')!r}"
            )

        try:
            info = sandbox.fs.get_file_info(path)
            total_bytes = getattr(info, "size", None)
        except Exception:
            total_bytes = None

        return {
            "status": "appended",
            "path": path,
            "appended_bytes": len(chunk),
            "total_bytes": total_bytes,
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

    def _exists(self, sandbox: Any, path: str) -> dict[str, Any]:
        try:
            info = sandbox.fs.get_file_info(path)
        except Exception:
            return {"path": path, "exists": False}
        return {
            "path": path,
            "exists": True,
            "is_dir": getattr(info, "is_dir", False),
        }

    def _find(self, sandbox: Any, path: str, pattern: str) -> dict[str, Any]:
        matches = sandbox.fs.find_files(path, pattern)
        return {
            "path": path,
            "pattern": pattern,
            "matches": [
                {
                    "file": getattr(m, "file", None),
                    "line": getattr(m, "line", None),
                    "content": getattr(m, "content", None),
                }
                for m in matches
            ],
        }

    def _search(self, sandbox: Any, path: str, pattern: str) -> dict[str, Any]:
        response = sandbox.fs.search_files(path, pattern)
        files = getattr(response, "files", None) or []
        return {"path": path, "pattern": pattern, "files": list(files)}

    def _chmod(
        self,
        sandbox: Any,
        path: str,
        *,
        mode: str | None,
        owner: str | None,
        group: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, str] = {}
        if mode is not None:
            kwargs["mode"] = mode
        if owner is not None:
            kwargs["owner"] = owner
        if group is not None:
            kwargs["group"] = group
        sandbox.fs.set_file_permissions(path, **kwargs)
        return {"status": "permissions_set", "path": path, **kwargs}

    def _replace(
        self,
        sandbox: Any,
        paths: list[str],
        pattern: str,
        replacement: str,
    ) -> dict[str, Any]:
        results = sandbox.fs.replace_in_files(paths, pattern, replacement)
        return {
            "pattern": pattern,
            "replacement": replacement,
            "results": [
                {
                    "file": getattr(r, "file", None),
                    "success": getattr(r, "success", None),
                    "error": getattr(r, "error", None),
                }
                for r in (results or [])
            ],
        }

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
