from __future__ import annotations

import fnmatch
import os
import shutil
from builtins import type as type_
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from crewai_tools.tools.vetto_sandbox_tool.vetto_base_tool import VettoBaseTool


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
]


class VettoFileToolSchema(BaseModel):
    action: FileAction = Field(
        ...,
        description=(
            "The filesystem action to perform: "
            "'read' (returns file contents); "
            "'write' (writes or overwrites a file); "
            "'append' (appends content to a file); "
            "'list' (lists directory entries); "
            "'delete' (removes a file or directory); "
            "'mkdir' (creates a directory); "
            "'info' (returns file size and metadata); "
            "'exists' (returns whether path exists); "
            "'move' (renames or moves file to destination); "
            "'find' (grep text inside files); "
            "'search' (search for files by filename pattern)."
        ),
    )
    path: str | None = Field(
        default=None,
        description="Target relative or absolute path within the sandbox root workspace.",
    )
    content: str | None = Field(
        default=None,
        description="Text content for 'write' or 'append' actions.",
    )
    destination: str | None = Field(
        default=None,
        description="Destination path for 'move' action.",
    )
    pattern: str | None = Field(
        default=None,
        description="Search pattern for 'find' (content substring) or 'search' (glob filename).",
    )
    recursive: bool = Field(
        default=False,
        description="For 'delete': recursively delete non-empty directory.",
    )

    @model_validator(mode="after")
    def _validate_action_args(self) -> VettoFileToolSchema:
        """Validate required arguments based on the selected file action.

        Returns:
            The validated VettoFileToolSchema instance.

        Raises:
            ValueError: If required fields for an action are missing.
        """
        if not self.path:
            raise ValueError(f"action={self.action!r} requires 'path'.")
        if self.action in ("write", "append") and self.content is None:
            raise ValueError(f"action={self.action!r} requires 'content'.")
        if self.action == "move" and not self.destination:
            raise ValueError("action='move' requires 'destination'.")
        if self.action in ("find", "search") and not self.pattern:
            raise ValueError(f"action={self.action!r} requires 'pattern'.")
        return self


class VettoFileTool(VettoBaseTool):
    """Filesystem manager for workspace paths inside a Vetto sandbox."""

    name: str = "Vetto Sandbox Files"
    description: str = (
        "Perform secured filesystem operations inside the Vetto sandbox "
        "workspace (read, write, append, list, delete, mkdir, info, exists, "
        "move, find, search). All paths are strictly validated against directory "
        "traversal and sandbox escape attempts."
    )
    args_schema: type_[BaseModel] = VettoFileToolSchema

    def _resolve_safe_path(self, target_path: str) -> Path:
        """Resolve path and ensure it does not escape the sandbox root.

        Args:
            target_path: Path string to resolve.

        Returns:
            Resolved absolute Path within the workspace.

        Raises:
            PermissionError: If path escapes sandbox workspace boundary.
        """
        root = Path(self.working_dir or os.getcwd()).resolve()
        try:
            candidate = (root / target_path).resolve(strict=False) if not os.path.isabs(target_path) else Path(target_path).resolve(strict=False)
            candidate.relative_to(root)
        except (ValueError, RuntimeError):
            raise PermissionError(
                f"Path {target_path} escapes sandbox workspace boundary {root}"
            )
        return candidate

    def _run(
        self,
        action: FileAction,
        path: str | None = None,
        content: str | None = None,
        destination: str | None = None,
        pattern: str | None = None,
        recursive: bool = False,
    ) -> Any:
        """Perform a contained filesystem operation within the sandbox.

        Args:
            action: The filesystem operation to perform.
            path: Target file or directory path.
            content: Content string for write/append actions.
            destination: Target destination path for move action.
            pattern: Search string or glob pattern.
            recursive: Boolean flag for recursive directory deletion.

        Returns:
            Dictionary containing operation status or queried data.
        """
        if not path:
            raise ValueError("Missing required 'path' parameter.")

        safe_path = self._resolve_safe_path(path)
        root = Path(self.working_dir or os.getcwd()).resolve()

        if action in ("delete", "move") and safe_path == root:
            raise PermissionError(
                f"Cannot {action} the root sandbox workspace directory {root}."
            )

        if action == "read":
            if not safe_path.exists():
                return {"error": f"Path not found: {path}", "exists": False}
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                return {"path": str(safe_path), "content": f.read()}

        elif action == "write":
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content or "")
            return {"status": "written", "path": str(safe_path), "bytes": len(content or "")}

        elif action == "append":
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, "a", encoding="utf-8") as f:
                f.write(content or "")
            return {"status": "appended", "path": str(safe_path), "bytes_appended": len(content or "")}

        elif action == "list":
            if not safe_path.exists():
                return {"error": f"Directory not found: {path}", "exists": False}
            if not safe_path.is_dir():
                return {"error": f"Path is not a directory: {path}"}
            entries = []
            for entry in safe_path.iterdir():
                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else None,
                })
            return {"path": str(safe_path), "entries": entries}

        elif action == "delete":
            if not safe_path.exists():
                return {"status": "already_absent", "path": str(safe_path)}
            if safe_path.is_dir():
                if recursive:
                    shutil.rmtree(safe_path)
                else:
                    safe_path.rmdir()
            else:
                safe_path.unlink()
            return {"status": "deleted", "path": str(safe_path)}

        elif action == "mkdir":
            safe_path.mkdir(parents=True, exist_ok=True)
            return {"status": "created", "path": str(safe_path)}

        elif action == "info":
            if not safe_path.exists():
                return {"exists": False, "path": str(safe_path)}
            stat = safe_path.stat()
            return {
                "exists": True,
                "path": str(safe_path),
                "is_dir": safe_path.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }

        elif action == "exists":
            return {"path": str(safe_path), "exists": safe_path.exists(), "is_dir": safe_path.is_dir() if safe_path.exists() else False}

        elif action == "move":
            if not destination:
                raise ValueError("action='move' requires 'destination'")
            safe_dest = self._resolve_safe_path(destination)
            safe_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(safe_path), str(safe_dest))
            return {"status": "moved", "from": str(safe_path), "to": str(safe_dest)}

        elif action == "find":
            if not pattern:
                raise ValueError("action='find' requires 'pattern'")
            matches = []
            if safe_path.is_file():
                try:
                    safe_path.resolve(strict=False).relative_to(root)
                    with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            if pattern in line:
                                matches.append({"file": str(safe_path), "line": line_no, "text": line.strip()})
                except (ValueError, OSError, RuntimeError, UnicodeDecodeError):
                    pass
            elif safe_path.is_dir():
                for root_dir, _, files in os.walk(safe_path):
                    for file in files:
                        fp = Path(root_dir) / file
                        # Defense-in-depth: skip files or symlinks resolving outside workspace boundary
                        try:
                            fp.resolve(strict=False).relative_to(root)
                        except (ValueError, OSError, RuntimeError):
                            continue
                        try:
                            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                                for line_no, line in enumerate(f, 1):
                                    if pattern in line:
                                        matches.append({"file": str(fp), "line": line_no, "text": line.strip()})
                        except (OSError, UnicodeDecodeError):
                            continue
            return {"pattern": pattern, "matches": matches[:100]}

        elif action == "search":
            if not pattern:
                raise ValueError("action='search' requires 'pattern'")
            results = []
            search_root = safe_path if safe_path.is_dir() else safe_path.parent
            for root_dir, _, files in os.walk(search_root):
                for f in files:
                    if fnmatch.fnmatch(f, pattern):
                        cand = Path(root_dir) / f
                        # Defense-in-depth: skip files or symlinks resolving outside workspace boundary
                        try:
                            cand.resolve(strict=False).relative_to(root)
                        except (ValueError, OSError, RuntimeError):
                            continue
                        results.append(str(cand))
            return {"pattern": pattern, "results": results[:100]}

        raise ValueError(f"Unknown action: {action}")
