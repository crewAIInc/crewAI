"""Private, temporary storage for deferred tool results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from tempfile import TemporaryDirectory


TOOL_RESULT_PREVIEW_LINE_COUNT = 10
TOOL_RESULT_PREVIEW_MAX_CHARS = 2_000
TOOL_RESULT_READ_CONTEXT_WINDOW_RATIO = 0.05
_TOOL_RESULT_TEMP_DIR = Path("/tmp")  # noqa: S108 - intentional POSIX temp dir


@dataclass(frozen=True)
class ToolResultArtifact:
    """An internally-addressable deferred tool result."""

    artifact_id: str
    directory: Path
    relative_path: Path
    file_path: Path
    line_count: int
    estimated_token_count: int
    preview: str
    preview_truncated: bool


class ToolResultArtifactStore:
    """Store tool results outside the conversation for one execution.

    Artifact identifiers are opaque capabilities scoped to this store. Each
    result gets an isolated directory so a pre-bound file tool can expose only
    that result. The temporary directory is removed by :meth:`cleanup` when
    the execution ends.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Create an execution-scoped store under the system temp directory.

        Args:
            base_dir: Parent directory for the private execution directory.
                Intended for tests; production uses ``/tmp`` when available,
                otherwise the platform's system temp directory.
        """
        # Keep potentially large tool output out of the project and user data
        # directories. ``/tmp`` is the conventional ephemeral location on
        # POSIX; retain Python's platform-specific fallback where it is absent.
        directory = (
            str(base_dir)
            if base_dir is not None
            else str(_TOOL_RESULT_TEMP_DIR)
            if _TOOL_RESULT_TEMP_DIR.is_dir()
            else None
        )
        self._temporary_directory = TemporaryDirectory(
            prefix="crewai-tool-results-", dir=directory
        )
        self._directory = Path(self._temporary_directory.name)
        self._artifact_ids: set[str] = set()

    def store(self, content: str, estimated_token_count: int) -> ToolResultArtifact:
        """Persist content and return its internal binding details.

        Args:
            content: Tool result text to store.
            estimated_token_count: Token count already produced for this content.
        """
        artifact_id = f"tool-result-{token_urlsafe(16)}"
        artifact_directory = self._directory / artifact_id
        artifact_directory.mkdir(mode=0o700)
        path = artifact_directory / "result.md"
        path.write_text(content, encoding="utf-8")
        self._artifact_ids.add(artifact_id)
        preview_lines = content.splitlines(keepends=True)
        line_count = len(preview_lines)
        preview = "".join(preview_lines[:TOOL_RESULT_PREVIEW_LINE_COUNT])
        preview_truncated = len(preview_lines) > TOOL_RESULT_PREVIEW_LINE_COUNT
        if len(preview) > TOOL_RESULT_PREVIEW_MAX_CHARS:
            preview = preview[:TOOL_RESULT_PREVIEW_MAX_CHARS]
            preview_truncated = True
        return ToolResultArtifact(
            artifact_id=artifact_id,
            directory=self._directory,
            relative_path=path.relative_to(self._directory),
            file_path=path,
            line_count=line_count,
            estimated_token_count=estimated_token_count,
            preview=preview,
            preview_truncated=preview_truncated,
        )

    @property
    def directory(self) -> Path:
        """Return the private root directory for this execution's artifacts."""
        return self._directory

    def cleanup(self) -> None:
        """Delete every artifact in this execution. Safe to call repeatedly."""
        self._artifact_ids.clear()
        self._temporary_directory.cleanup()
