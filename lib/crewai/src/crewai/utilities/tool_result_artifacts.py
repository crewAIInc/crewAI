"""Private, temporary storage for deferred tool results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class ToolResultArtifact:
    """An internally-addressable deferred tool result."""

    artifact_id: str
    file_path: Path


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
                Intended for tests; production defaults to the system temp
                directory.
        """
        directory = str(base_dir) if base_dir is not None else None
        self._temporary_directory = TemporaryDirectory(
            prefix="crewai-tool-results-", dir=directory
        )
        self._directory = Path(self._temporary_directory.name)
        self._artifact_ids: set[str] = set()

    def store(self, content: str) -> ToolResultArtifact:
        """Persist content and return its internal binding details."""
        artifact_id = f"tool-result-{token_urlsafe(16)}"
        artifact_directory = self._directory / artifact_id
        artifact_directory.mkdir(mode=0o700)
        path = artifact_directory / "result.md"
        path.write_text(content, encoding="utf-8")
        self._artifact_ids.add(artifact_id)
        return ToolResultArtifact(artifact_id=artifact_id, file_path=path)

    def cleanup(self) -> None:
        """Delete every artifact in this execution. Safe to call repeatedly."""
        self._artifact_ids.clear()
        self._temporary_directory.cleanup()
