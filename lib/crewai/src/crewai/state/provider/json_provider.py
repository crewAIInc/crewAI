"""Filesystem JSON state provider."""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import logging
import os
from pathlib import Path
from typing import Literal
import uuid

import aiofiles
import aiofiles.os

from crewai.state.provider.core import BaseProvider


logger = logging.getLogger(__name__)


def _safe_branch(base: str, branch: str) -> None:
    """Validate that a branch name doesn't escape the base directory.

    Raises:
        ValueError: If the branch resolves outside the base directory.
    """
    base_resolved = str(Path(base).resolve())
    target_resolved = str((Path(base) / branch).resolve())
    if (
        not target_resolved.startswith(base_resolved + os.sep)
        and target_resolved != base_resolved
    ):
        raise ValueError(f"Branch name escapes checkpoint directory: {branch!r}")


class JsonProvider(BaseProvider):
    """Persists runtime state checkpoints as JSON files on the local filesystem."""

    provider_type: Literal["json"] = "json"

    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a JSON checkpoint file.

        Args:
            data: The serialized JSON string to persist.
            location: Base directory where checkpoints are saved.
            parent_id: ID of the parent checkpoint for lineage tracking.
                Encoded in the filename for queryable lineage without
                parsing the blob.
            branch: Branch label. Files are stored under ``location/branch/``.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(location, branch, parent_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(data)
        return str(file_path)

    async def acheckpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a JSON checkpoint file asynchronously.

        Args:
            data: The serialized JSON string to persist.
            location: Base directory where checkpoints are saved.
            parent_id: ID of the parent checkpoint for lineage tracking.
                Encoded in the filename for queryable lineage without
                parsing the blob.
            branch: Branch label. Files are stored under ``location/branch/``.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(location, branch, parent_id)
        await aiofiles.os.makedirs(str(file_path.parent), exist_ok=True)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(data)
        return str(file_path)

    def prune(self, location: str, max_keep: int, *, branch: str = "main") -> int:
        """Remove oldest checkpoint files beyond *max_keep* on a branch."""
        _safe_branch(location, branch)
        branch_dir = os.path.join(location, branch)
        pattern = os.path.join(branch_dir, "*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        removed = 0
        for path in files if max_keep == 0 else files[:-max_keep]:
            try:
                os.remove(path)
                removed += 1
            except OSError:  # noqa: PERF203
                logger.debug("Failed to remove %s", path, exc_info=True)
        return removed

    def extract_id(self, location: str) -> str:
        """Extract the checkpoint ID from a file path.

        The filename format is ``{ts}_{uuid8}_p-{parent}.json``.
        The checkpoint ID is the ``{ts}_{uuid8}`` prefix.
        """
        stem = Path(location).stem
        idx = stem.find("_p-")
        return stem[:idx] if idx != -1 else stem

    def from_checkpoint(self, location: str) -> str:
        """Read a JSON checkpoint file.

        Args:
            location: Filesystem path to the checkpoint file.

        Returns:
            The raw JSON string.
        """
        return Path(location).read_text()

    async def afrom_checkpoint(self, location: str) -> str:
        """Read a JSON checkpoint file asynchronously.

        Args:
            location: Filesystem path to the checkpoint file.

        Returns:
            The raw JSON string.
        """
        async with aiofiles.open(location) as f:
            return await f.read()


def _build_path(
    directory: str, branch: str = "main", parent_id: str | None = None
) -> Path:
    """Build a timestamped checkpoint file path under a branch subdirectory.

    Filename format: ``{ts}_{uuid8}_p-{parent_id}.json``

    Args:
        directory: Base directory for checkpoints.
        branch: Branch label used as a subdirectory name.
        parent_id: Parent checkpoint ID to encode in the filename.

    Returns:
        The target file path.
    """
    _safe_branch(directory, branch)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    parent_suffix = parent_id or "none"
    filename = f"{ts}_{short_uuid}_p-{parent_suffix}.json"
    return Path(directory) / branch / filename
