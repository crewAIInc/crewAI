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


class JsonProvider(BaseProvider):
    """Persists runtime state checkpoints as JSON files on the local filesystem."""

    provider_type: Literal["json"] = "json"

    def checkpoint(self, data: str, location: str) -> str:
        """Write a JSON checkpoint file.

        Args:
            data: The serialized JSON string to persist.
            location: Directory where the checkpoint will be saved.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(location)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(data)
        return str(file_path)

    async def acheckpoint(self, data: str, location: str) -> str:
        """Write a JSON checkpoint file asynchronously.

        Args:
            data: The serialized JSON string to persist.
            location: Directory where the checkpoint will be saved.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(location)
        await aiofiles.os.makedirs(str(file_path.parent), exist_ok=True)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(data)
        return str(file_path)

    def prune(self, location: str, max_keep: int) -> None:
        """Remove oldest checkpoint files beyond *max_keep*."""
        pattern = os.path.join(location, "*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        for path in files if max_keep == 0 else files[:-max_keep]:
            try:
                os.remove(path)
            except OSError:  # noqa: PERF203
                logger.debug("Failed to remove %s", path, exc_info=True)

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


def _build_path(directory: str) -> Path:
    """Build a timestamped checkpoint file path.

    Args:
        directory: Parent directory for the checkpoint file.

    Returns:
        The target file path.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{ts}_{uuid.uuid4().hex[:8]}.json"
    return Path(directory) / filename
