"""Filesystem JSON state provider."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import aiofiles
import aiofiles.os

from crewai.state.provider.core import BaseProvider


class JsonProvider(BaseProvider):
    """Persists runtime state checkpoints as JSON files on the local filesystem."""

    def checkpoint(self, data: str, directory: str) -> str:
        """Write a JSON checkpoint file to the directory.

        Args:
            data: The serialized JSON string to persist.
            directory: Filesystem path where the checkpoint will be saved.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(directory)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(data)
        return str(file_path)

    async def acheckpoint(self, data: str, directory: str) -> str:
        """Write a JSON checkpoint file to the directory asynchronously.

        Args:
            data: The serialized JSON string to persist.
            directory: Filesystem path where the checkpoint will be saved.

        Returns:
            The path to the written checkpoint file.
        """
        file_path = _build_path(directory)
        await aiofiles.os.makedirs(str(file_path.parent), exist_ok=True)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(data)
        return str(file_path)


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
