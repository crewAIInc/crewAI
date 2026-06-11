import os
import uuid
import glob
import logging
from datetime import datetime, timezone
from pathlib import Path
import json
import aiofiles
import asyncio

logger = logging.getLogger(__name__)

def _safe_branch(base: str, branch: str) -> None:
    base_resolved = str(Path(base).resolve())
    target_resolved = str((Path(base) / branch).resolve())
    if (
        not target_resolved.startswith(base_resolved + os.sep)
        and target_resolved != base_resolved
    ):
        raise ValueError(f"Branch name escapes checkpoint directory: {branch!r}")

def _build_path(directory: str, branch: str = "main", parent_id: str | None = None) -> Path:
    _safe_branch(directory, branch)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    parent_suffix = parent_id or "none"
    filename = f"{ts}_{short_uuid}_p-{parent_suffix}.json"
    return Path(directory) / branch / filename

class JsonProvider:
    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """
        Write a JSON checkpoint file atomically.
        
        The use of a temporary file and os.replace provides atomicity for each 
        individual write (avoiding partial/corrupt files), but does not 
        prevent multiple concurrent writers from overwriting each other's 
        complete checkpoints. Concurrent-writer protection would require 
        additional coordination such as an advisory file lock or external 
        synchronization.
        """
        file_path = _build_path(location, branch, parent_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}.json")
        try:
            with open(tmp_path, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, file_path)
            
            # Ensure directory entry is also persisted
            dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                if tmp_path.exists():
                    os.remove(tmp_path)
            except Exception as removal_err:
                logger.debug("Failed to remove temp file %s: %s", tmp_path, removal_err)
            raise
            
        return str(file_path)

    async def acheckpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """
        Write a JSON checkpoint file atomically and asynchronously.
        
        Uses a temporary file and atomic replace to ensure that each checkpoint 
        write completes fully or not at all.
        """
        file_path = _build_path(location, branch, parent_id)
        await aiofiles.os.makedirs(str(file_path.parent), exist_ok=True)
        
        tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}.json")
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                await f.write(data)
                await f.flush()
                # fsync is not natively async in aiofiles, wrap in executor
                await asyncio.get_event_loop().run_in_executor(None, os.fsync, f.fileno())
            
            await asyncio.get_event_loop().run_in_executor(None, os.replace, tmp_path, file_path)
            
            # Sync parent directory
            def sync_dir():
                dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            
            await asyncio.get_event_loop().run_in_executor(None, sync_dir)
            
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as removal_err:
                logger.debug("Failed to remove temp file %s: %s", tmp_path, removal_err)
            raise
            
        return str(file_path)

    def prune(self, location: str, branch: str = "main", keep: int = 10) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        branch_dir = Path(location) / branch
        if not branch_dir.exists():
            return
            
        # Filter out temp files (.tmp_) and hidden files
        files = [
            f for f in glob.glob(str(branch_dir / "*.json")) 
            if not os.path.basename(f).startswith(".") and ".tmp_" not in os.path.basename(f)
        ]
        files = sorted(files, key=os.path.getmtime)
        
        # Delete all but the most recent 'keep' files
        for file in files[:-keep]:
            try:
                os.remove(file)
            except OSError as e:
                logger.debug("Failed to prune file %s: %s", file, e)
