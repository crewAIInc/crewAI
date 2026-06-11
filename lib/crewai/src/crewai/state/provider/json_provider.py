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

def _build_path(location: str, branch: str, parent_id: str | None = None) -> Path:
    """Build the path to a checkpoint file.
    
    Returns a path like: location/branch/ts_uuid8_p-parent.json
    """
    base_dir = Path(location) / branch
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    pid = parent_id or "root"
    return base_dir / f"{ts}_{uid}_p-{pid}.json"

def _safe_branch(base: str, branch: str) -> None:
    base_resolved = Path(base).resolve()
    branch_resolved = (base_resolved / branch).resolve()
    if not str(branch_resolved).startswith(str(base_resolved)):
        raise ValueError(f"Invalid branch name: {branch}")

class JsonProvider:
    def __init__(self, location: str = "checkpoints"):
        self.location = location

    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a JSON checkpoint file atomically.
        
        This method uses a temporary file and os.replace to ensure that each 
        checkpoint write completes fully or not at all, avoiding partial 
        or corrupt files. 
        
        Note: This provides atomicity for individual writes but does not prevent 
        multiple concurrent writers from overwriting each other's complete 
        checkpoints. Concurrent-writer protection would require additional 
        coordination such as an advisory file lock or external synchronization.
        """
        _safe_branch(location, branch)
        file_path = _build_path(location, branch, parent_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, file_path)
            # Ensure the directory entry is durably persisted
            dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
                
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                logger.debug("Failed to remove temp file %s", tmp_path, exc_info=True)
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
        """Write a JSON checkpoint file atomically and asynchronously."""
        _safe_branch(location, branch)
        file_path = _build_path(location, branch, parent_id)
        
        # Use aiofiles for directory creation
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: file_path.parent.mkdir(parents=True, exist_ok=True)
        )
        
        tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}.json")
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(data)
                await f.flush()
                # fsync is not native to aiofiles; use executor
                await asyncio.get_event_loop().run_in_executor(
                    None, os.fsync, f.fileno()
                )
            
            # Atomic replace
            await asyncio.get_event_loop().run_in_executor(
                None, os.replace, tmp_path, file_path
            )
            
            # Sync parent directory
            def sync_dir():
                dfd = os.open(str(file_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            
            await asyncio.get_event_loop().run_in_executor(None, sync_dir)
                
        except Exception as e:
            try:
                # Use run_in_executor for non-async unlink
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tmp_path.unlink(missing_ok=True)
                )
            except Exception:
                logger.debug("Failed to remove temp file %s", tmp_path, exc_info=True)
            raise
            
        return str(file_path)

    def prune(self, location: str, branch: str, keep: int = 10) -> int:
        """Remove old checkpoints, keeping only the most recent 'keep' files."""
        _safe_branch(location, branch)
        branch_dir = Path(location) / branch
        if not branch_dir.exists():
            return 0
            
        # Use a pattern that only matches .json files and excludes .tmp_ files
        pattern = os.path.join(str(branch_dir), "*.json")
        files = [f for f in glob.glob(pattern) if ".tmp_" not in f]
        files = sorted(files, key=os.path.getmtime)
        
        if len(files) <= keep:
            return 0
            
        deleted_count = 0
        for file in files[:-keep]:
            try:
                os.remove(file)
                deleted_count += 1
            except OSError:
                logger.debug("Failed to remove checkpoint %s", file, exc_info=True)
                
        return deleted_count
