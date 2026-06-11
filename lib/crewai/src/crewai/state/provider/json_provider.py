import os
import uuid
import glob
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json
import aiofiles
import asyncio

logger = logging.getLogger(__name__)

def _validate_branch(location: str, branch: str) -> Path:
    \"\"\"Validate branch doesn't escape location and return resolved branch_dir.\"\"\"
    root = Path(location).resolve()
    branch_dir = (root / branch).resolve()
    try:
        branch_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Branch '{branch}' escapes checkpoint directory") from exc
    return branch_dir

def _build_path(location: str, branch: str, parent_id: str | None = None) -> Path:
    base_dir = _validate_branch(location, branch)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    pid = parent_id or "root"
    return base_dir / "{}_{}_p-{}.json".format(ts, uid, pid)

class JsonProvider:
    def __init__(self, location: str = "checkpoints"):
        self.location = location

    def checkpoint(self, data: str, location: str, *, parent_id: str | None = None, branch: str = "main") -> str:
        \"\"\"Write a JSON checkpoint file atomically.
        
        The use of a temporary file and os.replace provides atomicity for each individual write 
        (avoiding partial/corrupt files) but does not prevent multiple concurrent writers from 
        overwriting each other's complete checkpoints.
        \"\"\"
        file_path = _build_path(location, branch, parent_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_name = ".{}.tmp_{}".format(file_path.name, uuid.uuid4().hex[:8])
        tmp_path = file_path.parent / tmp_name
        
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, file_path)
            
            dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception as e:
                logger.debug("Failed to remove temp file {}: {}".format(tmp_path, e))
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
        \"\"\"Write a JSON checkpoint file atomically and asynchronously.\"\"\"
        file_path = _build_path(location, branch, parent_id)
        await aiofiles.os.makedirs(str(file_path.parent), exist_ok=True)
        
        tmp_name = ".{}.tmp_{}".format(file_path.name, uuid.uuid4().hex[:8])
        tmp_path = file_path.parent / tmp_name
        
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                await f.write(data)
                await f.flush()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, os.fsync, f.fileno())
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.replace, tmp_path, file_path)
            
            def sync_dir():
                dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sync_dir)
            
        except Exception:
            try:
                if await aiofiles.os.path.exists(tmp_path):
                    await aiofiles.os.remove(tmp_path)
            except Exception as e:
                logger.debug("Failed to remove temp file {}: {}".format(tmp_path, e))
            raise
            
        return str(file_path)

    def prune(self, location: str, branch: str, keep: int = 10, max_keep: Optional[int] = None) -> int:
        branch_dir = _validate_branch(location, branch)
        if not branch_dir.exists():
            return 0
        
        if keep < 0:
            raise ValueError("keep parameter cannot be negative")
        if max_keep is not None and max_keep < 0:
            raise ValueError("max_keep parameter cannot be negative")
            
        effective_keep = min(max_keep, keep) if max_keep is not None else keep
        
        pattern = os.path.join(str(branch_dir), "*.json")
        files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith('.')]
        files = sorted(files, key=os.path.getmtime)
        
        if effective_keep >= len(files):
            return 0
            
        files_to_delete = files[:-effective_keep] if effective_keep > 0 else files
        
        deleted_count = 0
        for file in files_to_delete:
            try:
                os.remove(file)
                deleted_count += 1
            except OSError:
                pass
        return deleted_count
