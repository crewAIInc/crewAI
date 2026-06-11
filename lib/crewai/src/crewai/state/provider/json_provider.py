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
    base_dir = Path(location) / branch
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
        overwriting each other's complete checkpoints. Concurrent-writer protection would 
        require additional coordination such as an advisory file lock.
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
                await asyncio.get_event_loop().run_in_executor(None, os.fsync, f.fileno())
            
            await asyncio.get_event_loop().run_in_executor(None, os.replace, tmp_path, file_path)
            
            def sync_dir():
                dir_fd = os.open(str(file_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            
            await asyncio.get_event_loop().run_in_executor(None, sync_dir)
            
        except Exception:
            try:
                if await aiofiles.os.path.exists(tmp_path):
                    await aiofiles.os.remove(tmp_path)
            except Exception as e:
                logger.debug("Failed to remove temp file {}: {}".format(tmp_path, e))
            raise
            
        return str(file_path)

    def prune(self, location: str, branch: str, keep: int = 10) -> int:
        branch_dir = Path(location) / branch
        if not branch_dir.exists():
            return 0
        
        pattern = os.path.join(str(branch_dir), "*.json")
        files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith('.')]
        files = sorted(files, key=os.path.getmtime)
        
        if len(files) <= keep:
            return 0
            
        deleted_count = 0
        for file in files[:-keep]:
            try:
                os.remove(file)
                deleted_count += 1
            except OSError:
                pass
        return deleted_count
