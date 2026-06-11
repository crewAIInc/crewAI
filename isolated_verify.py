import os
import uuid
import glob
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

def _build_path(location: str, branch: str, parent_id: str | None = None) -> Path:
    base_dir = Path(location) / branch
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    pid = parent_id or "root"
    return base_dir / f"{ts}_{uid}_p-{pid}.json"

class JsonProvider:
    def __init__(self, location: str = "checkpoints"):
        self.location = location

    def checkpoint(self, data: str, location: str, *, parent_id: str | None = None, branch: str = "main") -> str:
        file_path = _build_path(location, branch, parent_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex[:8]}.json")
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
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return str(file_path)

    def prune(self, location: str, branch: str, keep: int = 10) -> int:
        branch_dir = Path(location) / branch
        if not branch_dir.exists():
            return 0
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
                pass
        return deleted_count

def test_durability():
    print("Testing isolated JsonProvider durability...")
    provider = JsonProvider()
    data = '{"status": "ok"}'
    
    # Test 1: Basic write
    path = provider.checkpoint(data, "test_dur", branch="main")
    print(f"✅ Checkpoint created: {path}")
    
    # Test 2: Prune ignores .tmp_ files
    tmp_file = Path("test_dur/main/fake.tmp_123.json")
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file.write_text("temp")
    
    provider.prune("test_dur", "main", keep=0) # Delete all
    
    if tmp_file.exists():
        print("✅ Prune correctly ignored .tmp_ file")
    else:
        print("❌ Prune deleted the .tmp_ file!")

if __name__ == "__main__":
    test_durability()
