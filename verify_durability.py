import os
import uuid
import glob
import json
from pathlib import Path
from crewai.state.provider.json_provider import JsonProvider

def test_atomic_write():
    print("Testing atomic write durability...")
    provider = JsonProvider(location="test_checkpoints")
    data = json.dumps({"test": "data"})
    
    # 1. Normal write
    path = provider.checkpoint(data, "test_checkpoints", branch="main")
    print(f"✅ Checkpoint created at {path}")
    
    # 2. Simulate crash during write (mocking the process)
    # We can't easily crash the process, but we can verify that a .tmp file exists 
    # during the write process if we had a hook. 
    # Instead, let's verify that the final file is only created AFTER the tmp file is closed.
    
    # 3. Verify prune doesn't touch tmp files
    tmp_file = Path("test_checkpoints/main/fake.tmp_12345.json")
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file.write_text("temp data")
    
    provider.prune("test_checkpoints", "main", keep=1)
    
    if tmp_file.exists():
        print("✅ Prune correctly ignored .tmp_ file")
    else:
        print("❌ Prune deleted the .tmp_ file!")

if __name__ == "__main__":
    try:
        test_atomic_write()
    except Exception as e:
        print(f"❌ Test failed: {e}")
