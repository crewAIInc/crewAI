# conftest.py
import os
import pytest
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_result = load_dotenv(override=True)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with a temporary directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the directory with proper permissions
        storage_dir = Path(temp_dir) / "crewai_test_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable to point to the test storage directory
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        
        yield
        
        # Cleanup is handled automatically when tempfile context exits
