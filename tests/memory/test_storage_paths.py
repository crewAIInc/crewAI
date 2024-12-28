import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch

from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage
from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities.paths import get_default_storage_path

class MockRAGStorage(BaseRAGStorage):
    """Mock implementation of BaseRAGStorage for testing."""
    def _sanitize_role(self, role: str) -> str:
        return role.lower()
    
    def save(self, value, metadata):
        pass
    
    def search(self, query, limit=3, filter=None, score_threshold=0.35):
        return []
    
    def reset(self):
        pass
    
    def _generate_embedding(self, text, metadata=None):
        return []
    
    def _initialize_app(self):
        pass

def test_default_storage_paths():
    """Test that default storage paths are created correctly."""
    ltm_path = get_default_storage_path('ltm')
    kickoff_path = get_default_storage_path('kickoff')
    rag_path = get_default_storage_path('rag')
    
    assert str(ltm_path).endswith('latest_long_term_memories.db')
    assert str(kickoff_path).endswith('latest_kickoff_task_outputs.db')
    assert isinstance(rag_path, Path)

def test_custom_storage_paths():
    """Test that custom storage paths are respected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_path = Path(temp_dir) / 'custom.db'
        
        ltm = LTMSQLiteStorage(storage_path=custom_path)
        assert ltm.storage_path == custom_path
        
        kickoff = KickoffTaskOutputsSQLiteStorage(storage_path=custom_path)
        assert kickoff.storage_path == custom_path
        
        rag = MockRAGStorage('test', storage_path=custom_path)
        assert rag.storage_path == custom_path

def test_directory_creation():
    """Test that storage directories are created automatically."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / 'test_storage'
        storage_path = test_dir / 'test.db'
        
        assert not test_dir.exists()
        LTMSQLiteStorage(storage_path=storage_path)
        assert test_dir.exists()

def test_permission_error():
    """Test that permission errors are handled correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / 'readonly'
        test_dir.mkdir()
        os.chmod(test_dir, 0o444)  # Read-only
        
        storage_path = test_dir / 'test.db'
        with pytest.raises((PermissionError, OSError)) as exc_info:
            LTMSQLiteStorage(storage_path=storage_path)
        # Verify that the error message mentions permission
        assert "permission" in str(exc_info.value).lower()

def test_invalid_path():
    """Test that invalid paths raise appropriate errors."""
    with pytest.raises(OSError):
        # Try to create storage in a non-existent root directory
        LTMSQLiteStorage(storage_path=Path('/nonexistent/dir/test.db'))
