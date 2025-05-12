"""Test to verify that the linting configuration is working correctly."""
import os
from pathlib import Path


def test_ruff_config_exists():
    """Test that the ruff configuration file exists."""
    repo_root = Path(__file__).parent.parent
    ruff_config_path = repo_root / ".ruff.toml"
    assert ruff_config_path.exists(), "Ruff configuration file (.ruff.toml) should exist"
    
    assert ruff_config_path.stat().st_size > 0, "Ruff configuration file should not be empty"
