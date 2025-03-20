import os
import shutil
import subprocess
import sys

import pytest


def test_pypika_installation():
    """Test that pypika can be installed without packaging.licenses errors."""
    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        pytest.skip("UV not available, skipping test")
        
    # Install pypika using uv
    result = subprocess.run(
        ["uv", "pip", "install", "pypika==0.48.9", "--no-deps"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to install pypika: {result.stderr}"


def test_chromadb_installation():
    """Test that chromadb can be installed without packaging.licenses errors."""
    # Skip this test if running in CI/CD to avoid long test times
    if "CI" in os.environ:
        pytest.skip("Skipping in CI environment")
    
    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        pytest.skip("UV not available, skipping test")
        
    # Install chromadb using uv
    result = subprocess.run(
        ["uv", "pip", "install", "chromadb>=0.5.23", "--no-deps"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to install chromadb: {result.stderr}"
