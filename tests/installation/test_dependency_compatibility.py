"""
Test module for verifying dependency compatibility with different package managers.
These tests ensure that critical dependencies can be installed without conflicts.
"""

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile

import pytest


@contextlib.contextmanager
def temporary_package_environment():
    """Create an isolated environment for package testing.
    
    This context manager creates a temporary directory where package installations
    can be tested in isolation, then cleans up afterward.
    """
    temp_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pypika_installation():
    """Test that pypika can be installed without packaging.licenses errors.
    
    This test verifies that pypika 0.48.9 (a dependency of chromadb, which is a
    dependency of CrewAI) can be installed without errors related to the
    packaging.licenses module when using the UV package manager.
    """
    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        pytest.skip("UV package manager not available, skipping test")
        
    # Use isolated environment for testing
    with temporary_package_environment():
        # Install pypika using uv
        result = subprocess.run(
            ["uv", "pip", "install", "pypika==0.48.9", "--no-deps"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to install pypika: {result.stderr}\nCommand output: {result.stdout}"


def test_chromadb_installation():
    """Test that chromadb can be installed without packaging.licenses errors.
    
    This test verifies that chromadb (a direct dependency of CrewAI) can be
    installed without errors related to the packaging.licenses module when
    using the UV package manager.
    """
    # Skip this test if running in CI/CD to avoid long test times
    if "CI" in os.environ:
        pytest.skip("Skipping in CI environment to reduce test time")
    
    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        pytest.skip("UV package manager not available, skipping test")
        
    # Use isolated environment for testing
    with temporary_package_environment():
        # Install chromadb using uv
        result = subprocess.run(
            ["uv", "pip", "install", "chromadb>=0.5.23", "--no-deps"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to install chromadb: {result.stderr}\nCommand output: {result.stdout}"
