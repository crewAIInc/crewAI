"""Tests for Python version compatibility."""
import sys
import pytest


def test_python_version_compatibility():
    """Test that the package supports the current Python version."""
    # This test will fail if the package doesn't support the current Python version
    import crewai
    
    # Print the Python version for debugging
    print(f"Python version: {sys.version}")
    
    # Print the crewai version for debugging
    print(f"CrewAI version: {crewai.__version__}")
    
    # If we got here, the import worked, which means the package supports this Python version
    assert True
