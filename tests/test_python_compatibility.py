"""Tests for Python version compatibility."""
import sys

import pytest
from packaging import version


def validate_python_version():
    """Validate that the current Python version is supported."""
    min_version = (3, 10)
    max_version = (3, 14)
    current = sys.version_info[:2]
    
    if not (min_version <= current < max_version):
        raise RuntimeError(
            f"This package requires Python {min_version[0]}.{min_version[1]} to "
            f"{max_version[0]}.{max_version[1]-1}. You have Python {current[0]}.{current[1]}"
        )


def test_python_version_compatibility():
    """Test that the package supports the current Python version."""
    assert isinstance(sys.version_info, tuple), "Version Information must be a tuple"
    
    current_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    assert current_version >= version.parse("3.10"), "Python version too old"
    assert current_version < version.parse("3.14"), "Python version too new"
    
    # This test will fail if the package doesn't support the current Python version
    import crewai
    
    # Print the Python version for debugging
    print(f"Python version: {sys.version}")
    
    # Print the crewai version for debugging
    print(f"CrewAI version: {crewai.__version__}")
    
    # Validate Python version
    validate_python_version()
