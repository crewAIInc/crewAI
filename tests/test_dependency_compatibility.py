"""
Dependency Compatibility Tests

Tests to verify compatibility between litellm and other packages that depend on httpx.

Known working versions:
- httpx >= 0.28.1
- litellm >= 1.65.1
- exa-py (optional)
- google-genai (optional)
"""
import importlib.util
import sys

import pytest


def _check_optional_dependency(package_name):
    """Centralized handling of optional dependency checks"""
    if importlib.util.find_spec(package_name) is None:
        pytest.skip(f"{package_name} not installed. Skipping compatibility test.")
    return True


@pytest.fixture(autouse=True)
def log_versions():
    """Log all relevant package versions before tests"""
    import httpx
    import litellm
    
    versions = {
        "httpx": httpx.__version__,
    }
    print("\nRunning tests with versions:", versions)
    return versions


class TestDependencyCompatibility:
    """Test suite for checking package dependency compatibility"""
    
    def test_httpx_litellm_compatibility(self):
        """Test that litellm is compatible with the latest httpx"""
        import httpx
        import litellm
        
        assert hasattr(httpx, "__version__")
        
        print(f"Using httpx version: {httpx.__version__}")
        print("Successfully imported litellm")
    
    def test_exa_py_compatibility(self):
        """Test that exa-py can be imported alongside litellm"""
        _check_optional_dependency("exa")
        
        import exa
        import httpx
        import litellm
        
        assert hasattr(exa, "__version__")
        assert hasattr(httpx, "__version__")
        
        print(f"Using exa-py version: {exa.__version__}")
        print("Successfully imported litellm")
        print(f"Using httpx version: {httpx.__version__}")
    
    def test_google_genai_compatibility(self):
        """Test that google-genai can be imported alongside litellm"""
        _check_optional_dependency("google.generativeai")
        
        import httpx
        import litellm
        from google import generativeai
        
        assert hasattr(generativeai, "version")
        assert hasattr(httpx, "__version__")
        
        print(f"Using google-genai version: {generativeai.version}")
        print("Successfully imported litellm")
        print(f"Using httpx version: {httpx.__version__}")
