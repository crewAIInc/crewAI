import importlib.util
import sys

import pytest


def test_httpx_litellm_compatibility():
    """Test that litellm is compatible with the latest httpx"""
    import httpx
    import litellm
    
    assert hasattr(httpx, "__version__")
    
    print(f"Using httpx version: {httpx.__version__}")
    print("Successfully imported litellm")


def test_exa_py_compatibility():
    """Test that exa-py can be imported alongside litellm"""
    if importlib.util.find_spec("exa") is None:
        pytest.skip("exa-py not installed")
    
    import exa
    import httpx
    import litellm
    
    assert hasattr(exa, "__version__")
    assert hasattr(httpx, "__version__")
    
    print(f"Using exa-py version: {exa.__version__}")
    print("Successfully imported litellm")
    print(f"Using httpx version: {httpx.__version__}")


def test_google_genai_compatibility():
    """Test that google-genai can be imported alongside litellm"""
    if importlib.util.find_spec("google.generativeai") is None:
        pytest.skip("google-genai not installed")
    
    import httpx
    import litellm
    from google import generativeai
    
    assert hasattr(generativeai, "version")
    assert hasattr(httpx, "__version__")
    
    print(f"Using google-genai version: {generativeai.version}")
    print("Successfully imported litellm")
    print(f"Using httpx version: {httpx.__version__}")
