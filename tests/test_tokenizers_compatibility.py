"""Test suite to verify compatibility between tokenizers and transformers packages.
Ensures the installed tokenizers version meets requirements and can work effectively with transformers.
"""

import pytest


def test_tokenizers_transformers_compatibility():
    """Test that the installed tokenizers version is compatible with transformers."""
    try:
        import tokenizers
        import transformers
    except ImportError:
        pytest.skip("tokenizers or transformers not installed")
    
    tokenizers_version = tokenizers.__version__
    transformers_version = transformers.__version__
    
    tokenizers_major, tokenizers_minor, _ = map(int, tokenizers_version.split('.'))
    
    assert tokenizers_major == 0, f"Expected tokenizers major version 0, got {tokenizers_major}"
    assert tokenizers_minor >= 21, f"Expected tokenizers minor version >=21, got {tokenizers_minor}"
    assert tokenizers_minor < 22, f"Expected tokenizers minor version <22, got {tokenizers_minor}"
    
    print(f"Tokenizers version: {tokenizers_version}")
    print(f"Transformers version: {transformers_version}")
