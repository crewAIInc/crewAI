"""Test the embeddings factory functionality, particularly ONNX provider."""

import pytest


def test_onnx_embedding_function_creation():
    """Test that ONNX embedding function can be created."""
    from crewai.rag.embeddings.factory import get_embedding_function
    
    embedding_func = get_embedding_function({"provider": "onnx"})
    assert embedding_func is not None


def test_onnx_embedding_function_basic_functionality():
    """Test that ONNX embedding function can process text."""
    import numpy as np
    from crewai.rag.embeddings.factory import get_embedding_function
    
    embedding_func = get_embedding_function({"provider": "onnx"})
    
    result = embedding_func(["test text"])
    assert result is not None
    assert len(result) > 0
    assert isinstance(result[0], np.ndarray)
    assert len(result[0]) > 0


def test_get_embedding_function_onnx_provider_in_list():
    """Test that onnx provider is available in the factory."""
    from crewai.rag.embeddings.factory import get_embedding_function
    
    try:
        embedding_func = get_embedding_function({"provider": "onnx"})
        assert embedding_func is not None
    except ValueError as e:
        if "Unsupported provider" in str(e):
            pytest.fail("ONNX provider should be supported")
        else:
            raise
