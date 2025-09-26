"""Tests for Watson embedding function name method."""

import pytest
from unittest.mock import patch, MagicMock

from crewai.rag.embeddings.providers.ibm.embedding_callable import WatsonEmbeddingFunction


class TestWatsonEmbeddingName:
    """Test Watson embedding function name method."""

    def test_watson_embedding_function_has_name_method(self):
        """Test that WatsonEmbeddingFunction has a name method."""
        assert hasattr(WatsonEmbeddingFunction, 'name')
        assert callable(getattr(WatsonEmbeddingFunction, 'name'))

    def test_watson_embedding_function_name_returns_watson(self):
        """Test that the name method returns 'watson'."""
        assert WatsonEmbeddingFunction.name() == "watson"

    def test_watson_embedding_function_name_is_static(self):
        """Test that the name method can be called without instantiation."""
        name = WatsonEmbeddingFunction.name()
        assert name == "watson"
        assert isinstance(name, str)

    def test_watson_embedding_function_name_with_chromadb_validation(self):
        """Test that the name method works in ChromaDB validation scenario."""
        config = {
            "model_id": "test-model",
            "api_key": "test-key",
            "url": "https://test.com"
        }
        
        watson_func = WatsonEmbeddingFunction(**config)
        
        try:
            name = watson_func.name()
            assert name == "watson"
        except AttributeError as e:
            pytest.fail(f"ChromaDB validation failed with AttributeError: {e}")

    def test_watson_embedding_function_name_method_signature(self):
        """Test that the name method has the correct signature."""
        import inspect
        
        name_method = getattr(WatsonEmbeddingFunction, 'name')
        
        assert isinstance(inspect.getattr_static(WatsonEmbeddingFunction, 'name'), staticmethod)
        
        sig = inspect.signature(name_method)
        if sig.return_annotation != inspect.Signature.empty:
            assert sig.return_annotation == str

    def test_watson_embedding_function_reproduces_original_issue(self):
        """Test that reproduces the original issue scenario from #3597."""
        
        
        config = {
            "model_id": "ibm/slate-125m-english-rtrvr",
            "api_key": "test-key",
            "url": "https://us-south.ml.cloud.ibm.com",
            "project_id": "test-project"
        }
        
        watson_func = WatsonEmbeddingFunction(**config)
        
        name = watson_func.name()
        
        assert name == "watson"
        assert isinstance(name, str)
        
        class_name = WatsonEmbeddingFunction.name()
        assert class_name == "watson"
        assert class_name == name
