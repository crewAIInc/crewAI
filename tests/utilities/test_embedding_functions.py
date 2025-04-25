import pytest
import requests
from unittest.mock import patch, MagicMock

from crewai.utilities.embedding_functions import FixedGoogleVertexEmbeddingFunction


class TestFixedGoogleVertexEmbeddingFunction:
    @pytest.fixture
    def embedding_function(self):
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"predictions": [[0.1, 0.2, 0.3]]}
            mock_post.return_value = mock_response
            
            function = FixedGoogleVertexEmbeddingFunction(
                model_name="test-model",
                api_key="test-key"
            )
            
            yield function, mock_post
            
            if hasattr(function, '_original_post'):
                requests.post = function._original_post
            
    def test_url_correction(self, embedding_function):
        function, mock_post = embedding_function
        
        typo_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/goole/models/test-model:predict"
        
        expected_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/test-model:predict"
        
        with patch.object(function, '_original_post') as mock_original_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"predictions": [[0.1, 0.2, 0.3]]}
            mock_original_post.return_value = mock_response
            
            response = function._patched_post(typo_url, json={})
            
            mock_original_post.assert_called_once()
            call_args = mock_original_post.call_args
            assert call_args[0][0] == expected_url
            
    def test_embedding_call(self, embedding_function):
        function, mock_post = embedding_function
        
        embeddings = function(["test text"])
        
        mock_post.assert_called_once()
        
        assert isinstance(embeddings, list)
