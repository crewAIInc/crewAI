from typing import Any, List, Optional

import requests
from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.google_embedding_function import (
    GoogleVertexEmbeddingFunction,
)
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


class FixedGoogleVertexEmbeddingFunction(GoogleVertexEmbeddingFunction):
    """
    A wrapper around ChromaDB's GoogleVertexEmbeddingFunction that fixes the URL typo
    where 'publishers/goole' is incorrectly used instead of 'publishers/google'.
    
    Issue reference: https://github.com/crewaiinc/crewai/issues/2690
    """
    
    def __init__(self, 
                model_name: str = "textembedding-gecko", 
                api_key: Optional[str] = None,
                **kwargs: Any):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        
        self._original_post = requests.post
        requests.post = self._patched_post
    
    def __del__(self):
        if hasattr(self, '_original_post'):
            requests.post = self._original_post
    
    def _patched_post(self, url, *args, **kwargs):
        if 'publishers/goole' in url:
            url = url.replace('publishers/goole', 'publishers/google')
        
        return self._original_post(url, *args, **kwargs)
    
    def __call__(self, input: Documents) -> Embeddings:
        return super().__call__(input)
