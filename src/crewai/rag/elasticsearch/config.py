"""Elasticsearch configuration model."""

from dataclasses import field
from typing import Literal, cast
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig
from crewai.rag.elasticsearch.types import (
    ElasticsearchClientParams,
    ElasticsearchEmbeddingFunctionWrapper,
)
from crewai.rag.elasticsearch.constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DIMENSION,
)


def _default_options() -> ElasticsearchClientParams:
    """Create default Elasticsearch client options.

    Returns:
        Default options with local Elasticsearch connection.
    """
    return ElasticsearchClientParams(
        hosts=[f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"],
        use_ssl=False,
        verify_certs=False,
        timeout=30,
    )


def _default_embedding_function() -> ElasticsearchEmbeddingFunctionWrapper:
    """Create default Elasticsearch embedding function.

    Returns:
        Default embedding function using sentence-transformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        
        def embed_fn(text: str) -> list[float]:
            """Embed a single text string.

            Args:
                text: Text to embed.

            Returns:
                Embedding vector as list of floats.
            """
            embedding = model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

        return cast(ElasticsearchEmbeddingFunctionWrapper, embed_fn)
    except ImportError:
        def fallback_embed_fn(text: str) -> list[float]:
            """Fallback embedding function when sentence-transformers is not available."""
            import hashlib
            import struct
            
            hash_obj = hashlib.md5(text.encode(), usedforsecurity=False)
            hash_bytes = hash_obj.digest()
            
            vector = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    value = struct.unpack('f', chunk)[0]
                    vector.append(float(value))
            
            while len(vector) < DEFAULT_VECTOR_DIMENSION:
                vector.extend(vector[:DEFAULT_VECTOR_DIMENSION - len(vector)])
            
            return vector[:DEFAULT_VECTOR_DIMENSION]
        
        return cast(ElasticsearchEmbeddingFunctionWrapper, fallback_embed_fn)


@pyd_dataclass(frozen=True)
class ElasticsearchConfig(BaseRagConfig):
    """Configuration for Elasticsearch client."""

    provider: Literal["elasticsearch"] = field(default="elasticsearch", init=False)
    options: ElasticsearchClientParams = field(default_factory=_default_options)
    vector_dimension: int = DEFAULT_VECTOR_DIMENSION
    similarity: str = "cosine"
    embedding_function: ElasticsearchEmbeddingFunctionWrapper = field(
        default_factory=_default_embedding_function
    )
