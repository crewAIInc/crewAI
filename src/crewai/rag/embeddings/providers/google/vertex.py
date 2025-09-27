"""Google Vertex AI embeddings provider."""

from chromadb.utils.embedding_functions.google_embedding_function import (
    GoogleVertexEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class VertexAIProvider(BaseEmbeddingsProvider[GoogleVertexEmbeddingFunction]):
    """Google Vertex AI embeddings provider."""

    embedding_callable: type[GoogleVertexEmbeddingFunction] = Field(
        default=GoogleVertexEmbeddingFunction,
        description="Vertex AI embedding function class",
    )
    model_name: str = Field(
        default="textembedding-gecko",
        description="Model name to use for embeddings",
        validation_alias="EMBEDDINGS_GOOGLE_VERTEX_MODEL_NAME",
    )
    api_key: str = Field(
        description="Google API key", validation_alias="EMBEDDINGS_GOOGLE_CLOUD_API_KEY"
    )
    project_id: str = Field(
        default="cloud-large-language-models",
        description="GCP project ID",
        validation_alias="EMBEDDINGS_GOOGLE_CLOUD_PROJECT",
    )
    region: str = Field(
        default="us-central1",
        description="GCP region",
        validation_alias="EMBEDDINGS_GOOGLE_CLOUD_REGION",
    )
