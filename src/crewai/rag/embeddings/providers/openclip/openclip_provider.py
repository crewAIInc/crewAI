"""OpenCLIP embeddings provider."""

from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class OpenCLIPProvider(BaseEmbeddingsProvider[OpenCLIPEmbeddingFunction]):
    """OpenCLIP embeddings provider."""

    embedding_callable: type[OpenCLIPEmbeddingFunction] = Field(
        default=OpenCLIPEmbeddingFunction,
        description="OpenCLIP embedding function class",
    )
    model_name: str = Field(
        default="ViT-B-32",
        description="Model name to use",
        validation_alias="EMBEDDINGS_OPENCLIP_MODEL_NAME",
    )
    checkpoint: str = Field(
        default="laion2b_s34b_b79k",
        description="Model checkpoint",
        validation_alias="EMBEDDINGS_OPENCLIP_CHECKPOINT",
    )
    device: str | None = Field(
        default="cpu",
        description="Device to run model on",
        validation_alias="EMBEDDINGS_OPENCLIP_DEVICE",
    )
