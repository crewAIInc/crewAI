"""OpenCLIP embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class OpenCLIPProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """OpenCLIP embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "open_clip_embedding_function", "OpenCLIPEmbeddingFunction"
        ),
        description="OpenCLIP embedding function class",
    )
    model_name: str = Field(
        default="ViT-B-32",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENCLIP_MODEL_NAME",
            "OPENCLIP_MODEL_NAME",
            "model",
        ),
    )
    checkpoint: str = Field(
        default="laion2b_s34b_b79k",
        description="Model checkpoint",
        validation_alias=AliasChoices(
            "EMBEDDINGS_OPENCLIP_CHECKPOINT", "OPENCLIP_CHECKPOINT"
        ),
    )
    device: str | None = Field(
        default="cpu",
        description="Device to run model on (e.g., cpu, cuda, mps, xpu)",
        validation_alias=AliasChoices("EMBEDDINGS_OPENCLIP_DEVICE", "OPENCLIP_DEVICE"),
    )
