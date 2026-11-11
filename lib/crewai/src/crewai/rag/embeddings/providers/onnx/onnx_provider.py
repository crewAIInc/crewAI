"""ONNX embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class ONNXProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """ONNX embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "onnx_mini_lm_l6_v2", "ONNXMiniLM_L6_V2"
        ),
        description="ONNX MiniLM embedding function class",
    )
    preferred_providers: list[str] | None = Field(
        default=None,
        description="Preferred ONNX execution providers",
        validation_alias=AliasChoices(
            "EMBEDDINGS_ONNX_PREFERRED_PROVIDERS", "ONNX_PREFERRED_PROVIDERS"
        ),
    )
