"""ONNX embeddings provider."""

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class ONNXProvider(BaseEmbeddingsProvider[ONNXMiniLM_L6_V2]):
    """ONNX embeddings provider."""

    embedding_callable: type[ONNXMiniLM_L6_V2] = Field(
        default=ONNXMiniLM_L6_V2, description="ONNX MiniLM embedding function class"
    )
    preferred_providers: list[str] | None = Field(
        default=None,
        description="Preferred ONNX execution providers",
        validation_alias="EMBEDDINGS_ONNX_PREFERRED_PROVIDERS",
    )
