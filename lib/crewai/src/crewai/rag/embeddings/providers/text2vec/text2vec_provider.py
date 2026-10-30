"""Text2Vec embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class Text2VecProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Text2Vec embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "text2vec_embedding_function", "Text2VecEmbeddingFunction"
        ),
        description="Text2Vec embedding function class",
    )
    model_name: str = Field(
        default="shibing624/text2vec-base-chinese",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_TEXT2VEC_MODEL_NAME",
            "TEXT2VEC_MODEL_NAME",
            "model",
        ),
    )
