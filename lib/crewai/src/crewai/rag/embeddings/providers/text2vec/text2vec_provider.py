"""Text2Vec embeddings provider."""

from chromadb.utils.embedding_functions.text2vec_embedding_function import (
    Text2VecEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class Text2VecProvider(BaseEmbeddingsProvider[Text2VecEmbeddingFunction]):
    """Text2Vec embeddings provider."""

    embedding_callable: type[Text2VecEmbeddingFunction] = Field(
        default=Text2VecEmbeddingFunction,
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
