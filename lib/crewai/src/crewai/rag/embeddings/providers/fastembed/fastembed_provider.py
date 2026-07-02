"""FastEmbed embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.fastembed.embedding_callable import (
    FastEmbedEmbeddingFunction,
)


class FastEmbedProvider(BaseEmbeddingsProvider[FastEmbedEmbeddingFunction]):
    """FastEmbed embeddings provider."""

    embedding_callable: type[FastEmbedEmbeddingFunction] = Field(
        default=FastEmbedEmbeddingFunction,
        description="FastEmbed embedding function class",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_MODEL_NAME",
            "FASTEMBED_MODEL_NAME",
            "model",
        ),
    )
    cache_dir: str | None = Field(
        default=None,
        description="Directory to cache downloaded FastEmbed models",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_CACHE_DIR", "FASTEMBED_CACHE_DIR"
        ),
    )
    threads: int | None = Field(
        default=None,
        description="Number of threads to use for inference",
        validation_alias=AliasChoices("EMBEDDINGS_FASTEMBED_THREADS", "FASTEMBED_THREADS"),
    )
    providers: list[str] | None = Field(
        default=None,
        description="ONNX Runtime execution providers",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_PROVIDERS", "FASTEMBED_PROVIDERS"
        ),
    )
    cuda: bool = Field(
        default=False,
        description="Whether to use CUDA execution",
        validation_alias=AliasChoices("EMBEDDINGS_FASTEMBED_CUDA", "FASTEMBED_CUDA"),
    )
    device_ids: list[int] | None = Field(
        default=None,
        description="CUDA device IDs to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_DEVICE_IDS", "FASTEMBED_DEVICE_IDS"
        ),
    )
    lazy_load: bool = Field(
        default=False,
        description="Whether to defer model loading until first embedding call",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_LAZY_LOAD", "FASTEMBED_LAZY_LOAD"
        ),
    )
    batch_size: int = Field(
        default=256,
        description="Batch size to use when embedding documents",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_BATCH_SIZE", "FASTEMBED_BATCH_SIZE"
        ),
    )
    parallel: int | None = Field(
        default=None,
        description="Number of parallel workers to use when embedding documents",
        validation_alias=AliasChoices(
            "EMBEDDINGS_FASTEMBED_PARALLEL", "FASTEMBED_PARALLEL"
        ),
    )
