"""Type definitions for NVIDIA embeddings provider."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict

# NVIDIA embedding models verified accessible via API testing
# Last verified: 2026-01-06 (7 of 13 models in catalog are accessible)
NvidiaEmbeddingModels = Literal[
    "baai/bge-m3",  # 1024 dimensions - General purpose embedding
    "nvidia/llama-3.2-nemoretriever-300m-embed-v1",  # 2048 dimensions - Compact retriever
    "nvidia/llama-3.2-nemoretriever-300m-embed-v2",  # 2048 dimensions - Compact retriever v2
    "nvidia/llama-3.2-nv-embedqa-1b-v2",  # 2048 dimensions - QA embedding
    "nvidia/nv-embed-v1",  # 4096 dimensions - NVIDIA's flagship (recommended)
    "nvidia/nv-embedcode-7b-v1",  # 4096 dimensions - Code embedding specialist
    "nvidia/nv-embedqa-e5-v5",  # 1024 dimensions - QA embedding based on E5
]


class NvidiaProviderConfig(TypedDict, total=False):
    """Configuration for NVIDIA provider."""

    api_key: str
    model_name: str
    api_base: str
    input_type: str  # 'query' or 'passage' for asymmetric models
    truncate: str  # 'NONE', 'START', or 'END'


class NvidiaProviderSpec(TypedDict, total=False):
    """NVIDIA provider specification."""

    provider: Required[Literal["nvidia"]]
    config: NvidiaProviderConfig
