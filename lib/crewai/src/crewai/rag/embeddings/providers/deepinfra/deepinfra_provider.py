"""DeepInfra embeddings provider."""

import logging
import os
from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
import httpx
from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Self

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_TAG = "embed"
CATALOG_PARAMS = {"filter": "with_meta", "sort_by": "crewai"}
CATALOG_TIMEOUT = 3.0

# Successful catalog picks per base URL. A process keeps the model it started
# with, so every store it opens shares one embedding space, and a new process
# picks up catalog changes. Failures are never cached.
_catalog_picks: dict[str, str] = {}


def effective_api_base(api_base: str | None = None) -> str:
    """Return the base URL the provider would use for ``api_base``.

    Mirrors the provider's precedence: an explicit value, then the
    ``DEEPINFRA_API_BASE`` environment variable, then the public default.
    """
    return api_base or os.environ.get("DEEPINFRA_API_BASE") or DEFAULT_API_BASE


def _embedding_model_id(entry: Any) -> str | None:
    """Return the id of a catalog entry tagged as an embedding model, else None."""
    if not isinstance(entry, dict):
        return None
    metadata = entry.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    if isinstance(tags, list) and EMBEDDING_TAG in tags and entry.get("id"):
        return str(entry["id"])
    return None


def resolve_default_model(api_base: str = DEFAULT_API_BASE) -> str:
    """Return the embedding model DeepInfra currently recommends.

    Reads the OpenAI-compatible model list sorted for crewAI and returns the
    first model tagged ``embed``, so the default follows the catalog without a
    crewAI release. Falls back to ``DEFAULT_MODEL`` when the catalog cannot be
    read or lists no embedding model; failures are not cached, so the next
    provider retries. A successful pick is kept for the life of the process,
    per base URL, because stores opened by one process must share an embedding
    space. Persisted stores that must stay stable across runs should pin
    ``model_name`` rather than rely on this default.

    Args:
        api_base: Base URL of the DeepInfra OpenAI-compatible API.

    Returns:
        The model id to embed with.
    """
    cached = _catalog_picks.get(api_base)
    if cached is not None:
        return cached
    url = f"{api_base.rstrip('/')}/models"
    try:
        response = httpx.get(url, params=CATALOG_PARAMS, timeout=CATALOG_TIMEOUT)
        response.raise_for_status()
        model_id = next(
            filter(None, map(_embedding_model_id, response.json()["data"])), None
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        logger.warning(
            "Could not read the DeepInfra model catalog at %s (%s); using %s",
            url,
            e,
            DEFAULT_MODEL,
        )
        return DEFAULT_MODEL
    if model_id is None:
        logger.warning(
            "No model tagged %r in the DeepInfra catalog at %s; using %s",
            EMBEDDING_TAG,
            url,
            DEFAULT_MODEL,
        )
        return DEFAULT_MODEL
    _catalog_picks[api_base] = model_id
    return model_id


class DeepInfraProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """DeepInfra embeddings provider.

    DeepInfra serves open-weight embedding models through an OpenAI-compatible
    endpoint, so this provider reuses the OpenAI embedding function with the
    DeepInfra base URL. When no model is configured, the embedding model
    DeepInfra currently recommends is read from its catalog; pin ``model_name``
    for stores that must stay stable across runs.
    """

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_alias(cls, data: Any) -> Any:
        """Map the ``model`` alias onto ``model_name`` and drop it.

        Extra fields are forwarded to the embedding function, which has no
        ``model`` parameter, so the alias must not survive validation.
        """
        if isinstance(data, dict) and "model" in data:
            data = data.copy()
            model = data.pop("model")
            data.setdefault("model_name", model)
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_dimensions(cls, data: Any) -> Any:
        """Refuse ``dimensions`` rather than accept a value that never reaches the API.

        chromadb's ``OpenAIEmbeddingFunction`` only sends ``dimensions`` for OpenAI
        ``text-embedding-3`` models, so for a DeepInfra model the vectors would
        silently keep the model's full size.
        """
        if isinstance(data, dict) and data.get("dimensions") is not None:
            raise ValueError(
                "dimensions is not supported by the deepinfra embedding provider: "
                "chromadb's OpenAIEmbeddingFunction only forwards it for OpenAI "
                "text-embedding-3 models, so the vectors would keep the model's "
                "full size. Choose a model with the size you need instead."
            )
        return data

    @model_validator(mode="after")
    def _default_model_from_catalog(self) -> Self:
        """Fill in the catalog's recommended model when none was configured."""
        if "model_name" not in self.model_fields_set:
            self.model_name = resolve_default_model(self.api_base)
        return self

    embedding_callable: type[OpenAIEmbeddingFunction] = Field(
        default=OpenAIEmbeddingFunction,
        description="OpenAI-compatible embedding function class",
    )
    api_key: str = Field(
        description="DeepInfra API key",
        validation_alias=AliasChoices(
            "DEEPINFRA_API_KEY",
        ),
    )
    model_name: str = Field(
        default=DEFAULT_MODEL,
        description=(
            "Model name to use for embeddings; when omitted, the embedding model "
            "DeepInfra currently recommends is read from its catalog"
        ),
        validation_alias=AliasChoices(
            "model_name",
        ),
    )
    api_base: str = Field(
        default=DEFAULT_API_BASE,
        description="Base URL for DeepInfra API requests",
        validation_alias=AliasChoices(
            "DEEPINFRA_API_BASE",
            "api_base",
        ),
    )
    default_headers: dict[str, Any] | None = Field(
        default=None, description="Default headers for API requests"
    )
