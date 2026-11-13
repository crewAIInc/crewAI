"""DeepInfra embeddings provider."""

import os
from typing import Any

from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
import httpx
from pydantic import AliasChoices, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from typing_extensions import Self

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


DEFAULT_API_BASE = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
AUTO_MODEL = "auto"
EMBEDDING_TAG = "embed"
CATALOG_PARAMS = {"filter": "with_meta", "sort_by": "crewai"}
CATALOG_TIMEOUT = 3.0
_PIN_HINT = (
    f"Set model_name to an embedding model id instead, such as {DEFAULT_MODEL!r}."
)

# Catalog picks per normalized base URL. A process keeps the model it started
# with, so every store it opens shares one embedding space. Failures raise and
# are not cached, so a later provider can succeed once the catalog is back.
_catalog_picks: dict[str, str] = {}


def _embedding_model_id(entry: Any) -> str | None:
    """Return the id of a catalog entry tagged as an embedding model, else None."""
    if not isinstance(entry, dict):
        return None
    metadata = entry.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    if isinstance(tags, list) and EMBEDDING_TAG in tags and entry.get("id"):
        return str(entry["id"])
    return None


def resolve_catalog_model(api_base: str = DEFAULT_API_BASE) -> str:
    """Return the embedding model DeepInfra currently recommends.

    Reads the OpenAI-compatible model list sorted for crewAI and returns the
    first model tagged ``embed``. This is what ``model_name="auto"`` resolves
    to. A pick is kept for the life of the process, per base URL, because
    stores opened by one process must share an embedding space.

    Args:
        api_base: Base URL of the DeepInfra OpenAI-compatible API.

    Returns:
        The model id to embed with.

    Raises:
        ValueError: The catalog could not be read or lists no embedding model.
    """
    base = api_base.rstrip("/")
    cached = _catalog_picks.get(base)
    if cached is not None:
        return cached
    url = f"{base}/models"
    try:
        response = httpx.get(url, params=CATALOG_PARAMS, timeout=CATALOG_TIMEOUT)
        response.raise_for_status()
        model_id = next(
            filter(None, map(_embedding_model_id, response.json()["data"])), None
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"model_name={AUTO_MODEL!r} needs the DeepInfra model catalog, which "
            f"could not be read at {url} ({e}). {_PIN_HINT}"
        ) from e
    if model_id is None:
        raise ValueError(
            f"model_name={AUTO_MODEL!r} found no model tagged {EMBEDDING_TAG!r} in "
            f"the DeepInfra model catalog at {url}. {_PIN_HINT}"
        )
    _catalog_picks[base] = model_id
    return model_id


# The environment variables the provider reads, by field.
_ENV_VARS = {"api_key": "DEEPINFRA_API_KEY", "api_base": "DEEPINFRA_API_BASE"}


class _DeepInfraEnvSource(PydanticBaseSettingsSource):
    """Read the provider's settings from DeepInfra's own variables only.

    The default settings source matches every field name and alias against the
    environment, ignoring case. Variables that belong to other tools would then
    configure the embedder: ``MODEL_NAME``, which crewAI reads for the chat
    model, would become the embedding model, ``API_BASE`` the endpoint and
    ``API_KEY`` the credential sent to DeepInfra.
    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        env_name = _ENV_VARS.get(field_name)
        return (os.environ.get(env_name) if env_name else None), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {
            field_name: os.environ[env_name]
            for field_name, env_name in _ENV_VARS.items()
            if env_name in os.environ
        }


class DeepInfraProvider(BaseEmbeddingsProvider[OpenAIEmbeddingFunction]):
    """DeepInfra embeddings provider.

    DeepInfra serves open-weight embedding models through an OpenAI-compatible
    endpoint, so this provider reuses the OpenAI embedding function with the
    DeepInfra base URL. With ``model_name="auto"`` the embedding model DeepInfra
    currently recommends is read from its catalog when the provider is built;
    that pick can change between runs, so persisted stores should use a fixed id.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Take settings from the embedder config, then DeepInfra's variables."""
        return init_settings, _DeepInfraEnvSource(settings_cls)

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
    def _resolve_auto_model(self) -> Self:
        """Replace ``model_name="auto"`` with the catalog's recommended model."""
        if self.model_name == AUTO_MODEL:
            self.model_name = resolve_catalog_model(self.api_base)
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
            "Model name to use for embeddings; 'auto' reads the embedding model "
            "DeepInfra currently recommends from its catalog"
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
