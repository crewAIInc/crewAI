"""Shared embedding helpers for the Azure CosmosDB NoSQL tools.

Wraps the OpenAI / AzureOpenAI clients so the tools can keep their
construction code small and the optional ``openai`` dependency stays lazy.
Also accepts any object satisfying the
:class:`crewai_tools.azure.cosmosdb_nosql._utils.EmbedderProtocol` so callers
can plug in their own embedder (HuggingFace, Bedrock, etc.) without taking
``openai`` as a hard dependency.
"""

from __future__ import annotations

import os
import time
from typing import Any


_INSTALL_HINT = (
    "Embedding generation requires the 'openai' package. "
    "Install it with: pip install openai"
)


def _require_openai() -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(_INSTALL_HINT) from exc
    import openai

    return openai


def build_openai_client(
    azure_openai_endpoint: str | None = None,
    openai_api_key: str | None = None,
    azure_api_version: str = "2024-02-01",
) -> Any:
    """Build either an ``AzureOpenAI`` or ``OpenAI`` client.

    Selection rules (mirrors what the three tools used independently before):

    * if ``azure_openai_endpoint`` is provided, use ``AzureOpenAI`` with the
      supplied API key (falling back to ``AZURE_OPENAI_API_KEY``).
    * else if ``AZURE_OPENAI_ENDPOINT`` is set in the environment, use
      ``AzureOpenAI`` with default settings (and the env-derived API key).
    * else use the standard ``OpenAI`` client (key from ``openai_api_key`` or
      ``OPENAI_API_KEY``).
    """
    openai = _require_openai()

    if azure_openai_endpoint:
        return openai.AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=openai_api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=azure_api_version,
        )
    if "AZURE_OPENAI_ENDPOINT" in os.environ:
        return openai.AzureOpenAI(
            api_key=openai_api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=azure_api_version,
        )
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Either an Azure OpenAI endpoint or an OpenAI API key must be "
            "provided (set 'azure_openai_endpoint', 'openai_api_key', or the "
            "AZURE_OPENAI_ENDPOINT / OPENAI_API_KEY environment variables)."
        )
    return openai.OpenAI(api_key=api_key)


def embed_texts(
    client: Any,
    texts: list[str],
    model: str,
    dimensions: int,
    *,
    max_attempts: int = 4,
    initial_backoff: float = 0.5,
) -> list[list[float]]:
    """Embed a list of texts via the given client and return the vectors.

    Retries on transient failures from the OpenAI / AzureOpenAI clients
    (``RateLimitError`` and ``APIConnectionError``) with exponential backoff.
    """
    try:
        from openai import APIConnectionError, RateLimitError

        retryable_errors: tuple[type[BaseException], ...] = (
            RateLimitError,
            APIConnectionError,
        )
    except ImportError:  # pragma: no cover - extras not installed
        retryable_errors = ()

    backoff = initial_backoff
    for attempt in range(max_attempts):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                dimensions=dimensions,
            )
            return [item.embedding for item in response.data]
        except retryable_errors:  # noqa: PERF203
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    raise RuntimeError(
        "embed_texts retry loop exited without result"
    )  # pragma: no cover


def embed_texts_via_embedder(
    embedder: Any,
    texts: list[str],
) -> list[list[float]]:
    """Call a langchain-compatible embedder (``embed_documents``)."""
    if not hasattr(embedder, "embed_documents"):
        raise TypeError(
            "Custom embedder must expose an 'embed_documents(texts)' method"
        )
    result: list[list[float]] = embedder.embed_documents(texts)
    return result
