"""Factory for creating file uploaders."""

from __future__ import annotations

import logging
from typing import Literal, TypedDict, overload, reveal_type

from typing_extensions import Unpack

from crewai.files.uploaders.base import FileUploader


logger = logging.getLogger(__name__)


ProviderType = Literal[
    "gemini", "google", "anthropic", "claude", "openai", "gpt", "bedrock", "aws"
]
UnknownProvider = str


class AllOptions(TypedDict):
    """Kwargs for uploader factory."""

    api_key: str | None
    chunk_size: int
    bucket_name: str
    bucket_owner: str
    prefix: str
    region: str


@overload
def get_uploader(provider: UnknownProvider, /) -> None:
    """Get file uploader for unknown provider."""


@overload
def get_uploader(
    provider: Literal["gemini", "google"],
    *,
    api_key: str | None = ...,
) -> FileUploader:
    """Get Gemini file uploader."""


@overload
def get_uploader(
    provider: Literal["anthropic", "claude"],
    *,
    api_key: str | None = ...,
) -> FileUploader:
    """Get Anthropic file uploader."""


@overload
def get_uploader(
    provider: Literal["openai", "gpt"],
    *,
    api_key: str | None = ...,
    chunk_size: int = ...,
) -> FileUploader | None:
    """Get OpenAI file uploader."""


@overload
def get_uploader(
    provider: Literal["bedrock", "aws"],
    *,
    bucket_name: str | None = ...,
    bucket_owner: str | None = ...,
    prefix: str = ...,
    region: str | None = ...,
) -> FileUploader | None:
    """Get Bedrock file uploader."""


@overload
def get_uploader(
    provider: ProviderType | UnknownProvider, **kwargs: Unpack[AllOptions]
) -> FileUploader | None:
    """Get any file uploader."""


def get_uploader(provider, **kwargs):  # type: ignore[no-untyped-def]
    """Get a file uploader for a specific provider.

    Args:
        provider: Provider name (e.g., "gemini", "anthropic").
        **kwargs: Additional arguments passed to the uploader constructor.

    Returns:
        FileUploader instance for the provider, or None if not supported.
    """
    provider_lower = provider.lower()

    if "gemini" in provider_lower or "google" in provider_lower:
        try:
            from crewai.files.uploaders.gemini import GeminiFileUploader

            return GeminiFileUploader(api_key=kwargs.get("api_key"))
        except ImportError:
            logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            return None

    if "anthropic" in provider_lower or "claude" in provider_lower:
        try:
            from crewai.files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(api_key=kwargs.get("api_key"))
        except ImportError:
            logger.warning(
                "anthropic not installed. Install with: pip install anthropic"
            )
            return None

    if "openai" in provider_lower or "gpt" in provider_lower:
        try:
            from crewai.files.uploaders.openai import OpenAIFileUploader

            return OpenAIFileUploader(
                api_key=kwargs.get("api_key"),
                chunk_size=kwargs.get("chunk_size", 67_108_864),
            )
        except ImportError:
            logger.warning("openai not installed. Install with: pip install openai")
            return None

    if "bedrock" in provider_lower or "aws" in provider_lower:
        import os

        if (
            not os.environ.get("CREWAI_BEDROCK_S3_BUCKET")
            and "bucket_name" not in kwargs
        ):
            logger.debug(
                "Bedrock S3 uploader not configured. "
                "Set CREWAI_BEDROCK_S3_BUCKET environment variable to enable."
            )
            return None
        try:
            from crewai.files.uploaders.bedrock import BedrockFileUploader

            return BedrockFileUploader(
                bucket_name=kwargs.get("bucket_name"),
                bucket_owner=kwargs.get("bucket_owner"),
                prefix=kwargs.get("prefix", "crewai-files"),
                region=kwargs.get("region"),
            )
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            return None

    logger.debug(f"No file uploader available for provider: {provider}")
    return None


t = get_uploader("openai")
reveal_type(t)
