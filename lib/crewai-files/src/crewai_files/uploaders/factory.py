"""Factory for creating file uploaders."""

from __future__ import annotations

import logging
from typing import Any as AnyType, Literal, TypeAlias, TypedDict, overload

from typing_extensions import NotRequired, Unpack

from crewai_files.uploaders.anthropic import AnthropicFileUploader
from crewai_files.uploaders.bedrock import BedrockFileUploader
from crewai_files.uploaders.gemini import GeminiFileUploader
from crewai_files.uploaders.openai import OpenAIFileUploader


logger = logging.getLogger(__name__)


FileUploaderType: TypeAlias = (
    GeminiFileUploader
    | AnthropicFileUploader
    | BedrockFileUploader
    | OpenAIFileUploader
)

GeminiProviderType = Literal["gemini", "google"]
AnthropicProviderType = Literal["anthropic", "claude"]
OpenAIProviderType = Literal["openai", "gpt", "azure"]
BedrockProviderType = Literal["bedrock", "aws"]

ProviderType: TypeAlias = (
    GeminiProviderType
    | AnthropicProviderType
    | OpenAIProviderType
    | BedrockProviderType
)


class _BaseOpts(TypedDict):
    """Kwargs for uploader factory."""

    api_key: NotRequired[str | None]
    client: NotRequired[AnyType]
    async_client: NotRequired[AnyType]


class OpenAIOpts(_BaseOpts):
    """Kwargs for openai uploader factory."""

    chunk_size: NotRequired[int]


class GeminiOpts(TypedDict):
    """Kwargs for gemini uploader factory."""

    api_key: NotRequired[str | None]
    client: NotRequired[AnyType]


class AnthropicOpts(_BaseOpts):
    """Kwargs for anthropic uploader factory."""


class BedrockOpts(TypedDict):
    """Kwargs for bedrock uploader factory."""

    bucket_name: NotRequired[str | None]
    bucket_owner: NotRequired[str | None]
    prefix: NotRequired[str]
    region: NotRequired[str | None]
    client: NotRequired[AnyType]
    async_client: NotRequired[AnyType]


class AllOptions(TypedDict):
    """Kwargs for uploader factory."""

    api_key: NotRequired[str | None]
    chunk_size: NotRequired[int]
    bucket_name: NotRequired[str | None]
    bucket_owner: NotRequired[str | None]
    prefix: NotRequired[str]
    region: NotRequired[str | None]
    client: NotRequired[AnyType]
    async_client: NotRequired[AnyType]


@overload
def get_uploader(
    provider: GeminiProviderType,
    **kwargs: Unpack[GeminiOpts],
) -> GeminiFileUploader:
    """Get Gemini file uploader."""


@overload
def get_uploader(
    provider: AnthropicProviderType,
    **kwargs: Unpack[AnthropicOpts],
) -> AnthropicFileUploader:
    """Get Anthropic file uploader."""


@overload
def get_uploader(
    provider: OpenAIProviderType,
    **kwargs: Unpack[OpenAIOpts],
) -> OpenAIFileUploader:
    """Get OpenAI file uploader."""


@overload
def get_uploader(
    provider: BedrockProviderType,
    **kwargs: Unpack[BedrockOpts],
) -> BedrockFileUploader:
    """Get Bedrock file uploader."""


@overload
def get_uploader(
    provider: ProviderType, **kwargs: Unpack[AllOptions]
) -> FileUploaderType:
    """Get any file uploader."""


def get_uploader(
    provider: ProviderType, **kwargs: Unpack[AllOptions]
) -> FileUploaderType:
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
            from crewai_files.uploaders.gemini import GeminiFileUploader

            return GeminiFileUploader(
                api_key=kwargs.get("api_key"),
                client=kwargs.get("client"),
            )
        except ImportError:
            logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            raise

    if "anthropic" in provider_lower or "claude" in provider_lower:
        try:
            from crewai_files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(
                api_key=kwargs.get("api_key"),
                client=kwargs.get("client"),
                async_client=kwargs.get("async_client"),
            )
        except ImportError:
            logger.warning(
                "anthropic not installed. Install with: pip install anthropic"
            )
            raise

    if (
        "openai" in provider_lower
        or "gpt" in provider_lower
        or "azure" in provider_lower
    ):
        try:
            from crewai_files.uploaders.openai import OpenAIFileUploader

            return OpenAIFileUploader(
                api_key=kwargs.get("api_key"),
                chunk_size=kwargs.get("chunk_size", 67_108_864),
                client=kwargs.get("client"),
                async_client=kwargs.get("async_client"),
            )
        except ImportError:
            logger.warning("openai not installed. Install with: pip install openai")
            raise

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
            raise
        try:
            from crewai_files.uploaders.bedrock import BedrockFileUploader

            return BedrockFileUploader(
                bucket_name=kwargs.get("bucket_name"),
                bucket_owner=kwargs.get("bucket_owner"),
                prefix=kwargs.get("prefix", "crewai-files"),
                region=kwargs.get("region"),
                client=kwargs.get("client"),
                async_client=kwargs.get("async_client"),
            )
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            raise

    logger.debug(f"No file uploader available for provider: {provider}")
    raise
