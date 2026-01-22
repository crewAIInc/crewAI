"""File uploader implementations for provider File APIs."""

from __future__ import annotations

import logging
from typing import Any

from crewai.files.uploaders.base import FileUploader, UploadResult


logger = logging.getLogger(__name__)

__all__ = [
    "FileUploader",
    "UploadResult",
    "get_uploader",
]


def get_uploader(provider: str, **kwargs: Any) -> FileUploader | None:
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

            return GeminiFileUploader(**kwargs)
        except ImportError:
            logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            return None

    if "anthropic" in provider_lower or "claude" in provider_lower:
        try:
            from crewai.files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(**kwargs)
        except ImportError:
            logger.warning(
                "anthropic not installed. Install with: pip install anthropic"
            )
            return None

    if "openai" in provider_lower or "gpt" in provider_lower:
        try:
            from crewai.files.uploaders.openai import OpenAIFileUploader

            return OpenAIFileUploader(**kwargs)
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

            return BedrockFileUploader(**kwargs)
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            return None

    logger.debug(f"No file uploader available for provider: {provider}")
    return None
