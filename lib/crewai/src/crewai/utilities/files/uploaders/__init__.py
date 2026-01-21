"""File uploader implementations for provider File APIs."""

from __future__ import annotations

import logging
from typing import Any

from crewai.utilities.files.uploaders.base import FileUploader, UploadResult


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
            from crewai.utilities.files.uploaders.gemini import GeminiFileUploader

            return GeminiFileUploader(**kwargs)
        except ImportError:
            logger.warning(
                "google-genai not installed. Install with: pip install google-genai"
            )
            return None

    if "anthropic" in provider_lower or "claude" in provider_lower:
        try:
            from crewai.utilities.files.uploaders.anthropic import AnthropicFileUploader

            return AnthropicFileUploader(**kwargs)
        except ImportError:
            logger.warning(
                "anthropic not installed. Install with: pip install anthropic"
            )
            return None

    if "openai" in provider_lower or "gpt" in provider_lower:
        try:
            from crewai.utilities.files.uploaders.openai import OpenAIFileUploader

            return OpenAIFileUploader(**kwargs)
        except ImportError:
            logger.warning("openai not installed. Install with: pip install openai")
            return None

    logger.debug(f"No file uploader available for provider: {provider}")
    return None
