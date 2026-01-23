"""High-level API for formatting multimodal content."""

from __future__ import annotations

import os
from typing import Any

from crewai_files.cache.upload_cache import get_upload_cache
from crewai_files.core.types import FileInput
from crewai_files.formatting.anthropic import AnthropicFormatter
from crewai_files.formatting.bedrock import BedrockFormatter
from crewai_files.formatting.gemini import GeminiFormatter
from crewai_files.formatting.openai import OpenAIFormatter, OpenAIResponsesFormatter
from crewai_files.processing.constraints import get_constraints_for_provider
from crewai_files.processing.processor import FileProcessor
from crewai_files.resolution.resolver import FileResolver, FileResolverConfig
from crewai_files.uploaders.factory import ProviderType


def _normalize_provider(provider: str | None) -> ProviderType:
    """Normalize provider string to ProviderType.

    Args:
        provider: Raw provider string.

    Returns:
        Normalized provider type.

    Raises:
        ValueError: If provider is None or empty.
    """
    if not provider:
        raise ValueError("provider is required")

    provider_lower = provider.lower()

    if "gemini" in provider_lower:
        return "gemini"
    if "google" in provider_lower:
        return "google"
    if "anthropic" in provider_lower:
        return "anthropic"
    if "claude" in provider_lower:
        return "claude"
    if "bedrock" in provider_lower:
        return "bedrock"
    if "aws" in provider_lower:
        return "aws"
    if "azure" in provider_lower:
        return "azure"
    if "gpt" in provider_lower:
        return "gpt"

    return "openai"


def _format_text_block(
    text: str, provider: str | None = None, api: str | None = None
) -> dict[str, Any]:
    """Format text as a provider-specific content block.

    Args:
        text: The text content to format.
        provider: Provider name for provider-specific formatting.
        api: API variant (e.g., "responses" for OpenAI Responses API).

    Returns:
        A content block dict in the provider's expected format.
    """
    if api == "responses":
        return OpenAIResponsesFormatter.format_text_content(text)
    if provider and ("bedrock" in provider.lower() or "aws" in provider.lower()):
        return {"text": text}
    if provider and ("gemini" in provider.lower() or "google" in provider.lower()):
        return {"text": text}
    return {"type": "text", "text": text}


def format_multimodal_content(
    files: dict[str, FileInput],
    provider: str | None = None,
    api: str | None = None,
    prefer_upload: bool | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Format text and files as provider-specific multimodal content blocks.

    This is the main high-level API for converting files to content blocks
    suitable for sending to LLM providers. It handles:
    - Text formatting according to API variant
    - File processing according to provider constraints
    - Resolution (upload vs inline) based on provider capabilities
    - Formatting into provider-specific content block structures

    Args:
        files: Dictionary mapping file names to FileInput objects.
        provider: Provider name (e.g., "openai", "anthropic", "bedrock", "gemini").
        api: API variant (e.g., "responses" for OpenAI Responses API).
        prefer_upload: Whether to prefer uploading files instead of inlining.
            If None, uses provider-specific defaults.
        text: Optional text content to include as the first content block.

    Returns:
        List of content blocks in the provider's expected format.
        If text is provided, it will be the first block.

    Example:
        >>> from crewai_files import format_multimodal_content, ImageFile
        >>> files = {"photo": ImageFile(source="image.jpg")}
        >>> blocks = format_multimodal_content(files, "openai", text="Describe this")
        >>> # For OpenAI Responses API:
        >>> blocks = format_multimodal_content(files, "openai", api="responses")
    """
    content_blocks: list[dict[str, Any]] = []
    provider_type = _normalize_provider(provider)

    # Add text block first if provided
    if text:
        content_blocks.append(_format_text_block(text, provider_type, api))

    if not files:
        return content_blocks

    # Use API-specific constraints for OpenAI
    constraints_key = provider_type
    if api == "responses" and "openai" in provider_type.lower():
        constraints_key = "openai_responses"

    processor = FileProcessor(constraints=constraints_key)
    processed_files = processor.process_files(files)

    if not processed_files:
        return content_blocks

    constraints = get_constraints_for_provider(constraints_key)
    supported_types = _get_supported_types(constraints)
    supported_files = _filter_supported_files(processed_files, supported_types)

    if not supported_files:
        return content_blocks

    config = _get_resolver_config(provider_type, prefer_upload)
    upload_cache = get_upload_cache()
    resolver = FileResolver(config=config, upload_cache=upload_cache)

    formatter = _get_formatter(provider_type, api)

    for name, file_input in supported_files.items():
        resolved = resolver.resolve(file_input, provider_type)
        block = _format_block(formatter, file_input, resolved, name)
        if block is not None:
            content_blocks.append(block)

    return content_blocks


async def aformat_multimodal_content(
    files: dict[str, FileInput],
    provider: str | None = None,
    api: str | None = None,
    prefer_upload: bool | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Async format text and files as provider-specific multimodal content blocks.

    Async version of format_multimodal_content with parallel file resolution.

    Args:
        files: Dictionary mapping file names to FileInput objects.
        provider: Provider name (e.g., "openai", "anthropic", "bedrock", "gemini").
        api: API variant (e.g., "responses" for OpenAI Responses API).
        prefer_upload: Whether to prefer uploading files instead of inlining.
            If None, uses provider-specific defaults.
        text: Optional text content to include as the first content block.

    Returns:
        List of content blocks in the provider's expected format.
        If text is provided, it will be the first block.
    """
    content_blocks: list[dict[str, Any]] = []
    provider_type = _normalize_provider(provider)

    if text:
        content_blocks.append(_format_text_block(text, provider_type, api))

    if not files:
        return content_blocks

    # Use API-specific constraints for OpenAI
    constraints_key = provider_type
    if api == "responses" and "openai" in provider_type.lower():
        constraints_key = "openai_responses"

    processor = FileProcessor(constraints=constraints_key)
    processed_files = await processor.aprocess_files(files)

    if not processed_files:
        return content_blocks

    constraints = get_constraints_for_provider(constraints_key)
    supported_types = _get_supported_types(constraints)
    supported_files = _filter_supported_files(processed_files, supported_types)

    if not supported_files:
        return content_blocks

    config = _get_resolver_config(provider_type, prefer_upload)
    upload_cache = get_upload_cache()
    resolver = FileResolver(config=config, upload_cache=upload_cache)

    resolved_files = await resolver.aresolve_files(supported_files, provider_type)

    formatter = _get_formatter(provider_type, api)

    for name, resolved in resolved_files.items():
        file_input = supported_files[name]
        block = _format_block(formatter, file_input, resolved, name)
        if block is not None:
            content_blocks.append(block)

    return content_blocks


def _get_supported_types(
    constraints: Any | None,
) -> list[str]:
    """Get list of supported MIME type prefixes from constraints.

    Args:
        constraints: Provider constraints.

    Returns:
        List of MIME type prefixes (e.g., ["image/", "application/pdf"]).
    """
    if constraints is None:
        return []

    supported: list[str] = []
    if constraints.image is not None:
        supported.append("image/")
    if constraints.pdf is not None:
        supported.append("application/pdf")
    if constraints.audio is not None:
        supported.append("audio/")
    if constraints.video is not None:
        supported.append("video/")
    if constraints.text is not None:
        supported.append("text/")
        supported.append("application/json")
        supported.append("application/xml")
        supported.append("application/x-yaml")
    return supported


def _filter_supported_files(
    files: dict[str, FileInput],
    supported_types: list[str],
) -> dict[str, FileInput]:
    """Filter files to those with supported content types.

    Args:
        files: All files.
        supported_types: MIME type prefixes to allow.

    Returns:
        Filtered dictionary of supported files.
    """
    return {
        name: f
        for name, f in files.items()
        if any(f.content_type.startswith(t) for t in supported_types)
    }


def _get_resolver_config(
    provider_lower: str,
    prefer_upload_override: bool | None = None,
) -> FileResolverConfig:
    """Get resolver config for provider.

    Args:
        provider_lower: Lowercase provider name.
        prefer_upload_override: Override for prefer_upload setting.
            If None, uses provider-specific defaults.

    Returns:
        Configured FileResolverConfig.
    """
    if "bedrock" in provider_lower:
        s3_bucket = os.environ.get("CREWAI_BEDROCK_S3_BUCKET")
        prefer_upload = (
            prefer_upload_override
            if prefer_upload_override is not None
            else bool(s3_bucket)
        )
        return FileResolverConfig(
            prefer_upload=prefer_upload, use_bytes_for_bedrock=True
        )

    prefer_upload = (
        prefer_upload_override if prefer_upload_override is not None else False
    )
    return FileResolverConfig(prefer_upload=prefer_upload)


def _get_formatter(
    provider_lower: str,
    api: str | None = None,
) -> (
    OpenAIFormatter
    | OpenAIResponsesFormatter
    | AnthropicFormatter
    | BedrockFormatter
    | GeminiFormatter
):
    """Get formatter for provider.

    Args:
        provider_lower: Lowercase provider name.
        api: API variant (e.g., "responses" for OpenAI Responses API).

    Returns:
        Provider-specific formatter instance.
    """
    if "anthropic" in provider_lower or "claude" in provider_lower:
        return AnthropicFormatter()

    if "bedrock" in provider_lower or "aws" in provider_lower:
        s3_bucket_owner = os.environ.get("CREWAI_BEDROCK_S3_BUCKET_OWNER")
        return BedrockFormatter(s3_bucket_owner=s3_bucket_owner)

    if "gemini" in provider_lower or "google" in provider_lower:
        return GeminiFormatter()

    if api == "responses":
        return OpenAIResponsesFormatter()

    return OpenAIFormatter()


def _format_block(
    formatter: OpenAIFormatter
    | OpenAIResponsesFormatter
    | AnthropicFormatter
    | BedrockFormatter
    | GeminiFormatter,
    file_input: FileInput,
    resolved: Any,
    name: str,
) -> dict[str, Any] | None:
    """Format a single file block using the appropriate formatter.

    Args:
        formatter: Provider formatter.
        file_input: Original file input.
        resolved: Resolved file.
        name: File name.

    Returns:
        Content block dict or None.
    """
    if isinstance(formatter, BedrockFormatter):
        return formatter.format_block(file_input, resolved, name=name)
    if isinstance(formatter, AnthropicFormatter):
        return formatter.format_block(file_input, resolved)
    if isinstance(formatter, OpenAIResponsesFormatter):
        return formatter.format_block(resolved, file_input.content_type)
    if isinstance(formatter, (OpenAIFormatter, GeminiFormatter)):
        return formatter.format_block(resolved)
    raise TypeError(f"Unknown formatter type: {type(formatter).__name__}")
