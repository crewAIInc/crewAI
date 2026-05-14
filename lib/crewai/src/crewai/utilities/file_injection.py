"""Helpers for deciding which input files can be injected into LLM messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from crewai.llms.base_llm import BaseLLM


try:
    from crewai_files import get_supported_content_types
except ImportError:

    def get_supported_content_types(provider: str, api: str | None = None) -> list[str]:
        return []


def get_auto_injected_files(
    files: Mapping[str, Any],
    llm: Any,
) -> dict[str, Any]:
    """Return files that the LLM can receive directly in message content."""
    if not isinstance(llm, BaseLLM) or not llm.supports_multimodal():
        return {}

    provider = (
        getattr(llm, "provider", None) or getattr(llm, "model", None) or "openai"
    )
    api = getattr(llm, "api", None)
    supported_types = get_supported_content_types(provider, api)

    return {
        name: file_input
        for name, file_input in files.items()
        if any(
            getattr(file_input, "content_type", "").startswith(content_type)
            for content_type in supported_types
        )
    }
