"""Shared helpers for working with LLM message content."""

from __future__ import annotations

from crewai.utilities.types import LLMMessage


def message_content_text(msg: LLMMessage) -> str:
    """Return text content without rendering multimodal parts as Python reprs."""
    content = msg.get("content")
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    text_parts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return " ".join(text_parts) if text_parts else "[multimodal content]"
