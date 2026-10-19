"""Shared helpers for creating and rendering LLM messages."""

from __future__ import annotations

from typing import Any, Literal

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


def format_message_for_llm(
    prompt: str, role: Literal["user", "assistant", "system"] = "user"
) -> LLMMessage:
    """Create an LLM message with trailing whitespace removed."""
    return {"role": role, "content": prompt.rstrip()}


def messages_content_text(messages: list[LLMMessage]) -> str:
    """Render conversation messages with role labels for a text prompt."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        if role == "assistant":
            prefix = "[ASSISTANT]:"
        elif role == "tool":
            prefix = f"[TOOL_RESULT ({message.get('name', 'unknown')})]:"
        else:
            prefix = "[USER]:"
        body = message_content_text(message)
        if message.get("content") is None:
            names = [
                call.get("function", {}).get("name", "unknown")
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and isinstance(call.get("function"), dict)
            ]
            body = f"[Called tools: {', '.join(names)}]" if names else ""
        lines.append(f"{prefix} {body}")
    return "\n\n".join(lines)


def collect_user_message_files(messages: list[LLMMessage]) -> dict[str, Any]:
    """Merge files attached to user messages."""
    files: dict[str, Any] = {}
    for message in messages:
        if message.get("role") == "user" and message.get("files"):
            files.update(message["files"])
    return files
