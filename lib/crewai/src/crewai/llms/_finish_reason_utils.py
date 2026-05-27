"""Shared extractors for ``finish_reason`` + ``response_id`` across LLM providers.

OpenAI Chat Completions, Azure AI Inference, and LiteLLM all expose the same
choices-based response shape (``response.id`` + ``response.choices[0].finish_reason``),
both as object attributes and (for LiteLLM stream chunks) as dict keys. This
module centralises that introspection so every provider doesn't reinvent the
defensive walk. Providers with genuinely different shapes — Anthropic
(``stop_reason``), Bedrock (``stopReason``), Gemini (protobuf enum), OpenAI
Responses (``status``) — keep their own helpers.
"""
from __future__ import annotations

from typing import Any


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def extract_choices_finish_reason_and_id(
    response_or_chunk: Any,
) -> tuple[str | None, str | None]:
    """Extract ``(finish_reason, response_id)`` from a choices-shaped response.

    Handles both object-style (``response.id``, ``response.choices[0].finish_reason``)
    and dict-style (``response["id"]``, ``response["choices"][0]["finish_reason"]``)
    inputs. Returns ``(None, None)`` on any failure; never raises. Non-string
    raw values are coerced to ``None`` so test mocks and exotic provider types
    (MagicMock, protobuf enums, etc.) don't propagate downstream.
    """
    raw_id = getattr(response_or_chunk, "id", None)
    if raw_id is None and isinstance(response_or_chunk, dict):
        raw_id = response_or_chunk.get("id")
    response_id = _as_str(raw_id)

    if isinstance(response_or_chunk, dict):
        choices = response_or_chunk.get("choices")
    else:
        choices = getattr(response_or_chunk, "choices", None)

    finish_reason: str | None = None
    if choices:
        try:
            first = choices[0]
        except (IndexError, TypeError, KeyError):
            first = None
        if first is not None:
            if isinstance(first, dict):
                raw_finish = first.get("finish_reason")
            else:
                raw_finish = getattr(first, "finish_reason", None)
            finish_reason = _as_str(raw_finish)

    return finish_reason, response_id
