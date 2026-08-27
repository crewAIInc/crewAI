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

import logging
from typing import Any


# Providers spell "I ran out of output budget" differently: OpenAI/Azure Chat use
# ``length``, Anthropic ``max_tokens``, Bedrock ``max_tokens`` via ``stopReason``,
# Gemini ``MAX_TOKENS``, and the OpenAI Responses API reports
# ``incomplete_details.reason == "max_output_tokens"`` rather than a finish reason
# at all. Compared case-insensitively with separators stripped so one predicate
# covers all of them.
_TRUNCATION_REASONS = frozenset(
    {"length", "maxtokens", "modellength", "maxoutputtokens"}
)


def _as_str(value: Any) -> str | None:
    """Return ``value`` when it is a string, else ``None``.

    Guards against provider types that are not plain strings (test doubles,
    protobuf enums) leaking downstream.
    """
    return value if isinstance(value, str) else None


def is_truncated(finish_reason: str | None) -> bool:
    """Whether ``finish_reason`` means the response was cut off by the token cap.

    Providers spell this differently: ``length`` (OpenAI, Azure), ``max_tokens``
    (Anthropic, Bedrock) and ``MAX_TOKENS`` (Gemini). Comparison is
    case-insensitive with separators stripped so one predicate covers all of them.

    Args:
        finish_reason: The raw finish/stop reason as reported by the provider.

    Returns:
        ``True`` if the value denotes truncation by the output-token cap.
    """
    if not isinstance(finish_reason, str):
        return False
    return (
        finish_reason.replace("_", "").replace("-", "").strip().lower()
        in _TRUNCATION_REASONS
    )


def warn_if_truncated(
    finish_reason: str | None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> bool:
    """Log a warning when a response was cut off by the token cap.

    A truncated response is otherwise indistinguishable from a complete one to
    everything downstream, so the caller cannot tell "the model was interrupted"
    from "the model answered badly".

    Args:
        finish_reason: The raw finish/stop reason as reported by the provider.
        max_tokens: The cap actually sent on the wire, so the message names the
            setting the reader needs to change. Providers differ here
            (``max_output_tokens`` on Gemini, ``max_completion_tokens`` on
            OpenAI and Azure reasoning models), so callers pass their effective
            value rather than a single shared attribute.
        model: The model the request was sent to, for diagnostics.

    Returns:
        ``True`` if a warning was emitted, ``False`` otherwise.
    """
    if not is_truncated(finish_reason):
        return False
    logging.warning(
        "Response truncated due to max_tokens limit (finish_reason=%r%s%s). "
        "The output is incomplete; consider increasing max_tokens.",
        finish_reason,
        f", model={model}" if model else "",
        f", max_tokens={max_tokens}" if max_tokens is not None else "",
    )
    return True


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
