"""Shared helpers for ``finish_reason`` handling across LLM providers.

Two concerns live here:

1. ``extract_choices_finish_reason_and_id`` — extraction. OpenAI Chat
   Completions, Azure AI Inference, and LiteLLM all expose the same
   choices-based response shape (``response.id`` +
   ``response.choices[0].finish_reason``), both as object attributes and
   (for LiteLLM stream chunks) as dict keys. This centralises that
   introspection so every provider doesn't reinvent the defensive walk.
   Providers with genuinely different shapes — Anthropic (``stop_reason``),
   Bedrock (``stopReason``), Gemini (protobuf enum), OpenAI Responses
   (``status``) — keep their own helpers.
2. ``warn_if_truncated`` — interpretation. Providers report token-cap
   truncation with provider-specific values (``"length"``, ``"max_tokens"``,
   ``"MAX_TOKENS"``, Responses ``"incomplete"``); this maps each to one
   normalized warning so no provider silently accepts a truncated response
   as complete (#7013).
"""

from __future__ import annotations

from enum import Enum
from logging import Logger
from typing import Any


#: Raw provider values that mean "generation stopped because the token cap
#: was hit", keyed by provider. OpenAI's ``"incomplete"`` covers the
#: Responses API, where ``status`` (not a finish reason) is forwarded;
#: most ``incomplete`` responses are cut off by ``max_output_tokens``.
_TRUNCATION_FINISH_REASONS: dict[str, frozenset[str]] = {
    "openai": frozenset({"length", "incomplete"}),
    "azure": frozenset({"length"}),
    "anthropic": frozenset({"max_tokens"}),
    "bedrock": frozenset({"max_tokens"}),
    "gemini": frozenset({"MAX_TOKENS"}),
}


def warn_if_truncated(finish_reason: str | None, provider: str, *, log: Logger) -> bool:
    """Emit one normalized warning when a provider reports a token-cap truncation.

    This is warning-only by design (matching the long-standing Bedrock
    behavior): the truncated text is still returned unchanged. Pointing this
    at a result-level flag or exception is deliberately left as a follow-up.

    Args:
        finish_reason: The raw terminal reason reported by the provider
            (``finish_reason``/``stop_reason``/``status``/enum ``.name``).
            SDK-specific ``str``-mixin enums (e.g. Azure's
            ``CompletionsFinishReason``) are matched and rendered by their
            underlying wire value.
        provider: CrewAI provider key (``openai``, ``azure``, ``anthropic``,
            ``bedrock``, ``gemini``).
        log: Logger to emit the warning through. Callers pass the root
            logger (``logging.getLogger()``), matching how the providers
            log today.

    Returns:
        True if a truncation warning was emitted, False otherwise.
    """
    if not finish_reason:
        return False
    # Azure SDK turns the wire value into a ``str``-mixin enum
    # (``CompletionsFinishReason.TOKEN_LIMIT_REACHED``); render and match on
    # the underlying value so logs and mapping stay provider-wire shaped.
    reason = (
        str(finish_reason.value) if isinstance(finish_reason, Enum) else finish_reason
    )
    if reason not in _TRUNCATION_FINISH_REASONS.get(provider, frozenset()):
        return False
    log.warning(
        "Response truncated by token cap (provider=%s, finish_reason=%s); "
        "output may be incomplete",
        provider,
        reason,
    )
    return True


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
