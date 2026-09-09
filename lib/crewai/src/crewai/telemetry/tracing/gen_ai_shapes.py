"""Spec-shape value builders for OpenTelemetry GenAI attributes.

That module owns the attribute *keys* (`gen_ai.input.messages`, `gen_ai.tool.definitions`, ...);
this one owns the attribute *values* — turning CrewAI/OpenAI-native payloads
into the JSON arrays of typed parts that the spec requires for
`gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions`, and `gen_ai.tool.definitions`, plus the
flat shapes for `gen_ai.tool.call.{arguments,result}` and
`gen_ai.response.finish_reasons`.

Spec-strict consumers reject the raw
`{role, content}` dicts CrewAI emits, so we transform just before span
emission.

Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import json
import os
from typing import Any

from crewai.utilities.serialization import to_serializable


_MAX_DEPTH = 14

DEFAULT_MAX_ATTR_BYTES = 32 * 1024
_PLACEHOLDER_ROLE = "system"
_TRUNCATION_LOOP_LIMIT = 8

_FINISH_REASONS = {"stop", "length", "content_filter", "tool_call", "error"}

# Provider-native finish-reason values mapped to the OTel GenAI enum.
# Keys are lowercased; lookups must lowercase the raw value first so we cover
# Gemini's UPPERCASE protobuf enum (STOP, MAX_TOKENS, SAFETY, ...) and the
# lowercase variants emitted by LiteLLM / Anthropic / Bedrock with a single
# entry per alias.
_FINISH_REASON_ALIASES = {
    # → stop
    "end_turn": "stop",  # Anthropic
    "stop_sequence": "stop",  # Anthropic
    "pause_turn": "stop",  # Anthropic (long-running interruption)
    "completed": "stop",  # OpenAI Responses API
    # → length
    "max_tokens": "length",  # Anthropic / Gemini
    "model_context_window_exceeded": "length",  # Anthropic
    "incomplete": "length",  # OpenAI Responses API
    # → content_filter
    "safety": "content_filter",  # Gemini
    "blocklist": "content_filter",  # Gemini
    "prohibited_content": "content_filter",  # Gemini
    "spii": "content_filter",  # Gemini (sensitive personal info)
    "model_armor": "content_filter",  # Gemini
    "image_safety": "content_filter",  # Gemini
    "image_prohibited_content": "content_filter",  # Gemini
    "refusal": "content_filter",  # Anthropic
    "content_filtered": "content_filter",  # Bedrock
    "guardrail_intervened": "content_filter",  # Bedrock
    # → tool_call
    "tool_calls": "tool_call",  # OpenAI (plural form)
    "function_call": "tool_call",  # Gemini / legacy OpenAI
    "tool_use": "tool_call",  # Anthropic / Bedrock
    # → error  (provider says the call completed but with a content anomaly)
    "recitation": "error",  # Gemini (verbatim reproduction)
    "image_recitation": "error",  # Gemini
    "malformed_function_call": "error",  # Gemini
    "other": "error",  # Gemini (catch-all)
    "image_other": "error",  # Gemini
    "language": "error",  # Gemini (unsupported language)
    "failed": "error",  # OpenAI Responses API
    "cancelled": "error",  # OpenAI Responses API
}

_FINISH_REASON_DROP = {
    "finish_reason_unspecified",  # Gemini
    "in_progress",  # OpenAI Responses API (lifecycle state, not a finish reason)
    "queued",  # OpenAI Responses API (lifecycle state, not a finish reason)
}


def _normalize_finish_reason(value: Any) -> str | None:
    """Normalize any provider-native finish reason to the canonical OTel enum.

    Returns ``None`` for missing / empty / ``FINISH_REASON_UNSPECIFIED`` values,
    and ``"error"`` for non-empty values that don't map to any known enum or
    alias. Callers pick their own policy for the ``None`` case:

    - :func:`coerce_finish_reason` (explicit-finish path) keeps ``None`` so the
      caller can fall back to inferring the reason from the output payload.
    - :func:`_coerce_finish` (inference path) substitutes the caller's default
      (typically ``"stop"``).
    """
    if not isinstance(value, str) or not value:
        return None
    lowered = value.lower()
    if lowered in _FINISH_REASON_DROP:
        return None
    if lowered in _FINISH_REASONS:
        return lowered
    if lowered in _FINISH_REASON_ALIASES:
        return _FINISH_REASON_ALIASES[lowered]
    # Intentional: unknown non-empty values surface as `"error"` so a new
    # provider enum value (e.g. Gemini ships a new variant) triggers anomaly
    # alerts and someone adds the mapping above, rather than silently
    # collapsing to `"stop"` and hiding the regression.
    return "error"


def coerce_finish_reason(raw: str | None) -> str | None:
    """Coerce a provider-native finish reason to the OTel enum value.

    Returns ``None`` for empty/missing values and the explicit
    ``FINISH_REASON_UNSPECIFIED`` sentinel; returns ``"error"`` for any
    other value that does not match the OTel enum or its known aliases so
    that unrecognised values surface as anomalies rather than being
    silently swallowed.
    """
    return _normalize_finish_reason(raw)


def _coerce_finish(value: Any, *, default: str) -> str:
    """Coerce an inferred finish reason to the OTel enum value.

    Used by the inference path in :func:`to_output_messages` where a
    missing/blank/unspecified value falls back to *default* (typically
    ``"stop"``). Explicit-finish callers should use
    :func:`coerce_finish_reason` instead, which returns ``None`` for the
    same cases.
    """
    return _normalize_finish_reason(value) or default


def to_input_messages(messages: Any) -> list[dict[str, Any]] | None:
    if not messages:
        return None
    norm = to_serializable(messages, max_depth=_MAX_DEPTH)
    if isinstance(norm, str):
        return [_msg("user", [_text(norm)])]
    if isinstance(norm, dict):
        return [_input_msg(norm)]
    if isinstance(norm, list):
        return [_input_msg(m) for m in norm] or None
    return [_msg("user", [_text(_to_str(norm))])]


def to_output_messages(response: Any) -> list[dict[str, Any]] | None:
    if response is None or response == "":
        return None
    norm = to_serializable(response, max_depth=_MAX_DEPTH)
    if isinstance(norm, str):
        return [_msg("assistant", [_text(norm)], finish="stop")]
    if isinstance(norm, list):
        if not norm:
            return None
        if all(_looks_like_tool_call(i) for i in norm):
            return [
                _msg("assistant", [_tool_call(i) for i in norm], finish="tool_call")
            ]
        return [_msg("assistant", [_text(_to_str(i)) for i in norm], finish="stop")]
    if isinstance(norm, dict):
        finish = _coerce_finish(norm.get("finish_reason"), default="stop")
        role = _role(norm.get("role"), "assistant")
        raw_parts = norm.get("parts")
        if isinstance(raw_parts, list):
            return [
                _msg(
                    role,
                    [
                        part if isinstance(part, dict) else _text(_to_str(part))
                        for part in raw_parts
                    ],
                    finish=finish,
                )
            ]
        parts = _input_msg(norm)["parts"]
        if norm.get("tool_calls"):
            finish = "tool_call"
        return [_msg(role, parts, finish=finish)]
    return [_msg("assistant", [_text(_to_str(norm))], finish="stop")]


def to_system_instructions(value: Any) -> list[dict[str, Any]] | None:
    if not value:
        return None
    norm = to_serializable(value, max_depth=_MAX_DEPTH)
    if isinstance(norm, str):
        return [_text(norm)]
    if isinstance(norm, list):
        parts = [
            p if isinstance(p, dict) and "type" in p else _text(_to_str(p))
            for p in norm
            if p
        ]
        return parts or None
    return [_text(_to_str(norm))]


def to_tool_definitions(tools: Any) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    raw = tools if isinstance(tools, list) else [tools]
    out: list[dict[str, Any]] = []
    for tool in raw:
        # CrewAI BaseTool: `args_schema` is a Pydantic class — `to_serializable`
        # would `repr()` it and lose the schema. Pull it out ourselves.
        if hasattr(tool, "name") and hasattr(tool, "args_schema"):
            defn: dict[str, Any] = {
                "type": "function",
                "name": getattr(tool, "name", "") or "",
            }
            description = getattr(tool, "description", None)
            if isinstance(description, str) and description:
                defn["description"] = description
            schema = getattr(tool, "args_schema", None)
            if schema is not None and hasattr(schema, "model_json_schema"):
                try:
                    defn["parameters"] = schema.model_json_schema()
                except Exception:  # noqa: S110 - intentional: see below
                    # Pydantic raises a wide set of types here
                    # (PydanticInvalidForJsonSchema, TypeError, AttributeError,
                    # ValueError, ...). Catching narrowly would let
                    # PydanticInvalidForJsonSchema propagate to `_safe_serialize`,
                    # which would then drop the whole `gen_ai.tool.definitions`
                    # attribute. We prefer graceful degradation: emit this tool's
                    # definition without `parameters` and keep the rest.
                    pass
            if defn["name"]:
                out.append(defn)
            continue
        norm = to_serializable(tool, max_depth=_MAX_DEPTH)
        if not isinstance(norm, dict):
            continue
        inner = norm.get("function")
        if not isinstance(inner, dict):
            inner = norm
        name = inner.get("name")
        if not isinstance(name, str) or not name:
            continue
        defn = {"type": norm.get("type") or "function", "name": name}
        if isinstance(inner.get("description"), str) and inner["description"]:
            defn["description"] = inner["description"]
        if inner.get("parameters") is not None:
            defn["parameters"] = inner["parameters"]
        out.append(defn)
    return out or None


def _parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def to_tool_call_arguments(arguments: Any) -> Any:
    if arguments is None:
        return None
    if isinstance(arguments, str):
        return _parse_json(arguments)
    return to_serializable(arguments, max_depth=_MAX_DEPTH)


def to_tool_call_result(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, str):
        return _parse_json(result)
    return to_serializable(result, max_depth=_MAX_DEPTH)


def finish_reasons_from_messages(
    messages: list[dict[str, Any]] | None,
) -> list[str] | None:
    if not messages:
        return None
    return [m["finish_reason"] for m in messages if m.get("finish_reason")] or None


def _input_msg(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _msg("user", [_text(_to_str(raw))])
    role = _role(raw.get("role"), "user")
    parts: list[dict[str, Any]] = []
    content = raw.get("content")
    if isinstance(content, str):
        if content:
            parts.append(_text(content))
    elif content is not None:
        parts.append(_text(_to_str(content)))
    parts.extend(_tool_call(tc) for tc in raw.get("tool_calls") or [])
    return _msg(role, parts or [_text("")])


def _tool_call(tc: Any) -> dict[str, Any]:
    if not isinstance(tc, dict):
        return _text(_to_str(tc))
    fn = tc.get("function")
    if not isinstance(fn, dict):
        fn = tc
    part: dict[str, Any] = {
        "type": "tool_call",
        "name": fn.get("name") or "",
        "arguments": to_tool_call_arguments(fn.get("arguments")),
    }
    if isinstance(tc.get("id"), str) and tc["id"]:
        part["id"] = tc["id"]
    return part


def _looks_like_tool_call(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return (
        "function" in item
        or item.get("type") in {"function", "tool_call"}
        or ("name" in item and "arguments" in item)
    )


def _msg(
    role: str, parts: list[dict[str, Any]], *, finish: str | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": role, "parts": parts}
    if finish is not None:
        msg["finish_reason"] = finish
    return msg


def _text(content: str) -> dict[str, Any]:
    return {"type": "text", "content": content}


def _role(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def truncate_attr(
    payload: str | None,
    *,
    attr: str,
    max_bytes: int | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Bound the byte size of a serialized GenAI attribute, preserving JSON."""
    if payload is None:
        return None, {}

    cap = max_bytes if max_bytes is not None else _max_attr_bytes()
    original_size = _byte_len(payload)
    if original_size <= cap:
        return payload, {}

    markers = {
        f"{attr}.truncated": True,
        f"{attr}.original_size_bytes": original_size,
    }

    try:
        parsed = json.loads(payload)
    except (ValueError, TypeError):
        return _envelope(payload, original_size, cap), markers

    if _is_message_array(parsed):
        return _truncate_messages(parsed, original_size, cap), markers
    return _envelope(payload, original_size, cap), markers


def _max_attr_bytes() -> int:
    raw = os.environ.get("CREWAI_OTEL_MAX_ATTR_BYTES")
    if not raw:
        return DEFAULT_MAX_ATTR_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_ATTR_BYTES
    return value if value > 0 else DEFAULT_MAX_ATTR_BYTES


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _is_message_array(parsed: Any) -> bool:
    if not isinstance(parsed, list) or not parsed:
        return False
    return all(isinstance(item, dict) and "role" in item for item in parsed)


def _truncate_messages(
    messages: list[dict[str, Any]], original_size: int, cap: int
) -> str:
    if len(messages) <= 2:
        return _shrink_text_until_fits(
            [_clone_message(m) for m in messages], original_size, cap
        )

    head, tail = messages[0], messages[-1]
    middle = messages[1:-1]
    middle_size = _byte_len(json.dumps(middle, default=str))
    placeholder = _placeholder_message(len(middle), middle_size)
    candidate = [_clone_message(head), placeholder, _clone_message(tail)]

    serialized = json.dumps(candidate, default=str)
    if _byte_len(serialized) <= cap:
        return serialized
    return _shrink_text_until_fits(candidate, original_size, cap)


def _placeholder_message(count: int, byte_size: int) -> dict[str, Any]:
    kb = max(1, byte_size // 1024)
    return _msg(
        _PLACEHOLDER_ROLE,
        [_text(f"[truncated {count} messages, ~{kb}KB]")],
    )


def _clone_message(msg: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {**msg}
    parts = msg.get("parts")
    if isinstance(parts, list):
        cloned["parts"] = [{**p} if isinstance(p, dict) else p for p in parts]
    return cloned


def _shrink_text_until_fits(
    messages: list[dict[str, Any]], original_size: int, cap: int
) -> str:
    """Iteratively halve the largest ``parts[*].content`` text until fits.

    Bounded by ``_TRUNCATION_LOOP_LIMIT`` so a pathological input can't
    spin forever. Falls back to an envelope if shrinking text alone
    can't get under cap (rare — happens when message envelopes/keys
    themselves dominate).
    """
    for _ in range(_TRUNCATION_LOOP_LIMIT):
        serialized = json.dumps(messages, default=str)
        if _byte_len(serialized) <= cap:
            return serialized
        target = _largest_text_part(messages)
        if target is None:
            break
        part, content = target
        new_content = _trunc_text(content, _byte_len(content) // 2)
        if new_content == content:
            break
        part["content"] = new_content
    return _envelope(json.dumps(messages, default=str), original_size, cap)


def _largest_text_part(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], str] | None:
    largest: tuple[dict[str, Any], str] | None = None
    largest_len = 0
    for msg in messages:
        for part in msg.get("parts") or []:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            content = part.get("content")
            if not isinstance(content, str):
                continue
            blen = _byte_len(content)
            if blen > largest_len:
                largest_len = blen
                largest = (part, content)
    return largest


def _trunc_text(content: str, target_bytes: int) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= target_bytes:
        return content
    head_bytes = max(target_bytes // 2, 256)
    tail_bytes = max(target_bytes // 4, 128)
    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore")
    omitted_kb = max(1, (len(encoded) - head_bytes - tail_bytes) // 1024)
    return f"{head}...[truncated {omitted_kb}KB]...{tail}"


def _envelope(payload: str, original_size: int, cap: int) -> str:
    """Replace any payload with a parseable JSON object containing a head
    preview and the truncation metadata.

    Used when the structural strategy doesn't apply (non-message JSON,
    malformed JSON) or didn't fit (rare — message envelopes dominate).
    """
    preview_bytes = max(min(cap // 4, 4 * 1024), 512)
    encoded = payload.encode("utf-8")
    preview = encoded[:preview_bytes].decode("utf-8", errors="ignore")
    return json.dumps(
        {
            "_truncated": True,
            "_original_size_bytes": original_size,
            "_preview": preview,
        }
    )
