"""Shared context-window lookup for CrewAI LLMs.

Limits come from a checked-in snapshot of LiteLLM's model catalog:
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

``resolve_context_window_size`` applies the usable context ratio, so every
call site shares one matching rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.resources import files
import json
import re
from typing import Any, Final


DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

_CATALOG_FILENAME: Final[str] = "model_prices_and_context_window.json"
_OPENAI_PROVIDERS: Final[frozenset[str]] = frozenset({"openai"})
_AZURE_PROVIDERS: Final[frozenset[str]] = frozenset({"azure", "azure_ai"})
_ANTHROPIC_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic"})
_GEMINI_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"gemini", "vertex_ai-language-models"}
)
_BEDROCK_PROVIDERS: Final[frozenset[str]] = frozenset({"bedrock", "bedrock_converse"})
_DATED_CLAUDE_ID: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>.+\.anthropic\.claude-(?:sonnet|opus|haiku)-\d+-\d+)"
    r"-\d{8}(?P<version>-v\d+:\d+)$"
)
# These routed Gemini IDs have no row anywhere in the LiteLLM catalog.
_GEMINI_CONTEXT_OVERRIDES: Final[dict[str, int]] = {
    "gemma-3-1b-it": 32_000,
    "gemma-3-4b-it": 128_000,
    "gemma-3-12b-it": 128_000,
}


def context_windows_from_catalog(
    catalog: Mapping[str, Any],
    *,
    providers: frozenset[str] | None = None,
) -> dict[str, int]:
    """Build model-id context windows from a LiteLLM catalog payload.

    Catalog keys are kept as published. When a key is ``provider/model`` and
    that provider matches ``litellm_provider``, the bare model id is also
    indexed so native providers can look up the id they send to the API.
    Exact catalog keys override those bare aliases.
    """
    aliases: dict[str, int] = {}
    exact: dict[str, int] = {}
    for name, props in catalog.items():
        if name == "sample_spec" or not isinstance(props, dict):
            continue
        provider = props.get("litellm_provider")
        if providers is not None and provider not in providers:
            continue
        limit = props.get("max_input_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int):
            continue
        exact[name] = limit
        provider_name, separator, remainder = name.partition("/")
        if separator and remainder and provider_name == provider:
            aliases.setdefault(remainder, limit)
    return {**aliases, **exact}


def _with_dated_claude_aliases(sizes: Mapping[str, int]) -> dict[str, int]:
    """Alias an undated Claude inference ID to its one dated catalog entry.

    ``us.anthropic.claude-sonnet-4-5-v1:0`` is not produced by removing a
    suffix from ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``. The date
    sits in the middle, so each routed short ID needs an explicit alias.
    """
    limits_by_short_id: dict[str, set[int]] = {}
    for name, limit in sizes.items():
        match = _DATED_CLAUDE_ID.fullmatch(name)
        if match is None:
            continue
        short_id = f"{match.group('family')}{match.group('version')}"
        limits_by_short_id.setdefault(short_id, set()).add(limit)
    aliases = {
        short_id: next(iter(limits))
        for short_id, limits in limits_by_short_id.items()
        if len(limits) == 1 and short_id not in sizes
    }
    return {**aliases, **sizes}


def _load_catalog() -> dict[str, Any]:
    resource = files("crewai.llms").joinpath("data", _CATALOG_FILENAME)
    catalog = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("LiteLLM context catalog must be a JSON object")
    return catalog


_CATALOG = _load_catalog()
LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    **_GEMINI_CONTEXT_OVERRIDES,
    **context_windows_from_catalog(_CATALOG),
}
OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_OPENAI_PROVIDERS
)
AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_AZURE_PROVIDERS
)
AZURE_OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
}
ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_ANTHROPIC_PROVIDERS
)
GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **_GEMINI_CONTEXT_OVERRIDES,
    **context_windows_from_catalog(_CATALOG, providers=_GEMINI_PROVIDERS),
}
BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = _with_dated_claude_aliases(
    context_windows_from_catalog(_CATALOG, providers=_BEDROCK_PROVIDERS)
)


def _lookup_context_limit(name: str, sizes: Mapping[str, int]) -> int | None:
    """Return the limit for an exact ID, then for the ID with trailing ``:`` segments removed."""
    seen: set[str] = set()
    candidate = name
    while candidate and candidate not in seen:
        seen.add(candidate)
        size = sizes.get(candidate)
        if size is not None:
            return size
        stem, separator, _segment = candidate.rpartition(":")
        if not separator or not stem:
            return None
        candidate = stem
    return None


def resolve_context_window_size(
    model: str,
    sizes: Mapping[str, int],
    *,
    default: int,
    extra_names: Sequence[str] = (),
) -> int:
    """Return the usable context window for a catalog model id."""
    for name in (model, *extra_names):
        size = _lookup_context_limit(name, sizes)
        if size is not None:
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)
    return int(default * CONTEXT_WINDOW_USAGE_RATIO)
