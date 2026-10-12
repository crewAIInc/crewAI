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
_REGION_PREFIXES: Final[tuple[str, ...]] = (
    "us-gov.",
    "global.",
    "apac.",
    "us.",
    "eu.",
    "au.",
    "jp.",
)
_VENDOR_PREFIXES: Final[tuple[str, ...]] = (
    "anthropic.",
    "amazon.",
    "meta.",
    "google.",
    "openai.",
    "mistral.",
    "cohere.",
    "deepseek.",
    "qwen.",
    "ai21.",
    "nvidia.",
    "moonshotai.",
    "moonshot.",
    "xai.",
    "zai.",
    "writer.",
    "twelvelabs.",
    "minimax.",
)
_DATE_IN_MODEL_ID: Final[re.Pattern[str]] = re.compile(
    r"-\d{4}-\d{2}-\d{2}|-\d{8}|@\d{8}"
)
_TRAILING_VERSION: Final[re.Pattern[str]] = re.compile(r"(?P<body>.+)-v\d+$")


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


def _load_catalog() -> dict[str, Any]:
    resource = files("crewai.llms").joinpath("litellm_catalog", _CATALOG_FILENAME)
    catalog = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("LiteLLM context catalog must be a JSON object")
    return catalog


_CATALOG = _load_catalog()
LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = context_windows_from_catalog(_CATALOG)
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
GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_GEMINI_PROVIDERS
)
BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_BEDROCK_PROVIDERS
)


def bare_model_name(name: str) -> str:
    """Return the model family without provider, region, date, or version suffix.

    ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` and
    ``us.anthropic.claude-sonnet-4-5-v1:0`` both become ``claude-sonnet-4-5``.
    A trailing ``-vN`` is removed only when a model family remains, so
    ``claude-v2`` stays intact.
    """
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    for prefix in _REGION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for prefix in _VENDOR_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = _DATE_IN_MODEL_ID.sub("", name)
    while True:
        stem, separator, segment = name.rpartition(":")
        if not separator or not segment:
            break
        name = stem
    version = _TRAILING_VERSION.fullmatch(name)
    if version is not None and "-" in version.group("body"):
        name = version.group("body")
    return name


def bare_context_windows(sizes: Mapping[str, int]) -> dict[str, int]:
    """Index context limits by bare model name.

    A bare name is kept only when every catalog entry that produces it agrees
    on the limit. Disagreeing providers stay on the exact-id lookup.
    """
    limits_by_name: dict[str, set[int]] = {}
    for name, limit in sizes.items():
        limits_by_name.setdefault(bare_model_name(name), set()).add(limit)
    return {
        name: next(iter(limits))
        for name, limits in limits_by_name.items()
        if name and len(limits) == 1
    }


BARE_CONTEXT_WINDOWS: Final[dict[str, int]] = bare_context_windows(
    LLM_CONTEXT_WINDOW_SIZES
)


def resolve_context_window_size(
    model: str,
    sizes: Mapping[str, int],
    *,
    default: int,
    extra_names: Sequence[str] = (),
) -> int:
    """Return the usable context window for a catalog model id.

    Exact IDs are resolved in ``sizes``. A miss falls back to
    ``BARE_CONTEXT_WINDOWS``, which is built from the full catalog rather than
    the provider map.
    """
    for name in (model, *extra_names):
        size = sizes.get(name)
        if size is not None:
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)

    for name in (model, *extra_names):
        size = BARE_CONTEXT_WINDOWS.get(bare_model_name(name))
        if size is not None:
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)
    return int(default * CONTEXT_WINDOW_USAGE_RATIO)
