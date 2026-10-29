"""Lookup LiteLLM's bundled model catalog by provider and model id."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from importlib.resources import files
import json
import re
from typing import Any, Final


_CATALOG_FILENAME: Final[str] = "model_prices_and_context_window.json"
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

# CrewAI's routing names are not always LiteLLM provider names. Each group is
# deliberately small: a lookup never crosses unrelated provider catalogs.
_PROVIDER_EQUIVALENTS: Final[dict[str, frozenset[str]]] = {
    "anthropic": frozenset({"anthropic"}),
    "aws": frozenset({"bedrock", "bedrock_converse"}),
    "azure": frozenset({"azure", "azure_ai"}),
    "azure_openai": frozenset({"azure", "azure_ai"}),
    "bedrock": frozenset({"bedrock", "bedrock_converse"}),
    "claude": frozenset({"anthropic"}),
    "gemini": frozenset({"gemini", "vertex_ai-language-models"}),
    "google": frozenset({"gemini", "vertex_ai-language-models"}),
    "openai": frozenset({"openai"}),
}


def bare_model_name(name: str) -> str:
    """Return a provider deployment id's catalog model-family name."""
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
    return _deployment_model_name(_DATE_IN_MODEL_ID.sub("", name))


def _deployment_model_name(name: str) -> str:
    """Remove only Bedrock-style invocation suffixes from a model id."""
    while True:
        stem, separator, segment = name.rpartition(":")
        if not separator or not segment:
            break
        name = stem
    version = _TRAILING_VERSION.fullmatch(name)
    if version is not None and "-" in version.group("body"):
        return version.group("body")
    return name


def _load_catalog() -> dict[str, Any]:
    resource = files("crewai.llms").joinpath("litellm_catalog", _CATALOG_FILENAME)
    catalog = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("LiteLLM model catalog must be a JSON object")
    return catalog


def _entry_model_names(name: str, provider: str) -> tuple[str, ...]:
    """Return catalog names by which ``provider`` publishes an entry."""
    prefix, separator, remainder = name.partition("/")
    if separator and remainder and prefix == provider:
        return (name, remainder)
    return (name,)


def _index_catalog(
    catalog: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], tuple[dict[str, Any], ...]],
    dict[tuple[str, str], tuple[dict[str, Any], ...]],
]:
    entries: dict[str, dict[str, Any]] = {}
    exact: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for name, props in catalog.items():
        if not isinstance(props, dict):
            continue
        provider = props.get("litellm_provider")
        if not isinstance(provider, str):
            continue
        entries[name] = props
        models = _entry_model_names(name, provider)
        for model in models:
            exact[provider, model].append(props)
        # ``provider/model`` and ``model`` normalize to one name; index it once.
        for model in dict.fromkeys(bare_model_name(model) for model in models):
            normalized[provider, model].append(props)
    return (
        entries,
        {key: tuple(value) for key, value in exact.items()},
        {key: tuple(value) for key, value in normalized.items()},
    )


CATALOG = _load_catalog()
_CATALOG_ENTRIES, _EXACT_ENTRIES, _NORMALIZED_ENTRIES = _index_catalog(CATALOG)


def catalog_provider_prefixes() -> frozenset[str]:
    """Return provider prefixes that catalog keys actually use."""
    prefixes: set[str] = set()
    for name, props in CATALOG.items():
        if not isinstance(props, dict):
            continue
        provider = props.get("litellm_provider")
        if not isinstance(provider, str):
            continue
        prefix, separator, remainder = name.partition("/")
        if separator and remainder and prefix == provider:
            prefixes.add(prefix.lower())
    return frozenset(prefixes)


def find_model_catalog_entry(
    model: str, provider: str | None = None
) -> dict[str, Any] | None:
    """Return the LiteLLM entry for an unambiguous model lookup.

    The entry is shared catalog data; callers must treat it as read-only.

    With no provider, only a literal catalog key can match. With a provider,
    exact provider/model matches take precedence over provider-scoped normalized
    deployment names. A match shared by more than one catalog entry is left
    unresolved rather than choosing one arbitrarily.
    """
    direct = _CATALOG_ENTRIES.get(model)
    if provider is None:
        return direct

    provider = provider.lower()
    providers = _PROVIDER_EQUIVALENTS.get(provider, frozenset({provider}))
    if direct is not None and direct.get("litellm_provider") in providers:
        return direct

    exact = tuple(
        entry for name in providers for entry in _EXACT_ENTRIES.get((name, model), ())
    )
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    deployment_model = _deployment_model_name(model)
    if deployment_model != model:
        deployment = tuple(
            entry
            for name in providers
            for entry in _EXACT_ENTRIES.get((name, deployment_model), ())
        )
        if len(deployment) == 1:
            return deployment[0]
        if len(deployment) > 1:
            return None

    normalized_model = bare_model_name(model)
    normalized = tuple(
        entry
        for name in providers
        for entry in _NORMALIZED_ENTRIES.get((name, normalized_model), ())
    )
    return normalized[0] if len(normalized) == 1 else None
