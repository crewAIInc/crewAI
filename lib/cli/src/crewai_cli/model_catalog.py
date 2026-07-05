"""Dynamic model catalog for the crew-creation wizard.

Resolves the models to offer for a given provider using a three-tier strategy:

1. **Vendor API** - when the provider's API key is already present in the
   environment, query the vendor's own model-listing endpoint. This is the only
   source that reliably reflects the *latest* models (real release dates /
   display names, straight from the vendor).
2. **Curated hardcoded fallback** - the hand-verified list baked into the
   wizard, used when no API key is available. Authoritative but frozen, so it is
   refreshed periodically.
3. **LiteLLM feed** - the community ``model_prices_and_context_window.json`` the
   CLI already caches. Only used for providers with *no* curated list: the feed
   lags real releases badly (it can miss a vendor's newest models entirely), so
   it must never preempt the curated fallback.

Every tier is best-effort: any network error, timeout, missing key, or empty
result quietly falls through to the next tier, and the caller's hardcoded list
is always the final backstop. The picker never blocks for long — network calls
use a short timeout and successful results are cached.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import certifi
import httpx

from crewai_cli.constants import JSON_URL


# ── Tunables ─────────────────────────────────────────────────────

#: How many models to surface per provider.
MAX_MODELS = 8

#: Timeout (seconds) for any network call made while resolving models.
_TIMEOUT = 6.0

#: How long a resolved (dynamic) catalog stays fresh before we refetch.
_CATALOG_TTL = 6 * 3600

#: How long the shared LiteLLM feed cache stays fresh.
_LITELLM_TTL = 24 * 3600

#: Env var holding each provider's API key. Providers absent here (or with a
#: ``None`` value, e.g. local Ollama) skip the vendor-API tier.
_PROVIDER_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "ollama": None,
}

# Substrings that mark a model id as *not* a chat/completion model. Used to
# filter noisy OpenAI-compatible ``/models`` listings.
_NON_CHAT_MARKERS = (
    "embedding",
    "embed",
    "whisper",
    "tts",
    "audio",
    "transcribe",
    "realtime",
    "dall-e",
    "dalle",
    "image",
    "moderation",
    "similarity",
    "search",
    "-edit",
    "davinci-002",
    "babbage-002",
    "computer-use",
    "guard",
)

_ACRONYMS = {
    "gpt": "GPT",
    "ai": "AI",
    "nim": "NIM",
    "llm": "LLM",
    "hd": "HD",
    "us": "US",
    "eu": "EU",
}


# ── Public API ───────────────────────────────────────────────────


def get_provider_models(
    provider_key: str, fallback: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Return ``(model_id, label)`` pairs for ``provider_key``, newest first.

    Tries the vendor API (if a key is in the environment) first, since it is the
    only reliably-fresh source. When no key is available it returns the curated
    ``fallback`` verbatim — the LiteLLM feed is consulted **only** for providers
    with no curated list, because the feed lags real releases and would
    otherwise surface a staler list than the hand-verified fallback. Never
    raises: any failure degrades to the next tier.

    Args:
        provider_key: Short provider identifier, e.g. ``"anthropic"``.
        fallback: Curated ``(model_id, label)`` pairs to use as the backstop and
            to source friendly labels for known models.

    Returns:
        Up to :data:`MAX_MODELS` ``(model_id, label)`` pairs. Falls back to
        ``fallback`` verbatim when no fresher list can be resolved.
    """
    cached = _read_catalog_cache(provider_key)
    if cached is not None:
        return cached

    label_map = {model_id: label for model_id, label in fallback}

    entries = _from_vendor(provider_key)
    # The LiteLLM feed lags real releases, so only reach for it when we have no
    # curated fallback for this provider — never let it override the fallback.
    if not entries and not fallback:
        entries = _from_litellm(provider_key)
    if not entries:
        return fallback

    result = _finalize(entries, label_map)
    if not result:
        return fallback

    _write_catalog_cache(provider_key, result)
    return result


# ── Tier 1: vendor APIs ──────────────────────────────────────────


def _from_vendor(provider_key: str) -> list[dict[str, Any]] | None:
    """Fetch models straight from the vendor, or ``None`` if unavailable."""
    fetcher = _VENDOR_FETCHERS.get(provider_key)
    if fetcher is None:
        return None

    key_env = _PROVIDER_KEY_ENV.get(provider_key)
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        # Provider needs a key and none is set — skip to the next tier.
        return None

    try:
        entries = fetcher(api_key)
    except Exception:
        # Network error, auth failure, unexpected payload — degrade quietly.
        return None
    return entries or None


def _fetch_openai(api_key: str | None) -> list[dict[str, Any]]:
    return _fetch_openai_compatible("https://api.openai.com/v1", api_key)


def _fetch_groq(api_key: str | None) -> list[dict[str, Any]]:
    return _fetch_openai_compatible("https://api.groq.com/openai/v1", api_key)


def _fetch_cerebras(api_key: str | None) -> list[dict[str, Any]]:
    return _fetch_openai_compatible("https://api.cerebras.ai/v1", api_key)


def _fetch_openai_compatible(
    base_url: str, api_key: str | None
) -> list[dict[str, Any]]:
    """Parse an OpenAI-shaped ``GET /models`` response."""
    data = _http_get_json(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    entries: list[dict[str, Any]] = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if not model_id or not _is_chat_model(model_id):
            continue
        created = _as_float(item.get("created"))
        entries.append(_entry(model_id, _humanize(model_id), created=created))
    return entries


def _fetch_anthropic(api_key: str | None) -> list[dict[str, Any]]:
    data = _http_get_json(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key or "", "anthropic-version": "2023-06-01"},
    )
    entries: list[dict[str, Any]] = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if not model_id:
            continue
        label = item.get("display_name") or _humanize(model_id)
        created = _parse_iso(item.get("created_at"))
        entries.append(_entry(model_id, label, created=created))
    return entries


def _fetch_gemini(api_key: str | None) -> list[dict[str, Any]]:
    data = _http_get_json(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key or "", "pageSize": 200},
    )
    entries: list[dict[str, Any]] = []
    for item in data.get("models", []):
        methods = item.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = (item.get("name") or "").removeprefix("models/")
        if not name or not _is_chat_model(name) or "aqa" in name:
            continue
        label = item.get("displayName") or _humanize(name)
        # Gemini has no timestamp; rank by the version embedded in name/version.
        version_hint = f"{name} {item.get('version') or ''}"
        entries.append(_entry(name, label, version_hint=version_hint))
    return entries


def _fetch_ollama(_api_key: str | None) -> list[dict[str, Any]]:
    """List models installed on the local Ollama server (no API key)."""
    base = (
        os.environ.get("OLLAMA_API_BASE")
        or os.environ.get("API_BASE")
        or "http://localhost:11434"
    ).rstrip("/")
    data = _http_get_json(f"{base}/api/tags")
    entries: list[dict[str, Any]] = []
    for item in data.get("models", []):
        model_id = item.get("model") or item.get("name")
        if not model_id:
            continue
        # Ollama returns an ISO 8601 modified_at we can rank by.
        created = _parse_iso(item.get("modified_at"))
        entries.append(_entry(model_id, _humanize(model_id), created=created))
    return entries


_VENDOR_FETCHERS: dict[str, Callable[[str | None], list[dict[str, Any]]]] = {
    "openai": _fetch_openai,
    "anthropic": _fetch_anthropic,
    "gemini": _fetch_gemini,
    "groq": _fetch_groq,
    "cerebras": _fetch_cerebras,
    "ollama": _fetch_ollama,
}


# ── Tier 2: LiteLLM feed ─────────────────────────────────────────


def _from_litellm(provider_key: str) -> list[dict[str, Any]] | None:
    """Build chat-model entries for ``provider_key`` from the LiteLLM feed."""
    data = _load_litellm_data()
    if not data:
        return None

    entries: list[dict[str, Any]] = []
    for model_name, props in data.items():
        if not isinstance(props, dict):
            continue
        if props.get("litellm_provider", "").strip().lower() != provider_key:
            continue
        if props.get("mode") != "chat":
            continue
        # LiteLLM keys are sometimes prefixed with the provider; the picker
        # re-adds ``provider/`` itself, so strip a leading one to avoid dupes.
        model_id = model_name
        if model_id.startswith(f"{provider_key}/"):
            model_id = model_id[len(provider_key) + 1 :]
        if not model_id:
            continue
        entries.append(_entry(model_id, _humanize(model_id), version_hint=model_id))
    return entries or None


def _load_litellm_data() -> dict[str, Any] | None:
    """Return the cached LiteLLM feed, fetching it once if the cache is cold."""
    cache_file = _litellm_cache_file()
    fresh = (
        cache_file.exists()
        and (time.time() - cache_file.stat().st_mtime) < _LITELLM_TTL
    )
    if fresh:
        data = _read_json(cache_file)
        if data:
            return data

    try:
        data = _http_get_json(JSON_URL)
    except Exception:
        # Fall back to a stale cache if we have one, else give up on this tier.
        return _read_json(cache_file)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data


# ── Ranking + labelling ──────────────────────────────────────────


def _finalize(
    entries: list[dict[str, Any]], label_map: dict[str, str]
) -> list[tuple[str, str]]:
    """Sort newest-first, dedupe, relabel with curated names, and truncate."""
    entries.sort(key=lambda e: e["sort"], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for entry in entries:
        model_id = entry["id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        label = label_map.get(model_id) or entry["label"]
        out.append((model_id, label))
        if len(out) >= MAX_MODELS:
            break
    return out


def _entry(
    model_id: str,
    label: str,
    *,
    created: float = 0.0,
    version_hint: str | None = None,
) -> dict[str, Any]:
    """Build a rankable catalog entry.

    ``sort`` is a comparable tuple ``(created, date_int, version_tuple)`` so a
    real vendor timestamp wins, then a date embedded in the id, then the numeric
    version. Types line up positionally, so entries compare cleanly.
    """
    date_int, version = _version_key(version_hint or model_id)
    return {
        "id": model_id,
        "label": label,
        "sort": (created, date_int, version),
    }


_DATE_RE = re.compile(r"(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")
_NUM_RE = re.compile(r"\d+")


def _version_key(text: str) -> tuple[int, tuple[int, ...]]:
    """Extract a ``(date_int, version_tuple)`` sort key from a model id.

    A trailing/embedded ``YYYYMMDD`` (or ``YYYY-MM-DD``) becomes ``date_int``;
    remaining numbers become the version tuple. ``claude-opus-4-6`` → version
    ``(4, 6)``; ``claude-3-5-sonnet-20241022`` → date ``20241022`` version
    ``(3, 5)``.
    """
    text = text or ""
    date_int = 0
    match = _DATE_RE.search(text)
    if match:
        date_int = int(match.group(1) + match.group(2) + match.group(3))
        text = _DATE_RE.sub(" ", text)
    version = tuple(int(n) for n in _NUM_RE.findall(text)[:4])
    return date_int, version


def _is_chat_model(model_id: str) -> bool:
    """Heuristically reject embedding/audio/image/etc. models by their id."""
    lowered = model_id.lower()
    return not any(marker in lowered for marker in _NON_CHAT_MARKERS)


def _humanize(model_id: str) -> str:
    """Derive a readable label from a raw model id.

    Best-effort only — vendor display names and the curated label map take
    precedence. Keeps version/date tokens verbatim and upper-cases known
    acronyms: ``gpt-4.1-mini`` → ``GPT 4.1 Mini``.
    """
    base = model_id.split("/")[-1]
    # Drop embedded release dates — they're noise in a label, and the picker
    # already shows the full model id alongside it.
    base = _DATE_RE.sub(" ", base)
    words: list[str] = []
    for part in re.split(r"[-_\s]+", base):
        if not part:
            continue
        low = part.lower()
        if low in _ACRONYMS:
            words.append(_ACRONYMS[low])
        elif any(ch.isdigit() for ch in part):
            words.append(part)
        else:
            words.append(part.capitalize())
    return " ".join(words) or base


# ── HTTP + parsing helpers ───────────────────────────────────────


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GET ``url`` and return parsed JSON, with a short timeout and TLS verify."""
    ssl_config = os.environ.get("SSL_CERT_FILE") or certifi.where()
    response = httpx.get(
        url,
        headers=headers,
        params=params,
        timeout=_TIMEOUT,
        verify=ssl_config,
        follow_redirects=True,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _parse_iso(value: Any) -> float:
    """Parse an ISO 8601 timestamp to an epoch float; ``0.0`` on failure."""
    if not value or not isinstance(value, str):
        return 0.0
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (OSError, json.JSONDecodeError):
        return None


# ── Caching ──────────────────────────────────────────────────────


def _cache_dir() -> Path:
    return Path.home() / ".crewai"


def _catalog_cache_file() -> Path:
    return _cache_dir() / "model_catalog_cache.json"


def _litellm_cache_file() -> Path:
    # Shared with crewai_cli.provider so both flows warm the same cache.
    return _cache_dir() / "provider_cache.json"


def _read_catalog_cache(provider_key: str) -> list[tuple[str, str]] | None:
    """Return a fresh cached catalog for ``provider_key``, or ``None``."""
    payload = _read_json(_catalog_cache_file())
    if not payload:
        return None
    entry = payload.get(provider_key)
    if not isinstance(entry, dict):
        return None
    if (time.time() - _as_float(entry.get("ts"))) >= _CATALOG_TTL:
        return None
    models = entry.get("models")
    if not isinstance(models, list) or not models:
        return None
    try:
        return [(str(m[0]), str(m[1])) for m in models]
    except (IndexError, TypeError):
        return None


def _write_catalog_cache(provider_key: str, models: list[tuple[str, str]]) -> None:
    cache_file = _catalog_cache_file()
    payload = _read_json(cache_file) or {}
    payload[provider_key] = {
        "ts": time.time(),
        "models": [[model_id, label] for model_id, label in models],
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
