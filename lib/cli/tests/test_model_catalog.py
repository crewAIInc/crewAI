"""Tests for the dynamic model catalog used by the crew-creation wizard."""

from __future__ import annotations

import json
import time

import pytest

import crewai_cli.model_catalog as mc

_ALL_KEY_ENVS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "OLLAMA_API_BASE",
    "API_BASE",
    "OLLAMA_HOST",
]

FALLBACK_ANTHROPIC = [
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Point the cache at a temp dir and clear provider keys for every test."""
    monkeypatch.setattr(mc, "_cache_dir", lambda: tmp_path)
    mc._reset_litellm_memo()  # clear the process-level LiteLLM memo per test
    for key in _ALL_KEY_ENVS:
        monkeypatch.delenv(key, raising=False)


# ── version / label helpers ──────────────────────────────────────


def test_version_key_parses_embedded_date():
    date_int, version = mc._version_key("claude-3-5-sonnet-20241022")
    assert date_int == 20241022
    assert version == (3, 5)


def test_version_key_parses_dashed_date():
    date_int, _ = mc._version_key("gpt-4o-2024-08-06")
    assert date_int == 20240806


def test_version_key_version_only():
    date_int, version = mc._version_key("claude-opus-4-6")
    assert date_int == 0
    assert version == (4, 6)


def test_version_key_ranks_newer_higher():
    older = mc._version_key("claude-sonnet-4-5")
    newer = mc._version_key("claude-sonnet-4-6")
    assert newer > older


def test_is_chat_model_rejects_non_chat():
    assert mc._is_chat_model("gpt-4.1-mini")
    assert not mc._is_chat_model("text-embedding-3-large")
    assert not mc._is_chat_model("whisper-1")
    assert not mc._is_chat_model("dall-e-3")


def test_search_substring_not_treated_as_non_chat():
    # 'search' must not drop legitimate completion models: a token like
    # *-search-preview, or 'research' (which contains 'search' as a substring).
    assert mc._is_chat_model("gpt-4o-search-preview")
    assert mc._is_chat_model("o3-deep-research")
    # genuine non-chat markers still filter
    assert not mc._is_chat_model("text-embedding-3-large")


def test_humanize():
    assert mc._humanize("gpt-4.1-mini") == "GPT 4.1 Mini"
    assert mc._humanize("anthropic/claude-opus-4-6") == "Claude Opus 4 6"
    # size suffixes uppercased, acronyms/brands cased, o-series preserved, ':' split
    assert mc._humanize("openai/gpt-oss-120b") == "GPT OSS 120B"
    assert mc._humanize("qwen/qwen3-32b") == "Qwen3 32B"
    assert mc._humanize("deepseek-r1-distill-llama-70b") == "DeepSeek R1 Distill Llama 70B"
    assert mc._humanize("o3-mini") == "o3 Mini"
    assert mc._humanize("chatgpt-4o-latest") == "ChatGPT 4o Latest"
    assert mc._humanize("llama3.3:70b") == "Llama3.3 70B"
    assert mc._humanize("gemma2-9b-it") == "Gemma2 9B IT"


# ── vendor tier ──────────────────────────────────────────────────


def test_vendor_anthropic_ranks_by_date_and_uses_display_name(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    payload = {
        "data": [
            {
                "id": "claude-3-5-sonnet-20240620",
                "display_name": "Claude 3.5 Sonnet (old)",
                "created_at": "2024-06-20T00:00:00Z",
            },
            {
                "id": "claude-opus-4-6",
                "display_name": "Claude Opus 4.6",
                "created_at": "2026-02-01T00:00:00Z",
            },
            {
                "id": "claude-haiku-4-5-20251001",
                "display_name": "Claude Haiku 4.5",
                "created_at": "2025-10-01T00:00:00Z",
            },
        ]
    }
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    models = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)

    # Newest first by created_at, display names preserved.
    assert models[0] == ("claude-opus-4-6", "Claude Opus 4.6")
    assert models[1] == ("claude-haiku-4-5-20251001", "Claude Haiku 4.5")
    assert models[2] == ("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet (old)")


def test_vendor_openai_filters_non_chat_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = {
        "data": [
            {"id": "gpt-4.1", "created": 1_700_000_000},
            {"id": "text-embedding-3-large", "created": 1_800_000_000},
            {"id": "whisper-1", "created": 1_800_000_000},
            {"id": "gpt-5.5", "created": 1_750_000_000},
        ]
    }
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    models = mc.get_provider_models("openai", [])
    ids = [m for m, _ in models]

    assert ids == ["gpt-5.5", "gpt-4.1"]  # embeddings/whisper dropped, newest first


def test_vendor_gemini_requires_generate_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    payload = {
        "models": [
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/text-embedding-004",
                "displayName": "Embedding",
                "supportedGenerationMethods": ["embedContent"],
            },
            {
                "name": "models/gemini-1.5-pro",
                "displayName": "Gemini 1.5 Pro",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    models = mc.get_provider_models("gemini", [])
    ids = [m for m, _ in models]

    # "models/" prefix stripped, embedding excluded, newer version first.
    assert ids == ["gemini-2.5-pro", "gemini-1.5-pro"]


def test_openai_excludes_fine_tunes_and_checkpoints(monkeypatch):
    # Fine-tunes/checkpoints have recent `created` timestamps and would otherwise
    # crowd out (and rank above) the base models — they must be excluded so the
    # picker shows clean foundation models.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = {
        "data": [
            {"id": "ft:gpt-4o-mini-2024-07-18:crewai::DyJG86uF", "created": 1_900_000_000},
            {
                "id": "ft:gpt-4o-mini-2024-07-18:crewai::DyJG7Q9N:ckpt-step-84",
                "created": 1_900_000_001,
            },
            {"id": "gpt-5.5", "created": 1_800_000_000},
            {"id": "gpt-4.1", "created": 1_700_000_000},
        ]
    }
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    ids = [m for m, _ in mc.get_provider_models("openai", [])]
    assert ids == ["gpt-5.5", "gpt-4.1"]  # fine-tunes + checkpoints dropped


def test_vendor_gemini_paginates(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    pages = {
        None: {
            "models": [
                {
                    "name": "models/gemini-3.5-flash",
                    "displayName": "Gemini 3.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "p2",
        },
        "p2": {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        },
    }

    def fetch(url, headers=None, params=None):
        return pages[(params or {}).get("pageToken")]

    monkeypatch.setattr(mc, "_http_get_json", fetch)

    ids = sorted(m for m, _ in mc.get_provider_models("gemini", []))
    # Both pages contributed (newest-first ranking is _finalize's job).
    assert ids == ["gemini-2.5-pro", "gemini-3.5-flash"]


def test_vendor_gemini_first_page_error_uses_fallback(monkeypatch):
    # A total (first-page) Gemini failure with a key set must fall back to the
    # curated list, not be mistaken for a successful empty result.
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    def boom(*a, **k):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(mc, "_http_get_json", boom)
    models = mc.get_provider_models("gemini", [("gemini-x", "Gemini X")])
    assert models == [("gemini-x", "Gemini X")]


def test_vendor_gemini_keeps_partial_on_later_page_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    def fetch(url, headers=None, params=None):
        if (params or {}).get("pageToken"):
            raise RuntimeError("page 2 down")
        return {
            "models": [
                {
                    "name": "models/gemini-3.5-flash",
                    "displayName": "Gemini 3.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "p2",
        }

    monkeypatch.setattr(mc, "_http_get_json", fetch)

    # Page-1 models are kept; the later-page error doesn't force the fallback.
    models = mc.get_provider_models("gemini", [("fallback-x", "Fallback X")])
    assert [m for m, _ in models] == ["gemini-3.5-flash"]


def test_ollama_empty_response_not_filled_with_fallback(monkeypatch):
    # A reachable Ollama with nothing installed -> empty (manual entry), not the
    # curated suggestions the crew can't actually run.
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: {"models": []})
    assert mc.get_provider_models("ollama", [("llama3.3", "Llama 3.3")]) == []


def test_ollama_unreachable_uses_fallback(monkeypatch):
    # Server down (fetch raises) is different from empty -> fall back to suggestions.
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mc, "_http_get_json", boom)
    models = mc.get_provider_models("ollama", [("llama3.3", "Llama 3.3")])
    assert models == [("llama3.3", "Llama 3.3")]


def test_ollama_excludes_embedding_models(monkeypatch):
    # /api/tags lists everything installed, including embeddings — filter them.
    monkeypatch.setattr(
        mc,
        "_http_get_json",
        lambda *a, **k: {
            "models": [
                {"model": "llama3.3:70b"},
                {"model": "nomic-embed-text"},
                {"model": "mxbai-embed-large"},
            ]
        },
    )
    ids = [m for m, _ in mc.get_provider_models("ollama", [])]
    assert ids == ["llama3.3:70b"]


def test_ollama_base_honors_ollama_host(monkeypatch):
    # OLLAMA_HOST (scheme-less runtime convention) is resolved with a scheme.
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    assert mc._ollama_base() == "http://10.0.0.5:11434"


def test_ollama_recovery_not_blocked_by_negative_cache(monkeypatch):
    # Ollama down -> fallback, but not negatively cached; once the server is up
    # the next call fetches live models rather than serving suggestions.
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection refused")
        return {"models": [{"model": "llama-installed"}]}

    monkeypatch.setattr(mc, "_http_get_json", flaky)
    first = mc.get_provider_models("ollama", [("llama3.3", "Llama 3.3")])
    assert first == [("llama3.3", "Llama 3.3")]  # down -> fallback (not cached)
    second = mc.get_provider_models("ollama", [("llama3.3", "Llama 3.3")])
    assert [m for m, _ in second] == ["llama-installed"]  # recovered live


def test_gemini_honors_google_api_key(monkeypatch):
    # GOOGLE_API_KEY (equivalent to GEMINI_API_KEY in crewai) enables the live tier.
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(
        mc,
        "_http_get_json",
        lambda *a, **k: {
            "models": [
                {
                    "name": "models/gemini-3.5-flash",
                    "displayName": "Gemini 3.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        },
    )
    models = mc.get_provider_models("gemini", [("gemini-x", "Gemini X")])
    assert [m for m, _ in models] == ["gemini-3.5-flash"]  # live, not fallback


def test_curated_label_overrides_raw_vendor_label(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = {"data": [{"id": "gpt-5.5", "created": 1}]}
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    models = mc.get_provider_models("openai", [("gpt-5.5", "GPT-5.5 (curated)")])
    assert models == [("gpt-5.5", "GPT-5.5 (curated)")]


def test_truncates_to_max_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    payload = {
        "data": [{"id": f"gpt-test-{i}", "created": i} for i in range(20)]
    }
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: payload)

    models = mc.get_provider_models("openai", [])
    assert len(models) == mc.MAX_MODELS


# ── litellm tier ─────────────────────────────────────────────────


def test_litellm_tier_for_uncurated_provider(monkeypatch):
    # A provider with no curated fallback ([]) -> the LiteLLM feed is consulted.
    litellm_data = {
        "claude-opus-4-6": {"litellm_provider": "anthropic", "mode": "chat"},
        "claude-sonnet-4-5": {"litellm_provider": "anthropic", "mode": "chat"},
        "voyage-embed": {"litellm_provider": "anthropic", "mode": "embedding"},
        "gpt-4.1": {"litellm_provider": "openai", "mode": "chat"},
    }
    mc._litellm_cache_file().write_text(json.dumps(litellm_data), encoding="utf-8")

    models = mc.get_provider_models("anthropic", [])  # empty == uncurated
    ids = [m for m, _ in models]

    # Only anthropic chat models, embedding + other providers excluded.
    assert ids == ["claude-opus-4-6", "claude-sonnet-4-5"]


def test_null_litellm_provider_does_not_crash(monkeypatch):
    # A present-but-null litellm_provider must be skipped, not raise.
    litellm_data = {
        "weird-model": {"litellm_provider": None, "mode": "chat"},
        "anthropic.claude-v2": {"litellm_provider": "bedrock", "mode": "chat"},
    }
    mc._litellm_cache_file().write_text(json.dumps(litellm_data), encoding="utf-8")

    models = mc.get_provider_models("bedrock", [])
    assert [m for m, _ in models] == ["anthropic.claude-v2"]


def test_litellm_strips_provider_prefix(monkeypatch):
    litellm_data = {
        "gemini/gemini-1.5-pro": {"litellm_provider": "gemini", "mode": "chat"},
    }
    mc._litellm_cache_file().write_text(json.dumps(litellm_data), encoding="utf-8")

    models = mc.get_provider_models("gemini", [])
    assert models == [("gemini-1.5-pro", "Gemini 1.5 Pro")]


# ── fallback + caching ───────────────────────────────────────────


def test_falls_back_when_everything_fails(monkeypatch):
    # No key, no litellm cache, network raises -> curated fallback verbatim.
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mc, "_http_get_json", boom)
    models = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)
    assert models == FALLBACK_ANTHROPIC


def test_result_is_cached(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = {"n": 0}

    def fetch(*a, **k):
        calls["n"] += 1
        return {"data": [{"id": "claude-opus-4-6", "created_at": "2026-01-01T00:00:00Z"}]}

    monkeypatch.setattr(mc, "_http_get_json", fetch)

    first = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)
    # Second call must hit the cache and not touch the network again.
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: pytest.fail("refetched"))
    second = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)

    assert first == second
    assert calls["n"] == 1


def test_curated_fallback_preferred_over_litellm(monkeypatch):
    # The feed lags real releases, so a non-empty curated fallback must win even
    # when a fresh LiteLLM cache is present (regression: Anthropic's feed lacked
    # Fable 5 / Opus 4.8 / Sonnet 5).
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: pytest.fail("no net"))
    litellm_data = {
        "claude-opus-4-6": {"litellm_provider": "anthropic", "mode": "chat"},
    }
    mc._litellm_cache_file().write_text(json.dumps(litellm_data), encoding="utf-8")

    models = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)
    assert models == FALLBACK_ANTHROPIC


def test_added_key_bypasses_negative_cache(monkeypatch):
    # A no-key call negatively-caches the fallback; adding a key afterwards must
    # fetch live models rather than serve the cached fallback (distinct cache key).
    first = mc.get_provider_models("openai", [("gpt-x", "GPT X")])
    assert first == [("gpt-x", "GPT X")]  # no key -> fallback

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        mc, "_http_get_json", lambda *a, **k: {"data": [{"id": "gpt-5.5", "created": 1}]}
    )
    second = mc.get_provider_models("openai", [("gpt-x", "GPT X")])
    assert [m for m, _ in second] == ["gpt-5.5"]  # live fetch, not cached fallback


def test_invalid_litellm_cache_falls_through_to_download(monkeypatch):
    # A corrupt-but-fresh cache must neither crash the picker nor block a
    # recoverable download — it falls through and refetches.
    mc._litellm_cache_file().write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(
        mc,
        "_http_get_json",
        lambda *a, **k: {
            "anthropic.claude-v2": {"litellm_provider": "bedrock", "mode": "chat"}
        },
    )
    models = mc.get_provider_models("bedrock", [])
    assert [m for m, _ in models] == ["anthropic.claude-v2"]  # recovered via download


def test_litellm_fetch_attempted_once_per_process(monkeypatch):
    # With no cache and a failing download, the feed is fetched at most once per
    # process — repeated lookups (across providers) must not re-hit the network.
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("offline")

    monkeypatch.setattr(mc, "_http_get_json", boom)
    mc.get_provider_models("bedrock", [])
    mc.get_provider_models("azure", [])
    assert calls["n"] == 1  # memoized after the first failed attempt


def test_litellm_fills_uncurated_bedrock(monkeypatch):
    # No vendor fetcher and no curated fallback -> LiteLLM feed fills the gap.
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: pytest.fail("no net"))
    litellm_data = {
        "anthropic.claude-v2": {"litellm_provider": "bedrock", "mode": "chat"},
    }
    mc._litellm_cache_file().write_text(json.dumps(litellm_data), encoding="utf-8")

    models = mc.get_provider_models("bedrock", [])
    assert models == [("anthropic.claude-v2", "Anthropic.claude V2")]


def test_failed_fetch_is_negatively_cached(monkeypatch):
    # A failed vendor fetch must not be retried on every call — the fallback is
    # cached briefly so the picker doesn't re-hit the timeout-prone endpoint.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("down")

    monkeypatch.setattr(mc, "_http_get_json", boom)
    first = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)
    second = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)

    assert first == second == FALLBACK_ANTHROPIC
    assert calls["n"] == 1  # second call served from the negative cache


def test_bad_cache_json_does_not_crash(monkeypatch):
    # A corrupt cache whose root is not a mapping must not raise (get_provider_models
    # is documented to never raise).
    mc._catalog_cache_file().write_text("[1, 2, 3]", encoding="utf-8")

    models = mc.get_provider_models("anthropic", FALLBACK_ANTHROPIC)
    assert models == FALLBACK_ANTHROPIC


def test_ollama_is_not_cached_reflects_installed_changes(monkeypatch):
    # Ollama is local and never cached: the picker re-probes /api/tags on every
    # call, so a model deleted locally drops out immediately (no stale entry).
    responses = iter(
        [
            {"models": [{"model": "llama3.3"}, {"model": "qwen3"}]},
            {"models": [{"model": "llama3.3"}]},  # qwen3 deleted between calls
        ]
    )
    monkeypatch.setattr(mc, "_http_get_json", lambda *a, **k: next(responses))

    first = {m for m, _ in mc.get_provider_models("ollama", [])}
    second = {m for m, _ in mc.get_provider_models("ollama", [])}

    assert first == {"llama3.3", "qwen3"}
    assert second == {"llama3.3"}  # re-probed, not served from a stale cache


def test_ollama_never_written_to_catalog_cache(monkeypatch):
    monkeypatch.setattr(
        mc, "_http_get_json", lambda *a, **k: {"models": [{"model": "llama3.3"}]}
    )
    mc.get_provider_models("ollama", [])
    assert not mc._catalog_cache_file().exists()


def test_different_api_key_uses_separate_cache_entry(monkeypatch):
    # Cache is keyed by the exact key: switching keys must refetch, not serve
    # the previous account's cached list.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-account-A")
    monkeypatch.setattr(
        mc, "_http_get_json", lambda *a, **k: {"data": [{"id": "gpt-5.5", "created": 1}]}
    )
    assert [m for m, _ in mc.get_provider_models("openai", [])] == ["gpt-5.5"]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-account-B")
    monkeypatch.setattr(
        mc, "_http_get_json", lambda *a, **k: {"data": [{"id": "gpt-4.1", "created": 1}]}
    )
    # New key -> distinct cache entry -> refetch, not the account-A cache.
    assert [m for m, _ in mc.get_provider_models("openai", [])] == ["gpt-4.1"]


def test_cache_key_hashes_key_and_never_stores_it(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    key = mc._cache_key("openai")
    assert key.startswith("openai#") and key != "openai#nokey"
    assert "sk-super-secret" not in key  # only a digest, never the raw key


def test_dynamic_cache_expires_after_catalog_ttl(monkeypatch):
    # A dynamic entry older than the (now short) catalog TTL is not served.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    entry_key = mc._cache_key("anthropic")
    stale_ts = time.time() - (mc._CATALOG_TTL + 5)
    mc._catalog_cache_file().write_text(
        json.dumps(
            {
                entry_key: {
                    "ts": stale_ts,
                    "source": "dynamic",
                    "models": [["stale-model", "Stale"]],
                }
            }
        ),
        encoding="utf-8",
    )

    assert mc._read_catalog_cache("anthropic") is None
