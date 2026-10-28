from typing import Any

import pytest

from crewai.llms import model_catalog
from crewai.llms.model_catalog import find_model_catalog_entry


@pytest.fixture
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog: dict[str, Any] = {
        "acme-1": {"litellm_provider": "acme", "max_input_tokens": 10},
        "acme/acme-2": {"litellm_provider": "acme", "tags": ["a"]},
        "acme-3-20250101": {"litellm_provider": "acme"},
        "acme-3-20250202": {"litellm_provider": "acme"},
        "acme/acme-4-20250101": {"litellm_provider": "acme", "max_input_tokens": 4},
    }
    entries, exact, normalized = model_catalog._index_catalog(catalog)
    monkeypatch.setattr(model_catalog, "_CATALOG_ENTRIES", entries)
    monkeypatch.setattr(model_catalog, "_EXACT_ENTRIES", exact)
    monkeypatch.setattr(model_catalog, "_NORMALIZED_ENTRIES", normalized)


def test_lookup_returns_the_catalog_entry_for_an_explicit_provider() -> None:
    entry = find_model_catalog_entry("gpt-4o", provider="openai")

    assert entry is not None
    assert entry["litellm_provider"] == "openai"


def test_lookup_does_not_cross_provider_catalogs() -> None:
    entry = find_model_catalog_entry("gpt-4o", provider="azure")

    assert entry is not None
    assert entry["litellm_provider"] == "azure"


def test_lookup_uses_an_exact_qualified_catalog_key() -> None:
    entry = find_model_catalog_entry("azure/eu/gpt-4o-2024-08-06", provider="azure")

    assert entry is not None
    assert entry["litellm_provider"] == "azure"
    assert entry["max_input_tokens"] == 128_000


def test_lookup_keeps_same_named_provider_models_separate() -> None:
    openai = find_model_catalog_entry("gpt-4o-2024-05-13", provider="openai")
    azure = find_model_catalog_entry("gpt-4o-2024-05-13", provider="azure")

    assert openai is not None
    assert azure is not None
    assert openai["litellm_provider"] == "openai"
    assert azure["litellm_provider"] == "azure"


def test_lookup_accepts_crewai_provider_aliases() -> None:
    entry = find_model_catalog_entry("gemini-2.5-flash", provider="google")

    assert entry is not None
    assert entry["litellm_provider"] in {"gemini", "vertex_ai-language-models"}


def test_lookup_normalizes_a_bedrock_deployment_id() -> None:
    entry = find_model_catalog_entry(
        "us.anthropic.claude-sonnet-4-6-v1:0", provider="bedrock"
    )

    assert entry is not None
    assert entry["litellm_provider"] == "bedrock_converse"


def test_lookup_keeps_an_exact_bedrock_deployment_id() -> None:
    entry = find_model_catalog_entry(
        "us.anthropic.claude-opus-4-1-20250805-v1:0", provider="bedrock"
    )

    assert entry is not None
    assert entry["litellm_provider"] == "bedrock_converse"
    assert entry["max_input_tokens"] == 200_000


@pytest.mark.usefixtures("synthetic_catalog")
def test_lookup_leaves_an_ambiguous_normalized_match_unresolved() -> None:
    assert find_model_catalog_entry("acme-3", provider="acme") is None
    assert find_model_catalog_entry("acme-3-20250101", provider="acme") is not None


@pytest.mark.usefixtures("synthetic_catalog")
def test_lookup_resolves_a_sole_provider_prefixed_dated_entry_by_family() -> None:
    entry = find_model_catalog_entry("acme-4", provider="acme")

    assert entry is not None
    assert entry["max_input_tokens"] == 4


@pytest.mark.usefixtures("synthetic_catalog")
def test_lookup_matches_an_unknown_provider_case_insensitively() -> None:
    entry = find_model_catalog_entry("acme-1", provider="ACME")

    assert entry is not None
    assert entry["max_input_tokens"] == 10


def test_lookup_returns_none_for_an_empty_model() -> None:
    assert find_model_catalog_entry("") is None
    assert find_model_catalog_entry("", provider="openai") is None


def test_lookup_without_provider_only_accepts_an_exact_catalog_key() -> None:
    assert find_model_catalog_entry("gpt-4o") is not None
    assert find_model_catalog_entry("us.anthropic.claude-sonnet-4-6-v1:0") is None


def test_lookup_preserves_missing_catalog_properties() -> None:
    entry = find_model_catalog_entry("gpt-4", provider="openai")

    assert entry is not None
    assert "supports_vision" not in entry


def test_lookup_returns_none_for_unknown_provider_or_model() -> None:
    assert find_model_catalog_entry("gpt-4o", provider="unknown") is None
    assert find_model_catalog_entry("unknown-model", provider="openai") is None
