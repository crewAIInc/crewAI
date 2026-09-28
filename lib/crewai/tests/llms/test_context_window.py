import pytest

from crewai.llms.context_window import (
    BEDROCK_CONTEXT_WINDOWS,
    CONTEXT_WINDOW_USAGE_RATIO,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    MAX_CONTEXT,
    LLM_CONTEXT_WINDOW_SIZES,
    resolve_context_window_size,
)


def test_resolver_prefers_the_longest_matching_prefix() -> None:
    sizes = {"gpt-5": 1_047_576, "gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-mini-2026-01-01", sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_uses_default_for_an_unknown_model() -> None:
    result = resolve_context_window_size(
        "unknown-model", {}, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_validates_all_declared_context_windows() -> None:
    with pytest.raises(ValueError, match="must be between 1024 and 10000000"):
        resolve_context_window_size("test-model", {"test-model": 500}, default=8192)


def test_bedrock_claude_aliases_use_the_anthropic_context_window() -> None:
    result = resolve_context_window_size(
        "us.anthropic.claude-sonnet-4-6-v1:0",
        BEDROCK_CONTEXT_WINDOWS,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(1_000_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bedrock_regional_aliases_preserve_the_base_model_context_window() -> None:
    result = resolve_context_window_size(
        "us.meta.llama3-3-70b-instruct-v1:0",
        BEDROCK_CONTEXT_WINDOWS,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(128_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_litellm_map_includes_the_openai_gpt5_family() -> None:
    result = resolve_context_window_size(
        "gpt-5", LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(400_000 * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "raw_context_window"),
    [
        ("o1", 200_000),
        ("o1-pro", 200_000),
        ("o3", 200_000),
        ("gemini-3.1-flash-lite", 1_048_576),
        ("gemini-3.1-pro-preview", 1_048_576),
        ("gemini-3-flash-preview", 1_048_576),
    ],
)
def test_affected_model_ids_use_their_specific_context_windows(
    model: str, raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "sizes", "raw_context_window"),
    [
        ("gpt-6-astra", LLM_CONTEXT_WINDOW_SIZES, 1_050_000),
        ("gpt-5.4-nano", LLM_CONTEXT_WINDOW_SIZES, 400_000),
        ("gemini-3.7-flash", LLM_CONTEXT_WINDOW_SIZES, 1_048_576),
        ("amazon.nova-2-lite-v1:0", BEDROCK_CONTEXT_WINDOWS, 1_000_000),
        (
            "meta.llama4-scout-17b-instruct-v1:0",
            BEDROCK_CONTEXT_WINDOWS,
            MAX_CONTEXT,
        ),
    ],
)
def test_new_catalog_models_use_their_documented_context_windows(
    model: str, sizes: dict[str, int], raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)
