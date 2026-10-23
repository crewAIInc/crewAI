import pytest

from crewai.llms.context_window import (
    ANTHROPIC_CONTEXT_WINDOWS,
    BEDROCK_CONTEXT_WINDOWS,
    GEMINI_CONTEXT_WINDOWS,
    CONTEXT_WINDOW_USAGE_RATIO,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    LLM_CONTEXT_WINDOW_SIZES,
    bare_context_windows,
    context_windows_from_catalog,
    resolve_context_window_size,
)


def test_resolver_uses_the_exact_model_id() -> None:
    sizes = {"gpt-5": 1_047_576, "gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-mini", sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bare_lookup_uses_the_full_catalog_when_the_provider_map_misses() -> None:
    result = resolve_context_window_size(
        "us.anthropic.claude-sonnet-4-5-v1:0",
        {},
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_unrecognized_provider_prefix_does_not_inherit_the_bare_window() -> None:
    result = resolve_context_window_size(
        "acme/gpt-5.6-luna",
        LLM_CONTEXT_WINDOW_SIZES,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_does_not_match_a_different_model_family() -> None:
    sizes = {"gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-nano", sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(272_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_uses_default_for_an_unknown_model() -> None:
    result = resolve_context_window_size(
        "unknown-model", {}, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_uses_a_published_limit_as_given() -> None:
    result = resolve_context_window_size(
        "test-model", {"test-model": 500}, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(500 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bedrock_claude_uses_its_documented_context_window() -> None:
    result = resolve_context_window_size(
        "us.anthropic.claude-sonnet-4-6",
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

    assert result == int(272_000 * CONTEXT_WINDOW_USAGE_RATIO)


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
        ("gpt-6-astra", LLM_CONTEXT_WINDOW_SIZES, 922_000),
        ("gpt-5.6-cyber", LLM_CONTEXT_WINDOW_SIZES, 400_000),
        ("gpt-3.5-turbo", LLM_CONTEXT_WINDOW_SIZES, 16_385),
        ("chat-latest", LLM_CONTEXT_WINDOW_SIZES, 400_000),
        ("azure_ai/gpt-oss-120b", LLM_CONTEXT_WINDOW_SIZES, 131_072),
        ("gpt-5.4-nano", LLM_CONTEXT_WINDOW_SIZES, 272_000),
        ("gemini-3.7-flash", LLM_CONTEXT_WINDOW_SIZES, 1_048_576),
        ("gemma-3-27b-it", GEMINI_CONTEXT_WINDOWS, 131_072),
        ("gemma-4-31b-it", GEMINI_CONTEXT_WINDOWS, 262_144),
        ("amazon.nova-2-lite-v1:0", BEDROCK_CONTEXT_WINDOWS, 1_000_000),
        ("global.moonshotai.kimi-k3", BEDROCK_CONTEXT_WINDOWS, 1_000_000),
        # 200k: LiteLLM's catalog had 1M here until its refresh for 1.15.28
        # corrected it to the model's documented window.
        (
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            BEDROCK_CONTEXT_WINDOWS,
            200_000,
        ),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", BEDROCK_CONTEXT_WINDOWS, 200_000),
        (
            "meta.llama4-scout-17b-instruct-v1:0",
            BEDROCK_CONTEXT_WINDOWS,
            10_000_000,
        ),
    ],
)
def test_new_catalog_models_use_their_documented_context_windows(
    model: str, sizes: dict[str, int], raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


def test_packaged_litellm_catalog_loads_known_models() -> None:
    assert LLM_CONTEXT_WINDOW_SIZES["gpt-4"] == 8192
    assert ANTHROPIC_CONTEXT_WINDOWS["claude-sonnet-4-6"] == 1_000_000
    assert len(LLM_CONTEXT_WINDOW_SIZES) > 1000


def test_catalog_parser_keeps_published_integer_limits() -> None:
    windows = context_windows_from_catalog(
        {
            "sample_spec": {"max_input_tokens": "max input tokens"},
            "too-small": {"litellm_provider": "openai", "max_input_tokens": 77},
            "not-a-number": {
                "litellm_provider": "openai",
                "max_input_tokens": "8192",
            },
            "huge": {
                "litellm_provider": "openai",
                "max_input_tokens": 20_000_000,
            },
            "openai/gpt-example": {
                "litellm_provider": "openai",
                "max_input_tokens": 16_000,
            },
        }
    )

    assert "sample_spec" not in windows
    assert windows["too-small"] == 77
    assert "not-a-number" not in windows
    assert windows["huge"] == 20_000_000
    assert windows["openai/gpt-example"] == 16_000
    assert windows["gpt-example"] == 16_000


def test_contested_provider_aliases_are_left_to_the_bare_index() -> None:
    windows = context_windows_from_catalog(
        {
            "azure/contested-model": {
                "litellm_provider": "azure",
                "max_input_tokens": 32_000,
            },
            "azure_ai/contested-model": {
                "litellm_provider": "azure_ai",
                "max_input_tokens": 128_000,
            },
        }
    )

    assert windows["azure/contested-model"] == 32_000
    assert windows["azure_ai/contested-model"] == 128_000
    assert "contested-model" not in windows
    assert resolve_context_window_size(
        "contested-model", windows, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "raw_context_window"),
    [
        ("us.anthropic.claude-sonnet-4-5-v1:0", 200_000),
        ("eu.anthropic.claude-sonnet-4-5-v1:0", 200_000),
        ("us.anthropic.claude-opus-4-6-v1:0", 1_000_000),
        ("us.anthropic.claude-opus-4-5-v1:0", 200_000),
        ("us.anthropic.claude-haiku-4-5-v1:0", 200_000),
        ("amazon.nova-lite-v1:0:24k", 300_000),
        ("anthropic.claude-v2:1:200k", 100_000),
    ],
)
def test_bare_model_name_matches_catalog_variants(
    model: str, raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, BEDROCK_CONTEXT_WINDOWS, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


def test_bare_context_windows_uses_the_majority_limit() -> None:
    windows = bare_context_windows(
        {
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": 1_000_000,
        }
    )

    assert windows["claude-sonnet-4-5"] == 200_000


def test_bare_context_windows_uses_the_smaller_limit_when_votes_tie() -> None:
    windows = bare_context_windows(
        {
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 1_000_000,
        }
    )

    assert windows["claude-sonnet-4-5"] == 200_000
