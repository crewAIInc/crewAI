"""Unit tests for context-window resolution.

Covers the refactor where one module (``crewai.llms.context_window``) owns the
raw window sizes and the longest-prefix lookup, while ``crewai.llm`` re-exports
the aggregate maps. The provider completion files resolve through the shared
``resolve_context_window_size`` helper instead of duplicating the lists.
"""

import pytest

from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO, DEFAULT_CONTEXT_WINDOW_SIZE, LLM
from crewai.llms.context_window import (
    ANTHROPIC_CONTEXT_WINDOWS,
    GEMINI_CONTEXT_WINDOWS,
    LITELLM_CONTEXT_WINDOWS,
    MAX_CONTEXT_WINDOW_SIZE,
    MIN_CONTEXT_WINDOW_SIZE,
    OPENAI_CONTEXT_WINDOWS,
    resolve_context_window_size,
)

RATIO = CONTEXT_WINDOW_USAGE_RATIO


# --------------------------------------------------------------------------
# crewai.llm still exports the aggregate symbols
# --------------------------------------------------------------------------


def test_context_window_symbols_are_exported_from_crewai_llm():
    """``crewai.llm`` keeps exporting the legacy aggregate constants."""
    from crewai.llm import (
        DEFAULT_CONTEXT_WINDOW_SIZE as exported_default,
        LLM_CONTEXT_WINDOW_SIZES,
    )

    assert isinstance(LLM_CONTEXT_WINDOW_SIZES, dict)
    assert LLM_CONTEXT_WINDOW_SIZES  # non-empty
    assert 0 < CONTEXT_WINDOW_USAGE_RATIO < 1
    assert exported_default == DEFAULT_CONTEXT_WINDOW_SIZE == 8192


# --------------------------------------------------------------------------
# resolve_context_window_size: longest-prefix matching, bounds, default
# --------------------------------------------------------------------------


def test_longest_prefix_wins_over_shorter_prefix():
    context_windows = {
        "gpt-5": 1_047_576,
        "gpt-5.4-mini": 200_000,
    }
    result = resolve_context_window_size(
        "gpt-5.4-mini-2026-01-01", context_windows, default=8192
    )
    assert result == int(200_000 * RATIO)


def test_longest_prefix_match_is_order_independent():
    """Lookup must not depend on dict insertion order (the old ``startswith`` loop did)."""
    context_windows = {
        "gpt-5": 1_047_576,
        "gpt-5.4-mini": 200_000,
    }
    reversed_windows = {
        "gpt-5.4-mini": 200_000,
        "gpt-5": 1_047_576,
    }
    assert resolve_context_window_size(
        "gpt-5.4-mini", reversed_windows, default=8192
    ) == resolve_context_window_size("gpt-5.4-mini", context_windows, default=8192)
    assert resolve_context_window_size(
        "gpt-5.4-mini", reversed_windows, default=8192
    ) == int(200_000 * RATIO)


def test_shortest_prefix_still_matches_when_nothing_longer_does():
    context_windows = {"gpt-5": 1_047_576, "gpt-5.4-mini": 200_000}
    result = resolve_context_window_size(
        "gpt-5-2026-01-01", context_windows, default=8192
    )
    assert result == int(1_047_576 * RATIO)


def test_unknown_model_uses_default():
    result = resolve_context_window_size(
        "some-unknown-model", OPENAI_CONTEXT_WINDOWS, default=8192
    )
    assert result == int(8192 * RATIO)


def test_default_applied_only_when_no_prefix_matches():
    context_windows = {"gpt-4": 8192}
    # "gpt-4o-mini" starts with "gpt-4" -- a valid prefix match must win
    # even when the default is much larger than the matched window.
    result = resolve_context_window_size(
        "gpt-4o-mini", context_windows, default=1_000_000
    )
    assert result == int(8192 * RATIO)


def test_out_of_bounds_default_raises():
    with pytest.raises(ValueError, match="must be between"):
        resolve_context_window_size(
            "gpt-4", OPENAI_CONTEXT_WINDOWS, default=MAX_CONTEXT_WINDOW_SIZE + 1
        )
    with pytest.raises(ValueError, match="must be between"):
        resolve_context_window_size(
            "gpt-4", OPENAI_CONTEXT_WINDOWS, default=MIN_CONTEXT_WINDOW_SIZE - 1
        )


def test_out_of_bounds_context_window_entry_raises():
    with pytest.raises(ValueError, match="must be between"):
        resolve_context_window_size(
            "test-model", {"test-model": MAX_CONTEXT_WINDOW_SIZE + 1}, default=8192
        )
    with pytest.raises(ValueError, match="must be between"):
        resolve_context_window_size(
            "test-model", {"test-model": MIN_CONTEXT_WINDOW_SIZE - 1}, default=8192
        )


def test_usage_ratio_is_applied_to_match_and_default():
    context_windows = {"test-model": 10000}
    assert resolve_context_window_size(
        "test-model", context_windows, default=8192
    ) == int(10000 * RATIO)
    assert resolve_context_window_size("nope", context_windows, default=8192) == int(
        8192 * RATIO
    )


# --------------------------------------------------------------------------
# Family maps keep raw sizes inside valid bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "windows",
    [
        OPENAI_CONTEXT_WINDOWS,
        LITELLM_CONTEXT_WINDOWS,
        ANTHROPIC_CONTEXT_WINDOWS,
        GEMINI_CONTEXT_WINDOWS,
    ],
)
def test_all_raw_window_sizes_are_in_bounds(windows):
    for prefix, size in windows.items():
        assert MIN_CONTEXT_WINDOW_SIZE <= size <= MAX_CONTEXT_WINDOW_SIZE, prefix


# --------------------------------------------------------------------------
# Parity: the same family id resolves identically via the native provider
# and through the LiteLLM fallback (is_litellm=True)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected_raw"),
    [
        ("gpt-5", 1047576),
        ("claude-sonnet-4-6", 1000000),
        ("gemini-2.0-flash", 1048576),
    ],
)
def test_context_window_parity_native_vs_litellm(model, expected_raw):
    native = LLM(model=model)
    via_litellm = LLM(model=model, is_litellm=True)

    expected = int(expected_raw * RATIO)
    assert native.get_context_window_size() == expected
    assert via_litellm.get_context_window_size() == expected


# --------------------------------------------------------------------------
# Specific acceptance scenarios
# --------------------------------------------------------------------------


def test_litellm_gpt5_uses_native_window_not_8192():
    """LiteLLM gpt-5 must resolve to the native 1,047,576 window, not 8192."""
    llm = LLM(model="gpt-5", is_litellm=True)
    assert llm.get_context_window_size() == int(1047576 * RATIO)


def test_litellm_gpt54_mini_stays_200k():
    """A more specific family must not inherit the gpt-5 window."""
    llm = LLM(model="gpt-5.4-mini", is_litellm=True)
    assert llm.get_context_window_size() == int(200000 * RATIO)


def test_provider_prefixed_litellm_gpt5_uses_native_window():
    """LiteLLM keeps the ``openai/`` prefix on the model; lookup must still hit gpt-5."""
    llm = LLM(model="openai/gpt-5", is_litellm=True)
    assert llm._context_window_model_name() == "gpt-5"
    assert llm.get_context_window_size() == int(1047576 * RATIO)


def test_bedrock_anthropic_claude_sonnet_4_6_is_1m_not_200k():
    """Bedrock Claude Sonnet 4.6 is a 1M-window model (native path)."""
    llm = LLM(model="bedrock/anthropic.claude-sonnet-4-6")
    assert llm.get_context_window_size() == int(1000000 * RATIO)

def test_azure_gpt_35_turbo_litellm_is_16385():
    llm = LLM(model="azure/gpt-35-turbo", is_litellm=True)
    assert llm.get_context_window_size() == int(16385 * RATIO)

# issue #7303: o1/o1-pro/o3 must resolve to the official 200k window.
def test_o_series_reasoning_models():
    """o1/o1-pro/o3 use 200k (native + LiteLLM); o1-preview/o1-mini keep 128k."""
    for model in ("o1", "o1-pro", "o3"):
        native = LLM(model=model)
        via_litellm = LLM(model=model, is_litellm=True)
        expected = int(200000 * RATIO)
        assert native.get_context_window_size() == expected
        assert via_litellm.get_context_window_size() == expected

    for model in ("o1-preview", "o1-mini"):
        native = LLM(model=model)
        via_litellm = LLM(model=model, is_litellm=True)
        expected = int(128000 * RATIO)
        assert native.get_context_window_size() == expected
        assert via_litellm.get_context_window_size() == expected
