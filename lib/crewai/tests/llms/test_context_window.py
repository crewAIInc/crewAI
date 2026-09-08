import pytest

from crewai.llms.context_window import (
    CONTEXT_WINDOW_USAGE_RATIO,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    MAX_CONTEXT,
    MIN_CONTEXT,
    resolve_context_window_size,
)


def test_resolve_context_window_size_exact_match():
    sizes = {"gpt-4": 8192, "gpt-4-turbo": 128000}
    result = resolve_context_window_size("gpt-4-turbo", sizes)
    assert result == int(128000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolve_context_window_size_longest_prefix():
    sizes = {"gpt-4": 8192, "gpt-4-turbo": 128000}
    # gpt-4-turbo-2024-04-09 should match gpt-4-turbo, not gpt-4
    result = resolve_context_window_size("gpt-4-turbo-2024-04-09", sizes)
    assert result == int(128000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolve_context_window_size_default():
    sizes = {"gpt-4": 8192}
    result = resolve_context_window_size("unknown-model", sizes, default=20000)
    assert result == int(20000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolve_context_window_size_extra_names():
    sizes = {"gpt-4": 8192}
    # Matches via extra name
    result = resolve_context_window_size(
        "openai/gpt-4-0613", sizes, extra_names=("gpt-4-0613",)
    )
    assert result == int(8192 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolve_context_window_size_bounds():
    sizes = {"too-small": 100, "too-big": 5000000}
    
    result_small = resolve_context_window_size("too-small", sizes)
    assert result_small == int(MIN_CONTEXT * CONTEXT_WINDOW_USAGE_RATIO)
    
    result_big = resolve_context_window_size("too-big", sizes)
    assert result_big == int(MAX_CONTEXT * CONTEXT_WINDOW_USAGE_RATIO)

def test_resolve_context_window_size_gpt5_regression():
    """Ensure bare gpt-5 resolves to its intended context window size."""
    from crewai.llms.context_window import OPENAI_CONTEXT_WINDOWS
    result = resolve_context_window_size("gpt-5", OPENAI_CONTEXT_WINDOWS)
    assert result == int(1047576 * CONTEXT_WINDOW_USAGE_RATIO)


def test_registry_expected_model_sizes():
    """Test that the centralized resolver correctly matches expected context window sizes in the registry for key models."""
    from crewai.llms.context_window import LLM_CONTEXT_WINDOW_SIZES
    
    # Native explicit overrides or matching values
    expected_parities = {
        "gpt-5": 1047576,
        "claude-sonnet-4-6": 1000000,
        "gemini-2.0-flash": 1048576,
    }
    
    for model, expected_size in expected_parities.items():
        result = resolve_context_window_size(model, LLM_CONTEXT_WINDOW_SIZES)
        assert result == int(expected_size * CONTEXT_WINDOW_USAGE_RATIO), f"Mismatch for {model}"
