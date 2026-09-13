"""Tests for o1, o1-pro, o3 context window sizes (issue #7303)."""

import pytest

from crewai.llm import LLM, LLM_CONTEXT_WINDOW_SIZES, CONTEXT_WINDOW_USAGE_RATIO


class TestReasoningModelContextWindows:
    """o1, o1-pro, o3 should have 200k context windows."""

    def test_o1_has_200k_context_window(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o1") == 200000

    def test_o1_pro_has_200k_context_window(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o1-pro") == 200000

    def test_o3_has_200k_context_window(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o3") == 200000

    def test_o1_preview_retains_128k(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o1-preview") == 128000

    def test_o1_mini_retains_128k(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o1-mini") == 128000

    def test_o3_mini_retains_200k(self):
        assert LLM_CONTEXT_WINDOW_SIZES.get("o3-mini") == 200000

    def test_o1_llm_instance_returns_correct_size(self):
        llm = LLM(model="o1")
        size = llm.get_context_window_size()
        expected = int(200000 * CONTEXT_WINDOW_USAGE_RATIO)
        assert size == expected

    def test_o1_preview_llm_instance_returns_128k(self):
        llm = LLM(model="o1-preview")
        size = llm.get_context_window_size()
        expected = int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
        assert size == expected

    def test_o3_llm_instance_returns_correct_size(self):
        llm = LLM(model="o3")
        size = llm.get_context_window_size()
        expected = int(200000 * CONTEXT_WINDOW_USAGE_RATIO)
        assert size == expected
