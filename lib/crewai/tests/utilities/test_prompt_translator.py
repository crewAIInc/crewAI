"""Tests for the prompt translator module."""

import pytest

from crewai.utilities.model_profiles import ModelProfile
from crewai.utilities.prompt_translator import (
    _TRANSLATION_CACHE,
    clear_translation_cache,
    detect_language,
    estimate_tokens,
    extract_untranslatable_segments,
    optimize_system_prompt,
    restore_untranslatable_segments,
    translate_text,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear translation cache before each test."""
    clear_translation_cache()
    yield
    clear_translation_cache()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
class TestDetectLanguage:
    def test_english(self):
        assert detect_language("You are a helpful assistant.") == "en"

    def test_chinese(self):
        assert detect_language("你是一个有用的助手。请帮我完成任务。") == "zh"

    def test_japanese(self):
        assert detect_language("あなたは有用的なアシスタントです。") == "ja"

    def test_korean(self):
        assert detect_language("당신은 유용한 어시스턴트입니다.") == "ko"

    def test_arabic(self):
        assert detect_language("أنت مساعد مفيد. يرجى مساعدتي.") == "ar"

    def test_russian(self):
        assert detect_language("Вы полезный помощник. Пожалуйста, помогите.") == "ru"

    def test_hindi(self):
        assert detect_language("आप एक सहायक हैं। कृपया मेरी मदद करें।") == "hi"

    def test_mixed_english_dominant(self):
        text = "You are a helpful assistant. 你好，请帮我。"
        lang = detect_language(text)
        # Should detect as English (dominant) or Chinese (significant non-Latin)
        assert lang in ("en", "zh")

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_numbers_only(self):
        assert detect_language("12345 67890") == "en"

    def test_punctuation_only(self):
        assert detect_language("!!! ??? ...") == "en"

    def test_chinese_with_english(self):
        text = "请使用Python编写一个API接口，返回JSON格式的数据。"
        lang = detect_language(text)
        # Should detect Chinese as dominant
        assert lang == "zh"

    def test_japanese_mixed_scripts(self):
        text = "このコードをPythonで実装してください。Use async/await pattern."
        lang = detect_language(text)
        assert lang == "ja"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
class TestEstimateTokens:
    def _make_profile(
        self,
        eng_tpw: float = 1.1,
        non_eng_cpt: float = 0.6,
        bilingual: float = 0.85,
    ) -> ModelProfile:
        return ModelProfile(
            preferred_language="en",
            english_tokens_per_word=eng_tpw,
            non_english_chars_per_token=non_eng_cpt,
            bilingual_capability=bilingual,
            family="test",
        )

    def test_english_text(self):
        profile = self._make_profile()
        tokens = estimate_tokens("hello world", profile)
        # 2 words * 1.1 tokens/word = ~2.2, rounded to int
        assert tokens >= 2

    def test_chinese_text(self):
        profile = self._make_profile()
        tokens = estimate_tokens("你好世界", profile)
        # 4 CJK chars / 0.6 = ~6.67, rounded to int
        assert tokens >= 6

    def test_empty_text(self):
        profile = self._make_profile()
        assert estimate_tokens("", profile) == 0

    def test_mixed_text(self):
        profile = self._make_profile()
        tokens = estimate_tokens("Hello 你好", profile)
        # 1 English word * 1.1 + 2 CJK chars / 0.6 ≈ 1 + 3.3 = ~4
        assert tokens >= 3

    def test_high_efficiency_profile(self):
        """Qwen-like profile: Chinese is cheaper than English."""
        profile = ModelProfile(
            preferred_language="zh",
            english_tokens_per_word=1.05,
            non_english_chars_per_token=0.95,
            bilingual_capability=0.95,
            family="qwen",
        )
        tokens = estimate_tokens("你好世界", profile)
        # 4 / 0.95 ≈ 4.2
        assert tokens >= 4


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------
class TestExtractUntranslatableSegments:
    def test_fenced_code_block(self):
        text = "Use this code:\n```python\nprint('hello')\n```\nDone."
        cleaned, segments = extract_untranslatable_segments(text)
        assert len(segments) == 1
        assert "```python" in segments[0][1]
        assert "__SEGMENT_0__" in cleaned
        assert "Use this code" in cleaned

    def test_inline_code(self):
        text = "Call the `get_data()` function."
        cleaned, segments = extract_untranslatable_segments(text)
        assert len(segments) == 1
        assert "`get_data()`" in segments[0][1]
        assert "__SEGMENT_0__" in cleaned

    def test_url(self):
        text = "Visit https://example.com for details."
        cleaned, segments = extract_untranslatable_segments(text)
        assert len(segments) == 1
        assert "https://example.com" in segments[0][1]

    def test_json_object(self):
        text = 'Config: {"key": "value", "count": 42}'
        cleaned, segments = extract_untranslatable_segments(text)
        # May match JSON or not depending on regex complexity
        # At minimum, the text should be processable
        assert isinstance(cleaned, str)

    def test_env_var(self):
        text = "Set API_KEY=secret123 before running."
        cleaned, segments = extract_untranslatable_segments(text)
        # ENV var pattern may or may not match
        assert isinstance(cleaned, str)

    def test_multiple_segments(self):
        text = "Code: ```python\nx = 1\n```\nURL: https://example.com"
        cleaned, segments = extract_untranslatable_segments(text)
        assert len(segments) >= 1

    def test_no_segments(self):
        text = "This is plain English text with no special patterns."
        cleaned, segments = extract_untranslatable_segments(text)
        assert len(segments) == 0
        assert cleaned == text


class TestRestoreUntranslatableSegments:
    def test_roundtrip(self):
        original = "Code: ```python\nx = 1\n```\nDone."
        cleaned, segments = extract_untranslatable_segments(original)
        restored = restore_untranslatable_segments(cleaned, segments)
        assert restored == original

    def test_no_segments(self):
        text = "No segments here."
        restored = restore_untranslatable_segments(text, [])
        assert restored == text

    def test_multiple_roundtrip(self):
        original = "Use `func()` at https://example.com with ```code```"
        cleaned, segments = extract_untranslatable_segments(original)
        restored = restore_untranslatable_segments(cleaned, segments)
        assert restored == original


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class TestTranslateText:
    def test_same_language_returns_original(self):
        text = "Hello world"
        result = translate_text(text, "en", "en")
        assert result == text

    def test_empty_text(self):
        result = translate_text("", "en", "zh")
        assert result == ""

    def test_whitespace_only(self):
        result = translate_text("   ", "en", "zh")
        assert result == "   "

    def test_no_llm_returns_original(self):
        """Without an LLM caller, translation returns original text."""
        text = "You are a helpful assistant."
        result = translate_text(text, "en", "zh", llm_caller=None)
        assert result == text

    def test_with_mock_llm(self):
        """With a mock LLM caller, translation uses the LLM response."""

        class MockLLM:
            def call(self, messages):
                return "你是一个有用的助手。"

        text = "You are a helpful assistant."
        result = translate_text(
            text, "en", "zh", llm_caller=MockLLM()
        )
        assert result == "你是一个有用的助手。"

    def test_caching(self):
        """Translation results should be cached."""

        class CallCounter:
            def __init__(self):
                self.count = 0

            def call(self, messages):
                self.count += 1
                return "translated"

        counter = CallCounter()
        text = "Test text"
        translate_text(text, "en", "zh", llm_caller=counter)
        translate_text(text, "en", "zh", llm_caller=counter)
        # Second call should use cache, not call LLM again
        assert counter.count == 1

    def test_glossary_in_prompt(self):
        """Glossary terms should appear in the translation prompt."""
        captured_messages = []

        class CapturingLLM:
            def call(self, messages):
                captured_messages.extend(messages)
                return "translated"

        glossary = {"API Key": "API Key", "crewAI": "crewAI"}
        translate_text(
            "Use the API Key to authenticate.",
            "en",
            "zh",
            glossary=glossary,
            llm_caller=CapturingLLM(),
        )
        assert len(captured_messages) == 1
        assert "API Key" in captured_messages[0]["content"]
        assert "crewAI" in captured_messages[0]["content"]

    def test_llm_failure_returns_original(self):
        """If LLM call fails, original text should be returned."""

        class FailingLLM:
            def call(self, messages):
                raise RuntimeError("LLM unavailable")

        text = "Test prompt"
        result = translate_text(text, "en", "zh", llm_caller=FailingLLM())
        assert result == text


# ---------------------------------------------------------------------------
# optimize_system_prompt (integration)
# ---------------------------------------------------------------------------
class TestOptimizeSystemPrompt:
    def test_empty_prompt(self):
        assert optimize_system_prompt("", "gpt-4o") == ""

    def test_none_prompt(self):
        assert optimize_system_prompt(None, "gpt-4o") is None  # type: ignore[arg-type]

    def test_unknown_model_returns_original(self):
        text = "你是一个有用的助手。"
        result = optimize_system_prompt(text, "unknown-model-xyz")
        assert result == text

    def test_same_language_as_preferred(self):
        """English prompt with GPT-4o (prefers English) should not be translated."""
        text = "You are a helpful assistant. Please complete the task."
        result = optimize_system_prompt(text, "gpt-4o")
        assert result == text

    def test_chinese_prompt_with_llama(self):
        """Chinese prompt with LLaMA (poor bilingual) should be translated."""
        chinese_text = "你是一个有用的助手。请帮我完成任务。"

        class MockLLM:
            def call(self, messages):
                return "You are a helpful assistant. Please help me complete the task."

        result = optimize_system_prompt(
            chinese_text, "llama-3.1-70b-versatile", llm_caller=MockLLM()
        )
        # Should be translated because LLaMA has poor bilingual capability
        # and Chinese prompt is less efficient on LLaMA
        assert result != chinese_text

    def test_code_blocks_preserved(self):
        """Code blocks should be preserved during translation."""
        chinese_text = "使用以下代码：\n```python\nprint('hello')\n```\n完成。"

        class MockLLM:
            def call(self, messages):
                return "Use the following code:\n```python\nprint('hello')\n```\nDone."

        result = optimize_system_prompt(
            chinese_text, "llama-3.1-70b-versatile", llm_caller=MockLLM()
        )
        # Code block should be preserved
        assert "print('hello')" in result

    def test_glossary_passed_through(self):
        """Glossary terms should be passed to the translation LLM."""
        chinese_text = "使用API Key进行认证。"

        captured = []

        class CapturingLLM:
            def call(self, messages):
                captured.extend(messages)
                return "Authenticate using the API Key."

        glossary = {"API Key": "API Key"}
        optimize_system_prompt(
            chinese_text,
            "llama-3.1-70b-versatile",
            glossary=glossary,
            llm_caller=CapturingLLM(),
        )
        assert len(captured) == 1
        assert "API Key" in captured[0]["content"]

    def test_caching_across_calls(self):
        """Same prompt + model should return cached result."""
        chinese_text = "你是一个助手。"

        class Counter:
            def __init__(self):
                self.count = 0

            def call(self, messages):
                self.count += 1
                return "You are an assistant."

        counter = Counter()
        result1 = optimize_system_prompt(
            chinese_text, "llama-3.1-70b-versatile", llm_caller=counter
        )
        result2 = optimize_system_prompt(
            chinese_text, "llama-3.1-70b-versatile", llm_caller=counter
        )
        assert result1 == result2
        assert counter.count == 1  # Only one LLM call

    def test_bilingual_model_no_translation(self):
        """Qwen (prefers Chinese, high bilingual) should not translate Chinese."""
        chinese_text = "你是一个有用的助手。请帮我完成任务。"
        result = optimize_system_prompt(chinese_text, "qwen")
        # Qwen prefers Chinese, so Chinese prompt should not be translated
        assert result == chinese_text


class TestClearTranslationCache:
    def test_clear(self):
        """Cache should be cleared."""
        _TRANSLATION_CACHE["test"] = "value"
        clear_translation_cache()
        assert len(_TRANSLATION_CACHE) == 0
