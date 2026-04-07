from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM


class TestLLMCallCompletedEventUsageField:
    def test_accepts_usage_dict(self):
        event = LLMCallCompletedEvent(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-id",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        assert event.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_usage_defaults_to_none(self):
        event = LLMCallCompletedEvent(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-id",
        )
        assert event.usage is None

    def test_accepts_none_usage(self):
        event = LLMCallCompletedEvent(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-id",
            usage=None,
        )
        assert event.usage is None

    def test_accepts_nested_usage_dict(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 50},
        }
        event = LLMCallCompletedEvent(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-id",
            usage=usage,
        )
        assert event.usage["prompt_tokens_details"]["cached_tokens"] == 50


class TestUsageToDict:
    def test_none_returns_none(self):
        assert LLM._usage_to_dict(None) is None

    def test_dict_passes_through(self):
        usage = {"prompt_tokens": 10, "total_tokens": 30}
        assert LLM._usage_to_dict(usage) is usage

    def test_pydantic_model_uses_model_dump(self):
        class Usage(BaseModel):
            prompt_tokens: int = 10
            completion_tokens: int = 20
            total_tokens: int = 30

        result = LLM._usage_to_dict(Usage())
        assert result == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

    def test_object_with_dict_attr(self):
        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 5
                self.completion_tokens = 15
                self.total_tokens = 20

        result = LLM._usage_to_dict(UsageObj())
        assert result == {
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20,
        }

    def test_object_with_dict_excludes_private_attrs(self):
        class UsageObj:
            def __init__(self):
                self.total_tokens = 42
                self._internal = "hidden"

        result = LLM._usage_to_dict(UsageObj())
        assert result == {"total_tokens": 42}
        assert "_internal" not in result

    def test_unsupported_type_returns_none(self):
        assert LLM._usage_to_dict(42) is None
        assert LLM._usage_to_dict("string") is None


class _StubLLM(BaseLLM):
    """Minimal concrete BaseLLM for testing event emission."""

    model: str = "test-model"

    def call(self, *args: Any, **kwargs: Any) -> str:
        return ""

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True


class TestEmitCallCompletedEventPassesUsage:
    @pytest.fixture
    def mock_emit(self):
        with patch.object(CrewAIEventsBus, "emit") as mock:
            yield mock

    @pytest.fixture
    def llm(self):
        return _StubLLM(model="test-model")

    def test_usage_is_passed_to_event(self, mock_emit, llm):
        usage_data = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        llm._emit_call_completed_event(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            messages="test prompt",
            usage=usage_data,
        )

        mock_emit.assert_called_once()
        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.usage == usage_data

    def test_none_usage_is_passed_to_event(self, mock_emit, llm):
        llm._emit_call_completed_event(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            messages="test prompt",
            usage=None,
        )

        mock_emit.assert_called_once()
        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.usage is None

    def test_usage_omitted_defaults_to_none(self, mock_emit, llm):
        llm._emit_call_completed_event(
            response="hello",
            call_type=LLMCallType.LLM_CALL,
            messages="test prompt",
        )

        mock_emit.assert_called_once()
        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.usage is None
