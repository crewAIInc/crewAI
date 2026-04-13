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

class TestUsageMetricsNewFields:
    def test_add_usage_metrics_aggregates_reasoning_and_cache_creation(self):
        from crewai.types.usage_metrics import UsageMetrics

        metrics1 = UsageMetrics(
            total_tokens=100,
            prompt_tokens=60,
            completion_tokens=40,
            cached_prompt_tokens=10,
            reasoning_tokens=15,
            cache_creation_tokens=5,
            successful_requests=1,
        )
        metrics2 = UsageMetrics(
            total_tokens=200,
            prompt_tokens=120,
            completion_tokens=80,
            cached_prompt_tokens=20,
            reasoning_tokens=25,
            cache_creation_tokens=10,
            successful_requests=1,
        )

        metrics1.add_usage_metrics(metrics2)

        assert metrics1.total_tokens == 300
        assert metrics1.prompt_tokens == 180
        assert metrics1.completion_tokens == 120
        assert metrics1.cached_prompt_tokens == 30
        assert metrics1.reasoning_tokens == 40
        assert metrics1.cache_creation_tokens == 15
        assert metrics1.successful_requests == 2

    def test_new_fields_default_to_zero(self):
        from crewai.types.usage_metrics import UsageMetrics

        metrics = UsageMetrics()
        assert metrics.reasoning_tokens == 0
        assert metrics.cache_creation_tokens == 0

    def test_model_dump_includes_new_fields(self):
        from crewai.types.usage_metrics import UsageMetrics

        metrics = UsageMetrics(reasoning_tokens=10, cache_creation_tokens=5)
        dumped = metrics.model_dump()
        assert dumped["reasoning_tokens"] == 10
        assert dumped["cache_creation_tokens"] == 5
