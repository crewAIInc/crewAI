from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    model: str = "test-model"

    def call(self, *args: Any, **kwargs: Any) -> str:
        return ""

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def supports_function_calling(self) -> bool:
        return False


@pytest.fixture
def mock_emit():
    with patch.object(CrewAIEventsBus, "emit") as mock:
        yield mock


class TestLLMCallCompletedEventFinishReasonAndResponseId:
    def test_accepts_string_values(self):
        event = LLMCallCompletedEvent(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            call_id="call-1",
            finish_reason="stop",
            response_id="resp_123",
        )
        assert event.finish_reason == "stop"
        assert event.response_id == "resp_123"

    def test_defaults_to_none(self):
        event = LLMCallCompletedEvent(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            call_id="call-1",
        )
        assert event.finish_reason is None
        assert event.response_id is None

    @pytest.mark.parametrize(
        "value",
        [MagicMock(), 42, 1.5, ["stop"], {"reason": "stop"}, object()],
    )
    def test_coerces_non_string_to_none(self, value):
        event = LLMCallCompletedEvent(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            call_id="call-1",
            finish_reason=value,
            response_id=value,
        )
        assert event.finish_reason is None
        assert event.response_id is None


class TestLLMCallStartedEventSamplingParams:
    def test_accepts_all_sampling_params(self):
        event = LLMCallStartedEvent(
            call_id="call-1",
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stream=True,
            seed=42,
            stop_sequences=["END"],
            frequency_penalty=0.1,
            presence_penalty=0.2,
            n=3,
        )
        assert event.temperature == 0.7
        assert event.top_p == 0.9
        assert event.max_tokens == 512
        assert event.stream is True
        assert event.seed == 42
        assert event.stop_sequences == ["END"]
        assert event.frequency_penalty == 0.1
        assert event.presence_penalty == 0.2
        assert event.n == 3

    def test_all_sampling_params_default_to_none(self):
        event = LLMCallStartedEvent(call_id="call-1")
        assert event.temperature is None
        assert event.top_p is None
        assert event.max_tokens is None
        assert event.stream is None
        assert event.seed is None
        assert event.stop_sequences is None
        assert event.frequency_penalty is None
        assert event.presence_penalty is None
        assert event.n is None


class TestEmitCallStartedEventIntrospectsSamplingParams:
    def test_reads_sampling_params_off_self(self, mock_emit):
        llm = _StubLLM(model="test-model", temperature=0.4)
        llm.top_p = 0.8
        llm.max_tokens = 256
        llm.stream = False
        llm.seed = 7
        llm.frequency_penalty = 0.5
        llm.presence_penalty = 0.6
        llm.n = 2
        llm.stop = ["STOP"]

        llm._emit_call_started_event(messages="hi")

        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, LLMCallStartedEvent)
        assert event.temperature == 0.4
        assert event.top_p == 0.8
        assert event.max_tokens == 256
        assert event.stream is False
        assert event.seed == 7
        assert event.stop_sequences == ["STOP"]
        assert event.frequency_penalty == 0.5
        assert event.presence_penalty == 0.6
        assert event.n == 2

    def test_explicit_kwargs_override_introspection(self, mock_emit):
        llm = _StubLLM(model="test-model", temperature=0.4)

        llm._emit_call_started_event(messages="hi", temperature=0.9)

        event = mock_emit.call_args[1]["event"]
        assert event.temperature == 0.9


class TestEmitCallCompletedEventPassesFinishReasonAndResponseId:
    def test_passes_through_to_event(self, mock_emit):
        llm = _StubLLM(model="test-model")

        llm._emit_call_completed_event(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            finish_reason="stop",
            response_id="resp_123",
        )

        event = mock_emit.call_args[1]["event"]
        assert isinstance(event, LLMCallCompletedEvent)
        assert event.finish_reason == "stop"
        assert event.response_id == "resp_123"

    def test_omitted_defaults_to_none(self, mock_emit):
        llm = _StubLLM(model="test-model")

        llm._emit_call_completed_event(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
        )

        event = mock_emit.call_args[1]["event"]
        assert event.finish_reason is None
        assert event.response_id is None


class TestLLMExtractFinishReasonAndResponseId:
    def test_non_streaming_litellm_shape(self):
        response = SimpleNamespace(
            id="chatcmpl-abc",
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace())],
        )

        finish_reason, response_id = LLM._extract_finish_reason_and_response_id(
            response
        )

        assert finish_reason == "stop"
        assert response_id == "chatcmpl-abc"

    def test_streaming_litellm_chunk_shape(self):
        last_chunk = SimpleNamespace(
            id="chatcmpl-stream-xyz",
            choices=[SimpleNamespace(finish_reason="tool_calls", delta=SimpleNamespace())],
        )

        finish_reason, response_id = LLM._extract_finish_reason_and_response_id(
            last_chunk
        )

        assert finish_reason == "tool_calls"
        assert response_id == "chatcmpl-stream-xyz"

    def test_dict_shape(self):
        chunk = {
            "id": "chatcmpl-dict",
            "choices": [{"finish_reason": "length", "delta": {}}],
        }

        finish_reason, response_id = LLM._extract_finish_reason_and_response_id(chunk)

        assert finish_reason == "length"
        assert response_id == "chatcmpl-dict"

    def test_missing_fields_return_none(self):
        finish_reason, response_id = LLM._extract_finish_reason_and_response_id(
            SimpleNamespace()
        )

        assert finish_reason is None
        assert response_id is None

    def test_non_string_values_coerced_to_none(self):
        response = SimpleNamespace(
            id=12345,
            choices=[SimpleNamespace(finish_reason=MagicMock(), delta=SimpleNamespace())],
        )

        finish_reason, response_id = LLM._extract_finish_reason_and_response_id(
            response
        )

        assert finish_reason is None
        assert response_id is None

    def test_never_raises_on_unexpected_input(self):
        assert LLM._extract_finish_reason_and_response_id(None) == (None, None)
        assert LLM._extract_finish_reason_and_response_id(42) == (None, None)
        assert LLM._extract_finish_reason_and_response_id("string") == (None, None)
