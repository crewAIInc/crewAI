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
from crewai.llms._finish_reason_utils import extract_choices_finish_reason_and_id
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


class TestStopSequencesCoercion:
    # The OTel SDK falls back to str(value) when a span attribute isn't a
    # recognised Sequence[str], producing the protobuf textproto repr
    # ("values { string_value: ... }") in downstream telemetry. The
    # field_validator coerces exotic iterables (Vertex/Gemini protobuf
    # containers, tuples, generators) to a clean list[str] up front so the
    # OTel attribute is always shaped correctly.
    def test_bare_string_is_wrapped_in_list(self):
        event = LLMCallStartedEvent(call_id="call-1", stop_sequences="\nObservation:")
        assert event.stop_sequences == ["\nObservation:"]

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (["\nObservation:", "Final Answer:"], ["\nObservation:", "Final Answer:"]),
            (("\nObservation:",), ["\nObservation:"]),
            ((s for s in ["a", "b"]), ["a", "b"]),
            ([], []),
        ],
    )
    def test_python_iterables_pass_through(
        self, raw: Any, expected: list[str]
    ) -> None:
        event = LLMCallStartedEvent(call_id="call-1", stop_sequences=raw)
        assert event.stop_sequences == expected

    def test_protobuf_like_repeated_container_is_coerced(self):
        # Mirrors google.protobuf RepeatedScalarContainer: iterable yielding
        # actual Python str objects. Should pass through cleanly.
        class _RepeatedScalar:
            def __init__(self, items: list[str]) -> None:
                self._items = items

            def __iter__(self):
                return iter(self._items)

        event = LLMCallStartedEvent(
            call_id="call-1",
            stop_sequences=_RepeatedScalar(["\nObservation:"]),
        )
        assert event.stop_sequences == ["\nObservation:"]

    def test_protobuf_listvalue_with_nested_values_coerces_to_textproto_strings(self):
        # Mirrors google.protobuf.struct_pb2.ListValue: iterable yielding
        # `Value` messages whose str() is "string_value: \"...\"". The
        # coercion will str() each element, which is still wrong-shaped but
        # at least lands as a real list[str] for the OTel attribute instead
        # of a single textproto-blob string. Documents observed behaviour;
        # the upstream fix is to pass list[str] to LLM.stop, not ListValue.
        class _PbValue:
            def __init__(self, string_value: str) -> None:
                self.string_value = string_value

            def __str__(self) -> str:
                return f'string_value: "{self.string_value}"'

        class _PbListValue:
            def __init__(self, values: list[_PbValue]) -> None:
                self.values = values

            def __iter__(self):
                return iter(self.values)

        event = LLMCallStartedEvent(
            call_id="call-1",
            stop_sequences=_PbListValue([_PbValue("\\nObservation:")]),
        )
        assert event.stop_sequences == ['string_value: "\\nObservation:"']

    @pytest.mark.parametrize("bad_input", [123, 12.5, object()])
    def test_non_iterable_falls_back_to_none(self, bad_input: Any) -> None:
        event = LLMCallStartedEvent(call_id="call-1", stop_sequences=bad_input)
        assert event.stop_sequences is None

    def test_none_stays_none(self):
        event = LLMCallStartedEvent(call_id="call-1", stop_sequences=None)
        assert event.stop_sequences is None


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


class TestExtractChoicesFinishReasonAndIdHelper:
    # The shared extractor is consumed by LLM (LiteLLM), OpenAI Chat, and Azure.
    # TestLLMExtractFinishReasonAndResponseId exercises the choices-shape paths
    # transitively; these tests cover the direct-call surface and the
    # import contract.
    @pytest.mark.parametrize(
        "response, expected",
        [
            (
                SimpleNamespace(
                    id="resp-1", choices=[SimpleNamespace(finish_reason="stop")]
                ),
                ("stop", "resp-1"),
            ),
            (
                {"id": "resp-2", "choices": [{"finish_reason": "length"}]},
                ("length", "resp-2"),
            ),
            (
                SimpleNamespace(
                    id="resp-3", choices=[{"finish_reason": "tool_calls"}]
                ),
                ("tool_calls", "resp-3"),
            ),
            (
                {
                    "id": "resp-4",
                    "choices": [SimpleNamespace(finish_reason="content_filter")],
                },
                ("content_filter", "resp-4"),
            ),
        ],
    )
    def test_extracts_choices_shape(
        self, response: Any, expected: tuple[str | None, str | None]
    ) -> None:
        assert extract_choices_finish_reason_and_id(response) == expected

    @pytest.mark.parametrize(
        "bad_input",
        [
            None,
            42,
            "string",
            {},
            SimpleNamespace(),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace()]),
            {"id": 12345, "choices": [{"finish_reason": MagicMock()}]},
        ],
    )
    def test_never_raises_returns_nones_or_coerces(self, bad_input: Any) -> None:
        finish_reason, response_id = extract_choices_finish_reason_and_id(bad_input)
        assert finish_reason is None or isinstance(finish_reason, str)
        assert response_id is None or isinstance(response_id, str)
