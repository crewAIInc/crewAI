"""Context compaction through the conversational Flow entry points."""

from copy import deepcopy
from typing import Any
from unittest.mock import patch

import httpx
from openai import BadRequestError
import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    ContextWindowSummarizationCompletedEvent,
    ContextWindowSummarizationStartedEvent,
    LLMStreamChunkEvent,
)
from crewai.flow import (
    ConversationConfig,
    ConversationMessage,
    ConversationState,
    Flow,
    RouterConfig,
    listen,
)
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.message_utils import message_content_text
from crewai.utilities.types import LLMMessage


class CompactionLLM(BaseLLM):
    """Use the real summarizer with deterministic model responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        super().__init__(model="test-compaction")
        object.__setattr__(self, "responses", list(responses or ["Friday."]))
        object.__setattr__(self, "requests", [])
        object.__setattr__(self, "summaries", [])

    def call(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        self.requests.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if self._effective_stream():
            self._emit_stream_chunk_event(chunk=str(response))
        return response

    async def acall(self, messages: list[LLMMessage], **kwargs: Any) -> str:
        self.summaries.append((deepcopy(messages), self._effective_stream()))
        return "<summary>Project ORBIT launches Friday.</summary>"

    def get_input_tokens_count(self, messages: Any, *args: Any) -> int:
        text = (
            messages
            if isinstance(messages, str)
            else "".join(message_content_text(message) for message in messages)
        )
        return (len(text) + 3) // 4

    def get_context_window_size(self) -> int:
        return 4096


def chat_flow(llm: CompactionLLM, **config: Any) -> Flow[ConversationState]:
    @ConversationConfig(llm=llm, system_prompt="Keep project details.", **config)
    class Chat(Flow[ConversationState]):
        pass

    return Chat()


def run_turn(flow: Flow[ConversationState], streaming: bool) -> Any:
    if not streaming:
        return flow.handle_turn("When does ORBIT launch?")
    stream = flow.stream_turn("When does ORBIT launch?")
    list(stream)
    return stream.result


@pytest.mark.parametrize("streaming", [False, True])
def test_converse_compacts_request_and_preserves_transcript(streaming: bool) -> None:
    llm = CompactionLLM()
    flow = chat_flow(llm)
    flow.state.messages = [
        ConversationMessage(
            role="user",
            content=[{"type": "text", "text": "Project ORBIT. " * 2000}],
        ),
        ConversationMessage(role="assistant", content="Launch Friday."),
    ]
    original = deepcopy(flow.state.messages)
    events: list[Any] = []

    def capture(_: Any, event: Any) -> None:
        events.append(event)

    with crewai_event_bus.scoped_handlers():
        for event_type in (
            ContextWindowSummarizationStartedEvent,
            ContextWindowSummarizationCompletedEvent,
            LLMStreamChunkEvent,
        ):
            crewai_event_bus.on(event_type)(capture)
        assert run_turn(flow, streaming) == "Friday."
        crewai_event_bus.flush()

    assert llm.summaries
    assert all(not stream for _, stream in llm.summaries)
    assert len(llm.requests) == 1
    assert llm.requests[0][0] == {"role": "system", "content": "Keep project details."}
    assert "Project ORBIT launches Friday." in llm.requests[0][-2]["content"]
    assert llm.requests[0][-1] == {"role": "user", "content": "When does ORBIT launch?"}
    assert llm.get_input_tokens_count(llm.requests[0]) < llm.get_context_window_size()
    assert flow.state.messages[:2] == original
    assert flow.state.messages[-1].content == "Friday."
    assert any(
        isinstance(event, ContextWindowSummarizationStartedEvent) for event in events
    )
    assert any(
        isinstance(event, ContextWindowSummarizationCompletedEvent) for event in events
    )
    assert [
        event.chunk for event in events if isinstance(event, LLMStreamChunkEvent)
    ] == (["Friday."] if streaming else [])


@pytest.mark.parametrize("streaming", [False, True])
def test_converse_compacts_and_retries_actual_provider_error(streaming: bool) -> None:
    error = (
        BadRequestError(
            "maximum context length exceeded",
            response=httpx.Response(
                400, request=httpx.Request("POST", "https://example.test")
            ),
            body={"code": "context_length_exceeded"},
        )
        if streaming
        else LLMContextLengthExceededError("maximum context length exceeded")
    )
    llm = CompactionLLM([error, "Friday."])
    flow = chat_flow(llm)

    assert run_turn(flow, streaming) == "Friday."

    assert len(llm.requests) == 2
    assert len(llm.summaries) == 1
    assert "Project ORBIT launches Friday." in llm.requests[1][-2]["content"]
    assert llm.requests[1][-1] == {"role": "user", "content": "When does ORBIT launch?"}
    assert [message.role for message in flow.state.messages] == ["user", "assistant"]


def test_context_recovery_is_bounded() -> None:
    error = LLMContextLengthExceededError("maximum context length exceeded")
    llm = CompactionLLM([error, error])
    flow = chat_flow(llm)

    with pytest.raises(LLMContextLengthExceededError):
        flow.handle_turn("When does ORBIT launch?")

    assert len(llm.requests) == 2
    assert len(llm.summaries) == 1
    assert [message.role for message in flow.state.messages] == ["user"]


def test_other_provider_errors_are_not_retried() -> None:
    llm = CompactionLLM([RuntimeError("Connection unavailable")])
    flow = chat_flow(llm)

    with pytest.raises(RuntimeError, match="Connection unavailable"):
        flow.handle_turn("When does ORBIT launch?")

    assert len(llm.requests) == 1
    assert not llm.summaries


@pytest.mark.parametrize("failure_source", ["get_input_tokens_count", "acall"])
def test_compaction_failure_does_not_block_a_valid_call(failure_source: str) -> None:
    llm = CompactionLLM()
    flow = chat_flow(llm)
    flow.state.messages = [ConversationMessage(role="user", content="History. " * 3000)]
    with patch.object(
        llm, failure_source, side_effect=RuntimeError("Compaction unavailable")
    ):
        assert flow.handle_turn("When does ORBIT launch?") == "Friday."
    assert not llm.summaries


def test_failed_recovery_summary_propagates_without_retrying() -> None:
    llm = CompactionLLM(
        [LLMContextLengthExceededError("maximum context length exceeded")]
    )
    flow = chat_flow(llm)

    with (
        patch.object(llm, "acall", side_effect=RuntimeError("Summary unavailable")),
        pytest.raises(RuntimeError, match="Summary unavailable"),
    ):
        flow.handle_turn("When does ORBIT launch?")

    assert len(llm.requests) == 1
    assert [message.role for message in flow.state.messages] == ["user"]


def test_disabling_compaction_sends_original_request_and_propagates_error() -> None:
    llm = CompactionLLM(
        [LLMContextLengthExceededError("maximum context length exceeded")]
    )
    flow = chat_flow(llm, respect_context_window=False)
    flow.state.messages = [ConversationMessage(role="user", content="History. " * 3000)]

    with pytest.raises(LLMContextLengthExceededError):
        flow.handle_turn("When does ORBIT launch?")

    assert not llm.summaries
    assert len(llm.requests) == 1
    assert llm.requests[0][1]["content"] == "History. " * 3000


@pytest.mark.parametrize("enabled", [False, True])
def test_compaction_setting_survives_declaration_roundtrip(enabled: bool) -> None:
    @ConversationConfig(respect_context_window=enabled)
    class Chat(Flow[ConversationState]):
        pass

    declared = Flow.from_declaration(
        contents=Chat.flow_definition().model_dump(mode="json")
    )

    assert declared._conversation_config.respect_context_window is enabled


def test_router_compacts_single_context_payload_before_selecting_route() -> None:
    llm = CompactionLLM([{"intent": "research"}])

    @ConversationConfig(router=RouterConfig(llm=llm, routes=["research"]))
    class Chat(Flow[ConversationState]):
        @listen("research")
        def run_research(self) -> str:
            return "Research complete."

    flow = Chat()
    flow.state.messages = [ConversationMessage(role="user", content="History. " * 3000)]

    assert flow.handle_turn("Research ORBIT.") == "Research complete."
    assert llm.summaries
    assert flow.state.last_intent == "research"
    assert len(llm.requests) == 1
    assert llm.get_input_tokens_count(llm.requests[0]) < llm.get_context_window_size()
    assert llm.requests[0][-1] == {"role": "user", "content": "Research ORBIT."}


def test_deprecated_history_handler_uses_shared_compaction() -> None:
    llm = CompactionLLM()
    with pytest.warns(DeprecationWarning):
        config = ConversationConfig(answer_from_history_llm=llm)

    @config
    class Chat(Flow[ConversationState]):
        def route_turn(self, context: dict[str, Any]) -> str:
            return "answer_from_history"

    flow = Chat()
    flow.state.messages = [ConversationMessage(role="user", content="History. " * 3000)]

    assert flow.handle_turn("When does ORBIT launch?") == "Friday."
    assert llm.summaries
