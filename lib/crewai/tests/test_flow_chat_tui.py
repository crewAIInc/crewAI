from unittest.mock import Mock

import pytest

from crewai.flow import ConversationState, Flow


class ChatFlow(Flow[ConversationState]):
    conversational = True


def test_tui_chat_uses_shared_launcher_and_preserves_turn_options(monkeypatch):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    launch = Mock()
    monkeypatch.setattr("crewai_cli.kickoff_flow._run_conversational_flow_tui", launch)
    flow = ChatFlow()
    flow.chat(
        tui=True,
        stream=False,
        session_id="session-1",
        exit_commands=("bye",),
        defer_trace_finalization=False,
        input_files={"brief": "brief.txt"},
    )
    launch.assert_called_once_with(
        flow,
        stream=False,
        session_id="session-1",
        exit_commands=("bye",),
        defer_trace_finalization=False,
        input_files={"brief": "brief.txt"},
    )


def test_tui_chat_requires_a_terminal(monkeypatch):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: False)
    with pytest.raises(ValueError, match="interactive terminal"):
        ChatFlow().chat(tui=True)


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": "User: "},
        {"assistant_prefix": "Bot: "},
        {"input_fn": lambda _: "quit"},
        {"output_fn": lambda _: None},
        {"skip_empty": False},
    ],
)
def test_tui_rejects_repl_only_options(monkeypatch, options):
    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    with pytest.raises(ValueError, match="plain chat"):
        ChatFlow().chat(tui=True, **options)


def test_tui_chat_requires_conversational_opt_in():
    with pytest.raises(ValueError, match="conversational flows"):
        Flow().chat(tui=True)


def test_stream_option_requires_tui_for_chat():
    with pytest.raises(ValueError, match="requires tui=True"):
        ChatFlow().chat(stream=False)


@pytest.mark.parametrize("kind", ["agent", "flow"])
@pytest.mark.parametrize(
    "llm_options,stream,expected",
    [({}, None, True), ({"stream": False}, None, False), ({"stream": True}, False, False), ({"stream": False}, True, True)],
)
def test_chat_turn_streaming_defaults_and_opt_outs(monkeypatch, kind, llm_options, stream, expected):
    from crewai import Agent, BaseLLM
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import LLMStreamChunkEvent
    from crewai.flow import ConversationConfig

    class StreamingLLM(BaseLLM):
        stream: bool = False

        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            seen.append(self._effective_stream())
            if self._effective_stream():
                crewai_event_bus.emit(self, LLMStreamChunkEvent(chunk="Public answer", call_id="stream-default-check"))
            return "Final Answer: Public answer"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    seen = []
    llm = StreamingLLM(model="recording", **llm_options)
    original_fields = llm.model_fields_set.copy()
    if kind == "agent":
        agent = Agent(role="Assistant", goal="Help", backstory="Helpful", llm=llm, conversational=True)
        flow = agent.as_flow()
    else:
        @ConversationConfig(llm=llm, defer_trace_finalization=False)
        class ModelChat(ChatFlow):
            pass
        flow = ModelChat()

    for prompt in ("Remember Cedar Finch", "What is my codename?"):
        turn = flow.stream_turn(prompt, stream=stream)
        frames = list(turn)
        assert bool([frame for frame in frames if frame.type == "llm_stream_chunk"]) is expected
        assert str(turn.result) in ("Public answer", "Final Answer: Public answer")

    assert seen == [expected, expected]
    assert len(flow.conversation_messages) == 4
    assert llm.stream is llm_options.get("stream", False)
    assert llm.model_fields_set == original_fields
    if kind == "agent":
        agent.kickoff("Independent execution")
        assert seen[-1] is llm_options.get("stream", False)
