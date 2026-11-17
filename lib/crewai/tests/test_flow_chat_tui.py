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


@pytest.mark.parametrize("transport", ["turn", "chat"])
@pytest.mark.parametrize("kind", ["agent", "flow"])
@pytest.mark.parametrize(
    "llm_options,stream,expected",
    [({}, None, True), ({"stream": False}, None, False), ({"stream": True}, False, False), ({"stream": False}, True, True)],
)
def test_chat_turn_streaming_defaults_and_opt_outs(monkeypatch, transport, kind, llm_options, stream, expected):
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

    if transport == "chat":
        messages = iter(["Remember Cedar Finch", "What is my codename?", "exit"])
        outputs = []
        chat = agent.chat if kind == "agent" else flow.chat
        chat(stream=stream, input_fn=lambda _: next(messages), output_fn=outputs.append)
        assert len(outputs) == 2
        assert all(output.endswith("Public answer") for output in outputs)
    else:
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


def test_plain_agent_chat_prints_chunks_before_model_finishes(monkeypatch):
    from io import StringIO
    from threading import Event

    from crewai import Agent, BaseLLM
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import (
        LLMCallType, LLMStreamChunkEvent, LLMThinkingChunkEvent,
    )

    displayed = Event()

    class Terminal(StringIO):
        def write(self, text):
            result = super().write(text)
            if "Public " in self.getvalue():
                displayed.set()
            return result

    class StreamingLLM(BaseLLM):
        stream: bool = False

        def supports_function_calling(self):
            return False

        def call(self, messages, **kwargs):
            assert self._effective_stream()
            for event in (
                LLMThinkingChunkEvent(chunk="private reasoning", call_id="live"),
                LLMStreamChunkEvent(chunk="private arguments", call_type=LLMCallType.TOOL_CALL, call_id="live"),
                LLMStreamChunkEvent(chunk="Public ", call_type=LLMCallType.LLM_CALL, call_id="live"),
            ):
                crewai_event_bus.emit(self, event)
            assert displayed.wait(5), "The first chunk was buffered until model completion"
            crewai_event_bus.emit(self, LLMStreamChunkEvent(chunk="answer", call_id="live"))
            return "Final Answer: Public answer"

        async def acall(self, messages, **kwargs):
            return self.call(messages, **kwargs)

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    terminal = Terminal()
    monkeypatch.setattr("sys.stdout", terminal)
    agent = Agent(role="Assistant", goal="Help", backstory="Helpful", llm=StreamingLLM(model="recording"), conversational=True, verbose=False)
    messages = iter(["Hello", "exit"])
    agent.chat(input_fn=lambda _: next(messages))
    assert terminal.getvalue() == "\nAssistant: Public answer\n"
    assert agent.as_flow().conversation_messages[-1] == {"role": "assistant", "content": "Public answer"}


def test_plain_chat_does_not_print_router_chunks(monkeypatch, capsys):
    import contextvars

    from crewai_core.printer import set_suppress_console_output

    from crewai import BaseLLM
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import LLMStreamChunkEvent
    from crewai.flow import ConversationConfig

    class ReplyLLM(BaseLLM):
        stream: bool = False

        def call(self, messages, **kwargs):
            assert self._effective_stream()
            crewai_event_bus.emit(self, LLMStreamChunkEvent(chunk="Public answer", call_id="public"))
            return "Public answer"

    @ConversationConfig(llm=ReplyLLM(model="recording"))
    class RoutedChat(ChatFlow):
        def route_turn(self, context):
            crewai_event_bus.emit(self, LLMStreamChunkEvent(chunk="private router text", call_id="router"))
            return "converse"

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    context = contextvars.copy_context()
    context.run(set_suppress_console_output, True)
    flow = RoutedChat()
    messages = iter(["Hello", "exit"])
    context.run(flow.chat, input_fn=lambda _: next(messages))
    assert capsys.readouterr().out == "\nAssistant: Public answer\n"
