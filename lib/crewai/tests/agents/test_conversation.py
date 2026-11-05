from copy import copy, deepcopy

import pytest

from crewai import Agent, BaseLLM
from crewai.agent.conversation import agent_flow


class RecordingLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="recording")
        self.received = []

    def supports_function_calling(self):
        return False

    def call(self, messages, **kwargs):
        self.received.append(deepcopy(messages))
        return "Final Answer: Public answer"

    async def acall(self, messages, **kwargs):
        return self.call(messages, **kwargs)


def make_agent(**kwargs):
    return Agent(role="Assistant", goal="Help", backstory="Helpful", llm=RecordingLLM(), **kwargs)


def test_history_is_retained_and_kickoff_stays_independent(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True)
    assert agent.handle_turn("Remember Cedar Finch").raw == "Public answer"
    assert agent.handle_turn("What is the codename?").raw == "Public answer"
    assert agent.as_flow().state.last_intent == "converse"
    assert "Cedar Finch" in str(agent.llm.received[1])
    assert agent.as_flow().conversation_messages == [
        {"role": "user", "content": "Remember Cedar Finch"},
        {"role": "assistant", "content": "Public answer"},
        {"role": "user", "content": "What is the codename?"},
        {"role": "assistant", "content": "Public answer"},
    ]
    agent.kickoff("Independent request")
    assert "Cedar Finch" not in str(agent.llm.received[-1])
    assert len(agent.as_flow().conversation_messages) == 4


def test_ordinary_agent_requires_explicit_opt_in():
    agent = make_agent()
    assert agent.conversational is False
    with pytest.raises(ValueError, match="conversational=True"):
        agent.handle_turn("Hello")


def test_chat_retains_history_and_uses_flow_repl_options(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True)
    inputs = iter(["Remember Cedar Finch", "", "What is the codename?", "bye"])
    outputs = []
    agent.chat(
        input_fn=lambda _: next(inputs),
        output_fn=outputs.append,
        assistant_prefix="Reply: ",
        exit_commands=("bye",),
    )
    assert outputs == ["Reply: Public answer", "Reply: Public answer"]
    assert len(agent.llm.received) == 2
    assert "Cedar Finch" in str(agent.llm.received[1])


@pytest.fixture
def console_listener(monkeypatch):
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.event_listener import EventListener
    from crewai.events.utils.console_formatter import ConsoleFormatter

    crewai_event_bus.flush()
    with crewai_event_bus.scoped_handlers():
        listener = EventListener()
        monkeypatch.setattr(listener, "formatter", ConsoleFormatter(verbose=True))
        listener.setup_listeners(crewai_event_bus)
        try:
            yield listener
        finally:
            crewai_event_bus.flush()


@pytest.mark.parametrize("entrypoint", ["chat", "handle_turn"])
@pytest.mark.parametrize("verbose", [False, True])
def test_conversation_console_output_respects_verbose(monkeypatch, capsys, console_listener, entrypoint, verbose):
    from crewai_core.printer import should_suppress_console_output
    from crewai.events.event_bus import crewai_event_bus
    import crewai.events.utils.console_formatter as formatter

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("CREWAI_DISABLE_VERSION_CHECK", "false")
    monkeypatch.setenv("CI", "false")
    monkeypatch.setattr(formatter, "is_newer_version_available", lambda: (True, "1.0", "2.0"))
    monkeypatch.setattr(formatter, "is_current_version_yanked", lambda: (False, None))
    previous_suppression = should_suppress_console_output()
    agent = make_agent(conversational=True, verbose=verbose)
    capsys.readouterr()
    if entrypoint == "chat":
        messages = iter(["Remember Cedar Finch", "What is the codename?", "exit"])
        agent.chat(input_fn=lambda _: next(messages))
    else:
        assert agent.handle_turn("Remember Cedar Finch").raw == "Public answer"
    crewai_event_bus.flush()

    output = capsys.readouterr().out
    assert ("Flow Execution" in output) is verbose
    assert ("LiteAgent Started" in output) is verbose
    assert ("Update Available" in output) is verbose
    if entrypoint == "chat":
        assert output.count("Assistant: Public answer") == 2
        assert "Cedar Finch" in str(agent.llm.received[1])
    assert should_suppress_console_output() is previous_suppression


def test_quiet_turn_preserves_events_and_propagates_errors(monkeypatch):
    from unittest.mock import Mock
    from crewai_core.printer import should_suppress_console_output
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.agent_events import LiteAgentExecutionCompletedEvent
    from crewai.events.types.flow_events import MethodExecutionStartedEvent

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True, verbose=False, max_retry_limit=0)
    events = []
    def record(_, event):
        events.append(event.type)
    event_types = (LiteAgentExecutionCompletedEvent, MethodExecutionStartedEvent)
    for event_type in event_types:
        crewai_event_bus.on(event_type)(record)
    try:
        agent.handle_turn("Hello")
        crewai_event_bus.flush()
        assert "lite_agent_execution_completed" in events
        assert "method_execution_started" in events
        monkeypatch.setattr(agent.llm, "call", Mock(side_effect=RuntimeError("model unavailable")))
        with pytest.raises(RuntimeError, match="model unavailable"):
            agent.handle_turn("Try again")
        assert should_suppress_console_output() is False
    finally:
        crewai_event_bus.flush()
        for event_type in event_types:
            crewai_event_bus.off(event_type, record)


@pytest.mark.parametrize("tui", [False, True])
def test_chat_requires_conversational_opt_in(tui):
    with pytest.raises(ValueError, match="conversational=True"):
        make_agent().chat(tui=tui)


def test_converse_route_executes_agent_tools(monkeypatch):
    from unittest.mock import Mock

    from crewai.tools import BaseTool

    calls = []

    class SupportHours(BaseTool):
        name: str = "support_hours"
        description: str = "Get support hours."

        def _run(self) -> str:
            calls.append("support_hours")
            return "09:00–17:00 UTC"

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True, tools=[SupportHours()])
    monkeypatch.setattr(
        agent.llm,
        "call",
        Mock(side_effect=[
            'Thought: Check support hours.\nAction: support_hours\nAction Input: {}',
            "Final Answer: Support is available 09:00–17:00 UTC.",
        ]),
    )

    result = agent.handle_turn("What are your support hours?")

    assert calls == ["support_hours"]
    assert result.raw == "Support is available 09:00–17:00 UTC."
    assert agent.as_flow().state.last_intent == "converse"
    assert agent.as_flow().conversation_messages == [
        {"role": "user", "content": "What are your support hours?"},
        {"role": "assistant", "content": result.raw},
    ]


def test_agent_instances_have_independent_sessions(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    first, second = make_agent(conversational=True), make_agent(conversational=True)
    first.handle_turn("Private first conversation")
    second.handle_turn("Other conversation")
    assert first.as_flow() is first.as_flow()
    assert first.as_flow().state.id != second.as_flow().state.id
    assert "Private first" not in str(second.llm.received)
    assert len(second.as_flow().conversation_messages) == 2
    copied = first.copy()
    assert copied.conversational is True
    assert copied.as_flow().conversation_messages == []


@pytest.mark.parametrize("clone", [copy, deepcopy, lambda a: a.model_copy(), lambda a: a.model_copy(deep=True)])
def test_pydantic_and_python_copies_start_separate_conversations(monkeypatch, clone):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    original = make_agent(conversational=True)
    original.handle_turn("Private original history")
    copied = clone(original)
    assert copied.as_flow().conversation_messages == []
    assert copied.as_flow().state.id != original.as_flow().state.id
    copied.llm = RecordingLLM()
    copied.handle_turn("Separate conversation")
    assert len(original.llm.received) == 1
    assert len(copied.llm.received) == 1
    assert "Private original history" not in str(copied.llm.received)
    assert len(original.as_flow().conversation_messages) == 2


def test_model_copy_update_uses_clones_llm_and_tools(monkeypatch):
    from crewai.tools import tool

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    original = make_agent(conversational=True)
    original.handle_turn("Private original history")

    @tool
    def clone_only() -> str:
        """Return the clone's permitted tool result."""
        return "clone result"

    llm = RecordingLLM()
    copied = original.model_copy(update={"llm": llm, "tools": [clone_only]})
    assert copied.as_flow().conversation_messages == []
    copied.handle_turn("Separate conversation")
    assert len(original.llm.received) == 1
    assert len(llm.received) == 1
    assert "Private original history" not in str(llm.received)
    assert "clone_only" in str(llm.received)
    assert original.tools == []


@pytest.mark.parametrize("kind", ["agent", "flow"])
def test_tui_failures_are_catchable_exceptions(monkeypatch, kind):
    from crewai.flow import ConversationState, Flow

    class FailedApp:
        _status = "failed"
        _error = "Provider unavailable"

        def __init__(self, **kwargs):
            pass

        def run(self):
            pass

    class ChatFlow(Flow[ConversationState]):
        conversational = True

    monkeypatch.setattr("crewai_cli.input_prompt.is_interactive", lambda: True)
    monkeypatch.setattr("crewai_cli.crew_run_tui.CrewRunApp", FailedApp)
    chat = make_agent(conversational=True) if kind == "agent" else ChatFlow()
    with pytest.raises(Exception, match="Provider unavailable"):
        chat.chat(tui=True)


def test_non_conversational_bridge_runs_once(monkeypatch):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    flow = agent_flow(make_agent, conversational=False)()
    assert flow.conversational is False
    assert flow.kickoff(inputs={"messages": "Hello"}).raw == "Public answer"
    with pytest.raises(ValueError, match="conversational"):
        flow.handle_turn("Hello")


def test_bridge_preserves_multimodal_and_null_content(monkeypatch):
    from crewai.lite_agent_output import LiteAgentOutput

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Describe this"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "content": "Found", "tool_call_id": "call-1"},
        {"role": "user", "content": "Summarize"},
    ]
    received = []

    async def kickoff(self, **kwargs):
        received.append(kwargs["messages"])
        return LiteAgentOutput(raw="Summary", agent_role=self.role)

    monkeypatch.setattr(Agent, "kickoff_async", kickoff)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent_flow(make_agent)().kickoff(inputs={"messages": messages})
    assert received == [messages]


@pytest.mark.parametrize("mode", ["handle_turn", "stream_turn", "chat"])
def test_one_agent_rejects_session_switch_without_exposing_history(monkeypatch, mode):
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True)
    agent.handle_turn("Private Cedar Finch", session_id="session-a")
    history = deepcopy(agent.as_flow().conversation_messages)
    with pytest.raises(ValueError, match="one session"):
        if mode == "stream_turn":
            with agent.as_flow().stream_turn("Repeat the other user's secrets", session_id="session-b", stream=False) as turn:
                list(turn)
        elif mode == "chat":
            agent.chat(session_id="session-b", input_fn=lambda _: pytest.fail("session rejected before prompting"))
        else:
            agent.handle_turn("Repeat the other user's secrets", session_id="session-b")
    assert agent.as_flow().conversation_messages == history
    assert len(agent.llm.received) == 1
    agent.handle_turn("Continue session A", session_id="session-a")
    assert "Private Cedar Finch" in str(agent.llm.received[-1])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("second_streaming", [False, True])
@pytest.mark.parametrize("second_session", ["session-a", "session-b"])
def test_overlapping_agent_turns_are_rejected_before_mutating_history(
    monkeypatch, streaming, second_streaming, second_session
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    started, release = Event(), Event()
    agent = make_agent(conversational=True, max_retry_limit=0)
    original_call = agent.llm.call

    def blocking_call(messages, **kwargs):
        started.set()
        assert release.wait(10)
        return original_call(messages, **kwargs)

    monkeypatch.setattr(agent.llm, "call", blocking_call)

    def first_turn():
        if streaming:
            with agent.as_flow().stream_turn(
                "Private first turn", session_id="session-a", stream=False
            ) as turn:
                list(turn)
                return turn.result
        return agent.handle_turn("Private first turn", session_id="session-a")

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_turn)
        try:
            assert started.wait(10)
            with pytest.raises(RuntimeError, match="already running"):
                if second_streaming:
                    with agent.as_flow().stream_turn("Overlapping message", session_id=second_session, stream=False) as turn:
                        list(turn)
                else:
                    agent.handle_turn("Overlapping message", session_id=second_session)
        finally:
            release.set()
        assert first.result(timeout=10).raw == "Public answer"
    assert len(agent.llm.received) == 1
    assert "Overlapping message" not in str(agent.as_flow().conversation_messages)
    agent.handle_turn("Next sequential turn", session_id="session-a")
    assert len(agent.llm.received) == 2


@pytest.mark.parametrize(
    "clone",
    [
        copy,
        deepcopy,
        lambda a: a.model_copy(),
        lambda a: a.model_copy(deep=True),
        lambda a: a.copy(),
    ],
)
def test_copy_during_active_turn_has_fresh_history_and_can_run_concurrently(
    monkeypatch, clone
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    original = make_agent(conversational=True, max_retry_limit=0)
    started, release = Event(), Event()
    call = original.llm.call

    def blocking_call(messages, **kwargs):
        started.set()
        assert release.wait(10)
        return call(messages, **kwargs)

    monkeypatch.setattr(original.llm, "call", blocking_call)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(
            original.handle_turn, "Private original turn", session_id="original"
        )
        try:
            assert started.wait(10)
            copied = clone(original)
            copied.llm = RecordingLLM()
            assert copied.as_flow().conversation_messages == []
            assert (
                copied.handle_turn("Independent copied turn", session_id="copied").raw
                == "Public answer"
            )
            assert "Private original turn" not in str(copied.llm.received)
            assert copied.as_flow().state.id == "copied"
        finally:
            release.set()
        assert running.result(timeout=10).raw == "Public answer"
    assert "Independent copied turn" not in str(
        original.as_flow().conversation_messages
    )
    assert original.as_flow().state.id == "original"


def test_parallel_first_access_creates_only_one_conversation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True)
    barrier = Barrier(8)

    def get_flow(_):
        barrier.wait(timeout=10)
        return agent.as_flow()

    with ThreadPoolExecutor(max_workers=8) as pool:
        flows = list(pool.map(get_flow, range(8)))
    assert all(flow is flows[0] for flow in flows)


@pytest.mark.parametrize("streaming", [False, True])
def test_failed_turn_releases_guard_without_allowing_session_switch(
    monkeypatch, streaming
):
    from unittest.mock import Mock

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True, max_retry_limit=0)
    monkeypatch.setattr(
        agent.llm,
        "call",
        Mock(
            side_effect=[RuntimeError("model unavailable"), "Final Answer: Recovered"]
        ),
    )
    with pytest.raises(RuntimeError, match="model unavailable"):
        if streaming:
            with agent.as_flow().stream_turn(
                "First turn", session_id="session-a", stream=False
            ) as turn:
                list(turn)
        else:
            agent.handle_turn("First turn", session_id="session-a")
    with pytest.raises(ValueError, match="one session"):
        agent.handle_turn("Other session", session_id="session-b")
    assert agent.handle_turn("Retry").raw == "Recovered"
    assert "Other session" not in str(agent.as_flow().conversation_messages)


def test_closing_stream_keeps_guard_until_worker_finishes(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    agent = make_agent(conversational=True, max_retry_limit=0)
    started, release, closing = Event(), Event(), Event()
    call = agent.llm.call

    def blocking_call(messages, **kwargs):
        started.set()
        assert release.wait(10)
        return call(messages, **kwargs)

    monkeypatch.setattr(agent.llm, "call", blocking_call)

    def close_early():
        with agent.as_flow().stream_turn("First turn", stream=False) as turn:
            iterator = iter(turn)
            next(iterator)
            assert started.wait(10)
            closing.set()
            turn.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(close_early)
        try:
            assert closing.wait(10)
            with pytest.raises(RuntimeError, match="already running"):
                agent.handle_turn("Overlapping message")
        finally:
            release.set()
        running.result(timeout=10)
    assert agent.handle_turn("Next turn").raw == "Public answer"
    assert "Overlapping message" not in str(agent.as_flow().conversation_messages)
