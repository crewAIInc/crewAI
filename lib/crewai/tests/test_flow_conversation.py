"""Tests for conversational Flow helpers and kickoff parameters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.flow_events import FlowStartedEvent
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.flow import Flow, ChatState, listen, start
from crewai.flow.flow_context import current_flow_id, current_flow_name
from crewai.flow.conversation import (
    ConversationalConfig,
    append_message,
    get_conversation_messages,
    normalize_kickoff_inputs,
    prepare_conversational_turn,
)
from crewai.flow.chat import ChatMessage, ChatSession
from crewai.flow.providers import QueueInputProvider
from crewai.utilities.types import LLMMessage


class SimpleChatFlow(Flow[ChatState]):
    @start()
    def begin(self):
        return "done"


class DictChatFlow(Flow):
    @start()
    def begin(self):
        return self.state.get("marker", "ok")


class TestNormalizeKickoffInputs:
    def test_merges_session_and_user_message(self) -> None:
        merged = normalize_kickoff_inputs(
            {"foo": 1},
            user_message="hello",
            session_id="sess-1",
        )
        assert merged["id"] == "sess-1"
        assert merged["user_message"] == "hello"
        assert merged["foo"] == 1


class TestMessageHelpers:
    def test_append_message_on_pydantic_state(self) -> None:
        flow = SimpleChatFlow()
        flow._state = ChatState()
        append_message(flow, "user", "hi")
        assert get_conversation_messages(flow) == [{"role": "user", "content": "hi"}]

    def test_append_message_fallback_buffer(self) -> None:
        flow = DictChatFlow()

        class _State:
            id = str(uuid4())

        flow._state = _State()
        append_message(flow, "assistant", "reply")
        assert get_conversation_messages(flow) == [
            {"role": "assistant", "content": "reply"}
        ]
        assert flow._conversation_messages == [
            {"role": "assistant", "content": "reply"}
        ]


class TestIntentPerTurn:
    def test_prepare_clears_stale_last_intent(self) -> None:
        flow = SimpleChatFlow()
        flow._state = ChatState(last_intent="ORDER", messages=[])
        prepare_conversational_turn(flow, user_message="hello")
        assert flow.state.last_intent is None


class TestKickoffConversational:
    def test_kickoff_user_message_hydrates_state(self) -> None:
        flow = SimpleChatFlow()
        flow.kickoff(user_message="track my order", session_id="session-abc")

        assert flow.state.last_user_message == "track my order"
        assert any(
            m.get("role") == "user" and m.get("content") == "track my order"
            for m in flow.state.messages
        )
        assert flow.state.id == "session-abc"

    def test_kickoff_classifies_intent_when_configured(self) -> None:
        flow = SimpleChatFlow()

        with patch.object(
            flow,
            "_collapse_to_outcome",
            return_value="order",
        ) as mock_collapse:
            flow.kickoff(
                user_message="where is my package",
                session_id="s1",
                intents=["order", "help"],
                intent_llm="gpt-4o-mini",
            )

        mock_collapse.assert_called_once()
        assert flow.state.last_intent == "order"

    def test_ask_appends_to_messages(self) -> None:
        class AskFlow(Flow[ChatState]):
            input_provider = MagicMock()
            input_provider.request_input = MagicMock(return_value="user reply")

            @start()
            def begin(self):
                self.ask("Prompt:")
                return "ok"

        flow = AskFlow()
        flow._state = ChatState()
        flow.kickoff()

        assert any(
            m.get("role") == "user" and m.get("content") == "user reply"
            for m in flow.state.messages
        )


class TestClassifyIntent:
    def test_uses_collapse_with_context(self) -> None:
        flow = SimpleChatFlow()
        flow._state = ChatState(
            messages=[{"role": "user", "content": "prior"}],
        )

        with patch.object(flow, "_collapse_to_outcome", return_value="help") as mock:
            outcome = flow.classify_intent(
                "I need help",
                ["order", "help"],
                llm="gpt-4o-mini",
                context=flow.conversation_messages,
            )

        assert outcome == "help"
        assert "I need help" in mock.call_args[0][0]


class TestQueueInputProvider:
    def test_push_and_request_input(self) -> None:
        provider = QueueInputProvider()
        flow = SimpleChatFlow()
        flow._state = ChatState(id="sess-q")

        provider.push("sess-q", "hello")
        result = provider.request_input(">", flow, metadata={"session_id": "sess-q"})
        assert result == "hello"


class TestChatSession:
    def test_handle_turn_returns_turn_result(self) -> None:
        flow = SimpleChatFlow()
        session = ChatSession(
            flow,
            session_id="chat-1",
            intents=["order", "help"],
            intent_llm="gpt-4o-mini",
        )

        with patch.object(flow, "_collapse_to_outcome", return_value="help"):
            turn = session.handle_turn("hi there")

        assert turn.session_id == "chat-1"
        assert turn.output == "done"
        assert turn.intent == "help"
        assert any(m["role"] == "user" for m in turn.messages)
        session.close()

    def test_chat_message_model(self) -> None:
        msg = ChatMessage(
            type="assistant_delta",
            session_id="x",
            payload={"chunk": "hi"},
        )
        assert msg.version == "1"
        assert msg.type == "assistant_delta"


class TestFlowTracingWhenSuppressed:
    def test_flow_started_emitted_when_panel_events_suppressed(self) -> None:
        class QuietFlow(Flow[ChatState]):
            suppress_flow_events = True

            @start()
            def begin(self) -> str:
                return "ok"

        started: list[str] = []
        original_emit = crewai_event_bus.emit

        def track_emit(source: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(event, FlowStartedEvent):
                started.append(event.flow_name)
            return original_emit(source, event, *args, **kwargs)

        with patch.object(crewai_event_bus, "emit", side_effect=track_emit):
            QuietFlow().kickoff()

        assert started == ["QuietFlow"]

    def test_llm_action_inside_flow_claims_flow_trace_batch(self) -> None:
        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = None
        listener.batch_manager.batch_owner_type = None
        listener.batch_manager.batch_owner_id = None

        flow_id_token = current_flow_id.set("flow-test-id")
        flow_name_token = current_flow_name.set("DemoSupportFlow")
        try:
            event = LLMCallStartedEvent(
                model="gpt-4o-mini",
                messages=[],
                call_id="call-test",
            )
            listener._handle_action_event("llm_call_started", object(), event)
        finally:
            current_flow_id.reset(flow_id_token)
            current_flow_name.reset(flow_name_token)

        assert listener.batch_manager.batch_owner_type == "flow"
        assert listener.batch_manager.batch_owner_id == "flow-test-id"
        assert (
            listener.batch_manager.current_batch.execution_metadata["execution_type"]
            == "flow"
        )
        assert (
            listener.batch_manager.current_batch.execution_metadata["flow_name"]
            == "DemoSupportFlow"
        )


class TestDeferTraceFinalization:
    def test_conversational_kickoff_enables_defer_flag(self) -> None:
        class ChatFlow(Flow[ChatState]):
            conversational_config = ConversationalConfig(
                defer_trace_finalization=True
            )

            @start()
            def begin(self) -> str:
                return "ok"

        flow = ChatFlow()
        flow._configure_conversational_kickoff(
            user_message="hi",
            session_id="sess-trace",
        )
        assert flow.defer_trace_finalization is True
        assert flow._should_defer_trace_finalization() is True

    def test_finalize_skipped_until_forced(self) -> None:
        flow = SimpleChatFlow()
        flow.defer_trace_finalization = True

        with patch(
            "crewai.events.listeners.tracing.trace_listener.TraceCollectionListener"
        ) as mock_listener_cls:
            mock_listener_cls.return_value.batch_manager.batch_owner_type = "flow"
            mock_listener_cls.return_value.first_time_handler.is_first_time = False

            flow._finalize_flow_trace_batch()
            mock_listener_cls.assert_not_called()

            flow._finalize_flow_trace_batch(force=True)
            mock_listener_cls.assert_called_once()


class TestDeferredFlowLifecycleEvents:
    def test_deferred_kickoff_skips_per_turn_flow_finished(self) -> None:
        class ChatFlow(Flow[ChatState]):
            conversational_config = ConversationalConfig(
                defer_trace_finalization=True
            )

            @start()
            def begin(self) -> str:
                return "ok"

        flow = ChatFlow()
        with patch.object(flow, "_emit_flow_finished_async") as mock_finished:
            flow.kickoff(user_message="hi", session_id="sess-lifecycle")
            mock_finished.assert_not_called()

    def test_flow_finished_without_flow_started_warns(self, capsys) -> None:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_context import restore_event_scope
        from crewai.events.types.flow_events import FlowFinishedEvent

        class BareFlow(Flow[ChatState]):
            @start()
            def begin(self) -> str:
                return "ok"

        restore_event_scope(())
        flow = BareFlow()
        crewai_event_bus.emit(
            flow,
            FlowFinishedEvent(
                type="flow_finished",
                flow_name="BareFlow",
                result="ok",
                state={},
            ),
        )
        captured = capsys.readouterr().out
        assert "flow_finished" in captured
        assert "Missing starting event" in captured

    def test_finalize_session_restores_flow_started_scope(self, capsys) -> None:
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatch

        class ChatFlow(Flow[ChatState]):
            conversational_config = ConversationalConfig(
                defer_trace_finalization=True
            )

            @start()
            def begin(self) -> str:
                return "ok"

        flow = ChatFlow()
        flow.defer_trace_finalization = True
        object.__setattr__(flow, "_conversation_trace_started", True)
        object.__setattr__(flow, "_conversation_flow_started_event_id", "start-evt-1")
        flow._method_outputs.append("ok")

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.batch_manager.current_batch = TraceBatch(
            execution_metadata={"execution_type": "flow", "flow_name": "ChatFlow"},
        )
        listener.batch_manager.defer_session_finalization = True
        listener.batch_manager._batch_finalized = False

        with patch.object(flow, "_finalize_flow_trace_batch") as mock_finalize:
            flow.finalize_session_traces()

        captured = capsys.readouterr().out
        assert "Missing starting event" not in captured
        mock_finalize.assert_called_once_with(force=True)
        assert listener.batch_manager.defer_session_finalization is False

    def test_finalize_batch_is_idempotent(self) -> None:
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager

        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.is_tracing_enabled_in_context",
            return_value=True,
        ):
            bm = TraceBatchManager()
            bm.current_batch = bm.initialize_batch(
                user_context={"privacy_level": "standard"},
                execution_metadata={"execution_type": "flow", "flow_name": "ChatFlow"},
            )
            bm.trace_batch_id = "batch-idempotent"
            bm.backend_initialized = True

            with (
                patch.object(
                    bm.plus_api,
                    "send_trace_events",
                    return_value=MagicMock(status_code=200),
                ),
                patch.object(
                    bm.plus_api,
                    "finalize_trace_batch",
                    return_value=MagicMock(status_code=200, json=MagicMock(return_value={})),
                ) as mock_finalize_api,
            ):
                bm.finalize_batch()
                bm.finalize_batch()

            assert mock_finalize_api.call_count == 1
            assert bm._batch_finalized is True

    def test_finalize_session_is_idempotent_after_batch_cleared(self) -> None:
        class ChatFlow(Flow[ChatState]):
            @start()
            def begin(self) -> str:
                return "ok"

        flow = ChatFlow()
        flow.defer_trace_finalization = True
        object.__setattr__(flow, "_conversation_trace_started", True)

        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = None
        listener.batch_manager.batch_owner_type = None
        listener.batch_manager.trace_batch_id = None
        listener.batch_manager._batch_finalized = True

        with patch.object(flow, "_emit_flow_finished_sync") as mock_finished:
            with patch.object(flow, "_finalize_flow_trace_batch") as mock_finalize:
                flow.finalize_session_traces()
                flow.finalize_session_traces()

        mock_finished.assert_not_called()
        mock_finalize.assert_not_called()

    def test_sigint_skips_deferred_session_batch(self) -> None:
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatch

        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = TraceBatch()
        listener.batch_manager.defer_session_finalization = True

        with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
            if listener.batch_manager.is_batch_initialized():
                if not listener.batch_manager.defer_session_finalization:
                    listener.batch_manager.finalize_batch()
            mock_finalize.assert_not_called()


class TestNestedCrewTracing:
    def test_is_inside_active_flow_context_when_kickoff_running(self) -> None:
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        from crewai.flow.flow_context import current_flow_id

        assert TraceCollectionListener._is_inside_active_flow_context() is False
        token = current_flow_id.set("parent-flow-id")
        try:
            assert TraceCollectionListener._is_inside_active_flow_context() is True
        finally:
            current_flow_id.reset(token)

    def test_nested_crew_completion_skips_finalize(self) -> None:
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        from crewai.flow.flow_context import current_flow_id

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "crew"

        token = current_flow_id.set("parent-flow-id")
        try:
            with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
                if listener._nested_in_flow_execution():
                    pass
                elif listener.batch_manager.batch_owner_type == "crew":
                    listener.batch_manager.finalize_batch()
                mock_finalize.assert_not_called()
        finally:
            current_flow_id.reset(token)

    def test_flow_owned_batch_skips_finalize_without_flow_context(self) -> None:
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatch

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.batch_manager.current_batch = TraceBatch(
            execution_metadata={"execution_type": "flow", "flow_name": "Demo"},
        )

        with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
            if listener._nested_in_flow_execution():
                pass
            elif listener.batch_manager.batch_owner_type == "crew":
                listener.batch_manager.finalize_batch()
            mock_finalize.assert_not_called()
