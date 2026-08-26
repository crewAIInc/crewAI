"""Tests for conversational Flow helpers and kickoff parameters."""

from __future__ import annotations

import logging
import sys
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pathlib import Path
import yaml

from pydantic import BaseModel, ValidationError, create_model

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.flow_events import (
    ConversationMessageAddedEvent,
    ConversationRouteSelectedEvent,
    ConversationTurnCompletedEvent,
    ConversationTurnFailedEvent,
    ConversationTurnStartedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import LLMCallStartedEvent, LLMStreamChunkEvent
from crewai.experimental import (
    ConversationConfig,
    ConversationMessage,
    ConversationState,
    RouterConfig,
)
from crewai.flow import Flow, ChatState, listen, start
from crewai.flow.persistence import SQLiteFlowPersistence, persist
from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow_context import (
    current_flow_defer_trace_finalization,
    current_flow_id,
    current_flow_name,
)
from crewai.llms.base_llm import BaseLLM
from crewai.flow.conversation import (
    append_message,
    get_conversation_messages,
    normalize_kickoff_inputs,
    prepare_conversational_turn,
)

class ConversationalFlow(Flow[ConversationState]):
    """Test base: a ``Flow[ConversationState]`` with conversational mode enabled.

    Mirrors the documented ``class MyChat(Flow): conversational = True`` pattern
    so the conversational subclasses below stay terse.
    """

    conversational = True


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


class TestConversationalFlow:
    def test_stream_turn_emits_ordered_conversation_frames(self) -> None:
        flow = ConversationalFlow()
        flow.stream = True
        stream_values_seen_by_kickoff: list[bool] = []

        def kickoff_side_effect(*_: Any, **__: Any) -> str:
            stream_values_seen_by_kickoff.append(flow.stream)
            crewai_event_bus.emit(
                flow,
                LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk="pong",
                    call_id="call-1",
                ),
            )
            return "pong"

        with patch.object(flow, "kickoff", side_effect=kickoff_side_effect):
            stream = flow.stream_turn("ping", session_id="session-1")

            with pytest.raises(RuntimeError, match="Streaming has not completed yet"):
                _ = stream.result

            frames = list(stream.events)

        assert stream.result == "pong"
        assert stream_values_seen_by_kickoff == [False]
        assert flow.stream is True
        assert [frame.seq for frame in frames] == sorted(frame.seq for frame in frames)
        assert [frame.type for frame in frames] == [
            "conversation_turn_started",
            "llm_stream_chunk",
            "conversation_message_added",
            "conversation_turn_completed",
        ]
        assert [frame.channel for frame in frames] == [
            "flow",
            "llm",
            "messages",
            "flow",
        ]
        assert frames[1].data["chunk"] == "pong"
        assert flow.state.messages[-1].content == "pong"

    def test_stream_turn_enables_streaming_on_conversation_llm(self) -> None:
        class FakeLLM(BaseLLM):
            stream_values: ClassVar[list[bool | None]] = []

            def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
                self.stream_values.append(self._effective_stream())
                for chunk in ("po", "ng"):
                    crewai_event_bus.emit(
                        flow,
                        LLMStreamChunkEvent(
                            type="llm_stream_chunk",
                            chunk=chunk,
                            call_id="call-1",
                        ),
                    )
                return "pong"

        FakeLLM.stream_values = []
        llm = FakeLLM(model="gpt-4o-mini", stream=False)

        @ConversationConfig(llm=llm)
        class StreamingChatFlow(ConversationalFlow):
            pass

        flow = StreamingChatFlow()
        stream = flow.stream_turn("ping", session_id="session-1")
        frames = list(stream.events)

        assert stream.result == "pong"
        assert llm.stream_values == [True]
        assert llm.stream is False
        assert [
            frame.data["chunk"]
            for frame in frames
            if frame.type == "llm_stream_chunk"
        ] == ["po", "ng"]

    def test_stream_turn_returns_pending_feedback_without_failure_event(self) -> None:
        flow = ConversationalFlow()
        pending = HumanFeedbackPending(
            context=PendingFeedbackContext(
                flow_id="session-1",
                flow_class="tests.PendingFeedbackFlow",
                method_name="review",
                method_output="draft",
                message="Please review",
            )
        )

        def kickoff_side_effect(*_: Any, **__: Any) -> None:
            raise pending

        with patch.object(flow, "kickoff", side_effect=kickoff_side_effect):
            stream = flow.stream_turn("review this", session_id="session-1")
            frames = list(stream.events)

        assert stream.result is pending
        assert [frame.type for frame in frames] == ["conversation_turn_started"]

    def test_deferred_multi_turn_emits_single_flow_finished(self) -> None:
        """A deferred multi-turn session lands as one trace: exactly one
        ``FlowFinishedEvent`` is emitted at ``finalize_session_traces()``, not
        one per turn. (Each turn still opens its own ``flow_started``.)
        """
        from crewai.events.types.flow_events import FlowFinishedEvent

        @ConversationConfig(defer_trace_finalization=True)
        class TraceFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                reply = f"worked: {self.state.current_user_message}"
                self.append_assistant_message(reply)
                return reply

        flow = TraceFlow()
        finished: list[FlowFinishedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture(_: Any, event: FlowFinishedEvent) -> None:
                finished.append(event)

            flow.handle_turn("research apple stock")
            flow.handle_turn("research google stock")
            crewai_event_bus.flush()
            assert finished == [], "deferred turns must not emit per-turn flow_finished"

            flow.finalize_session_traces()
            crewai_event_bus.flush()

        assert len(finished) == 1, (
            "a deferred session must emit exactly one flow_finished at finalize"
        )


    def test_handle_turn_routes_to_listener_and_records_public_result(self) -> None:
        @ConversationConfig(default_intents=["research"], intent_llm="gpt-4o-mini")
        class ResearchFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_agent_result(
                    "researcher",
                    "researched answer",
                    visibility="public",
                )
                return "researched answer"

        flow = ResearchFlow()

        with patch.object(flow, "_collapse_to_outcome", return_value="research"):
            result = flow.handle_turn("research CrewAI")

        assert result == "researched answer"
        assert flow.state.current_user_message == "research CrewAI"
        assert flow.state.last_intent == "research"
        assert [message.role for message in flow.state.messages] == [
            "user",
            "assistant",
        ]
        assert flow.state.messages[-1].content == "researched answer"
        assert flow.state.events[0].agent_name == "researcher"
        assert flow.state.events[0].visibility == "public"

    def test_private_agent_results_stay_out_of_shared_history(self) -> None:
        class PrivateFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> None:
                self.append_agent_result("planner", "private scratch")

        flow = PrivateFlow()
        flow.handle_turn("plan quietly")

        assert [message.role for message in flow.state.messages] == ["user"]
        assert flow.state.events[0].visibility == "private"
        assert flow.state.agent_threads["planner"][0].content == "private scratch"

    def test_answer_from_history_uses_configured_llm_and_appends_reply(self) -> None:
        with pytest.warns(
            DeprecationWarning,
            match="answer_from_history_prompt.*answer_from_history_llm",
        ) as warning_records:
            @ConversationConfig(answer_from_history_llm="gpt-4o-mini")
            class HistoryFlow(ConversationalFlow):
                pass

        assert warning_records[0].filename == __file__

        flow = HistoryFlow()
        flow._state = ConversationState(
            messages=[
                ConversationMessage(role="user", content="research topic"),
                ConversationMessage(role="assistant", content="prior findings"),
            ]
        )
        llm = MagicMock()
        llm.call.return_value = "summary from history"

        with (
            patch.object(
                flow,
                "_collapse_to_outcome",
                return_value="answer_from_history",
            ),
            patch.object(flow, "_coerce_llm", return_value=llm),
        ):
            result = flow.handle_turn("summarize this")

        assert result == "summary from history"
        assert flow.state.messages[-1].role == "assistant"
        assert flow.state.messages[-1].content == "summary from history"
        llm.call.assert_called_once()

    def test_router_config_uses_structured_intent_response(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "clarify"]

        llm = MagicMock()
        llm.call.return_value = ResearchRoute(intent="research")

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify the next action.",
                response_format=ResearchRoute,
                llm=llm,
                routes=["research", "clarify"],
                default_intent="clarify",
                fallback_intent="clarify",
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_assistant_message("researched")
                return "researched"

            @listen("clarify")
            def ask_clarification(self) -> str:
                self.append_assistant_message("clarify")
                return "clarify"

        flow = RoutedFlow()
        result = flow.handle_turn("research CrewAI")

        assert result == "researched"
        llm.call.assert_called_once()
        assert llm.call.call_args.kwargs["response_format"] is ResearchRoute
        assert flow.state.messages[-1].content == "researched"

    def test_router_config_falls_back_for_invalid_intent(self) -> None:
        class ResearchRoute(BaseModel):
            intent: str

        llm = MagicMock()
        llm.call.return_value = ResearchRoute(intent="unknown")

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify the next action.",
                response_format=ResearchRoute,
                llm=llm,
                routes=["research", "clarify"],
                default_intent="clarify",
                fallback_intent="clarify",
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_assistant_message("researched")
                return "researched"

            @listen("clarify")
            def ask_clarification(self) -> str:
                self.append_assistant_message("clarify")
                return "clarify"

        flow = RoutedFlow()
        result = flow.handle_turn("something vague")

        assert result == "clarify"
        assert flow.state.messages[-1].content == "clarify"

    def test_router_effective_routes_include_builtins(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify.",
                response_format=ResearchRoute,
                routes=["research"],
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                return "researched"

        flow = RoutedFlow()

        assert flow._effective_routes(flow.conversational_config.router) == {
            "research",
            "converse",
            "end",
        }

    def test_router_infers_custom_routes_without_internal_routes(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify.",
                response_format=ResearchRoute,
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                return "researched"

        flow = RoutedFlow()

        assert flow._effective_routes(flow.conversational_config.router) == {
            "research",
            "converse",
            "end",
        }

    def test_router_config_uses_conversational_defaults(self) -> None:
        llm = MagicMock()

        @ConversationConfig(
            llm=llm,
            router=RouterConfig(),
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_assistant_message("researched")
                return "researched"

        flow = RoutedFlow()
        response_format = flow._router_response_format(flow.conversational_config.router)
        llm.call.return_value = response_format(intent="research")

        result = flow.handle_turn("research CrewAI")

        assert result == "researched"
        llm.call.assert_called_once()
        assert llm.call.call_args.kwargs["response_format"].__name__ == (
            "ConversationRoute"
        )
        assert flow.state.messages[-1].content == "researched"

    def test_builtin_converse_appends_assistant_message_and_uses_history(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = ResearchRoute(intent="converse")
        chat_llm = MagicMock()
        chat_llm.call.return_value = "summary from built-in converse"

        @ConversationConfig(
            system_prompt="You are a helpful research assistant.",
            llm=chat_llm,
            router=RouterConfig(
                prompt="Classify.",
                response_format=ResearchRoute,
                llm=router_llm,
                routes=["research"],
                default_intent="converse",
            ),
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_agent_result(
                    "researcher",
                    "prior findings",
                    visibility="public",
                )
                return "prior findings"

        flow = RoutedFlow()
        flow.state.messages = [
            ConversationMessage(role="user", content="research CrewAI"),
            ConversationMessage(role="assistant", content="prior findings"),
        ]
        result = flow.handle_turn("summarize findings")

        assert result == "summary from built-in converse"
        assert flow.state.messages[-1].content == "summary from built-in converse"
        messages = chat_llm.call.call_args.kwargs["messages"]
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful research assistant.",
        }
        assert any(message["content"] == "prior findings" for message in messages)
        assert any(message["content"] == "summarize findings" for message in messages)

    def test_conversational_turn_emits_message_and_route_events(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = ResearchRoute(intent="converse")
        chat_llm = MagicMock()
        chat_llm.call.return_value = "hello back"

        @ConversationConfig(
            llm=chat_llm,
            router=RouterConfig(
                response_format=ResearchRoute,
                llm=router_llm,
                routes=["research"],
            ),
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_assistant_message("researched")
                return "researched"

        messages: list[ConversationMessageAddedEvent] = []
        routes: list[ConversationRouteSelectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ConversationMessageAddedEvent)
            def capture_message(_: Any, event: ConversationMessageAddedEvent) -> None:
                messages.append(event)

            @crewai_event_bus.on(ConversationRouteSelectedEvent)
            def capture_route(_: Any, event: ConversationRouteSelectedEvent) -> None:
                routes.append(event)

            flow = RoutedFlow()
            flow.handle_turn("just chat")
            crewai_event_bus.flush()

        assert [(event.role, event.content) for event in messages] == [
            ("user", "just chat"),
            ("assistant", "hello back"),
        ]
        assert [event.message_index for event in messages] == [0, 1]
        assert len(routes) == 1
        assert routes[0].route == "converse"
        assert routes[0].user_message == "just chat"
        assert routes[0].session_id == messages[0].session_id

    def test_builtin_end_marks_conversation_ended(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = ResearchRoute(intent="end")

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify.",
                response_format=ResearchRoute,
                llm=router_llm,
                routes=["research"],
                default_intent="converse",
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                return "researched"

        flow = RoutedFlow()
        result = flow.handle_turn("bye")

        assert result == "Conversation ended."
        assert flow.state.ended is True
        assert flow.state.messages[-1].content == "Conversation ended."

    def test_router_auto_enables_when_custom_routes_declared_and_no_explicit_config(
        self,
    ) -> None:
        """``ConversationConfig(llm=...)`` alone wires LLM routing for custom listeners.

        Users shouldn't have to pass ``router=RouterConfig()`` just to flip
        the router on — declaring custom ``@listen`` handlers + giving the
        config an LLM is sufficient. Only opt out by setting
        ``default_intents`` (legacy path).
        """

        class Route(BaseModel):
            intent: Literal["INTERNET_SEARCH", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = Route(intent="INTERNET_SEARCH")

        @ConversationConfig(llm=router_llm)  # no router= here
        class AutoEnabledFlow(ConversationalFlow):
            @listen("INTERNET_SEARCH")
            def handle_search(self) -> str:
                """Fresh web research."""
                self.append_assistant_message("searched")
                return "searched"

        flow = AutoEnabledFlow()
        result = flow.handle_turn("research today's AI news")

        assert result == "searched"
        # Router LLM should have been invoked.
        assert router_llm.call.call_count >= 1

    def test_router_auto_enable_skipped_when_only_builtin_routes(self) -> None:
        """No custom routes → no auto-enable; falls through to converse."""

        chat_llm = MagicMock()
        chat_llm.call.return_value = "hi there"

        @ConversationConfig(llm=chat_llm)
        class NoCustomFlow(ConversationalFlow):
            pass

        flow = NoCustomFlow()
        flow.handle_turn("hello")

        assert flow.state.last_intent == "converse"
        # chat_llm was used by converse_turn, not as a router.
        assert chat_llm.call.call_count == 1

    def test_router_auto_enable_skipped_when_default_intents_set(self) -> None:
        """Legacy ``default_intents`` opts out of router auto-enable."""

        @ConversationConfig(default_intents=["search"], intent_llm="gpt-4o-mini")
        class LegacyFlow(ConversationalFlow):
            @listen("search")
            def handle_search(self) -> str:
                """Web research."""
                self.append_assistant_message("legacy-searched")
                return "legacy-searched"

        flow = LegacyFlow()
        with patch.object(flow, "_collapse_to_outcome", return_value="search"):
            result = flow.handle_turn("look it up")

        # Legacy path set state.last_intent via classify_intent; auto-router did NOT
        # overwrite it because default_intents short-circuits the auto-enable.
        assert result == "legacy-searched"
        assert flow.state.last_intent == "search"

    def test_user_start_methods_run_sequentially_before_router_in_conversational_mode(
        self,
    ) -> None:
        """Conversational flows: user ``@start`` methods finish before router fires.

        Non-chat flows run ``@start`` methods in parallel via ``asyncio.gather``,
        which would race with ``route_conversation`` and let the router fire
        before user setup finished. In conversational mode the framework runs
        them sequentially, with ``route_conversation`` last.
        """
        order: list[str] = []

        @ConversationConfig()
        class BootstrapFlow(ConversationalFlow):
            @start()
            def load_profile(self) -> None:
                if not self.state.session_ready:
                    order.append("load_profile")
                    self.state.session_ready = True

            @start()
            def attach_bus(self) -> None:
                order.append("attach_bus")

            def route_turn(self, context: dict[str, Any]) -> str | None:
                order.append("route_turn")
                return "work"

            @listen("work")
            def do_work(self) -> str:
                order.append("do_work")
                self.append_assistant_message("worked")
                return "worked"

        flow = BootstrapFlow()
        flow.handle_turn("turn 1")

        # Both user @start methods complete before route_turn fires.
        load_idx = order.index("load_profile")
        attach_idx = order.index("attach_bus")
        route_idx = order.index("route_turn")
        assert load_idx < route_idx
        assert attach_idx < route_idx

        # Bootstrap gate works: load_profile only fires on the first turn.
        order.clear()
        flow.handle_turn("turn 2")
        assert "load_profile" not in order
        assert "attach_bus" in order  # still fires every turn
        assert "route_turn" in order

    def test_subclass_can_override_conversation_start_helper(
        self,
    ) -> None:
        """The compatibility helper remains overridable without adding a Flow node."""

        bootstrap_calls: list[str] = []

        @ConversationConfig()
        class BootstrapFlow(ConversationalFlow):
            def conversation_start(self) -> str | None:
                bootstrap_calls.append("ran")
                return super().conversation_start()

            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = BootstrapFlow()
        flow.handle_turn("hi")

        assert bootstrap_calls == ["ran"]
        assert "conversation_start" not in BootstrapFlow.flow_definition().methods
        route_definition = BootstrapFlow.flow_definition().methods["route_conversation"]
        assert route_definition.start is True
        assert route_definition.router is True
        assert flow.state.messages[-1].content == "worked"

    def test_legacy_decorated_conversation_start_runs_once_per_turn(
        self,
    ) -> None:
        """Legacy ``@start`` overrides are not invoked again by the router."""

        bootstrap_calls: list[str] = []

        @ConversationConfig()
        class BootstrapFlow(ConversationalFlow):
            @start()
            def conversation_start(self) -> str | None:
                bootstrap_calls.append("ran")
                return super().conversation_start()

            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = BootstrapFlow()
        flow.handle_turn("hi")

        assert bootstrap_calls == ["ran"]
        assert flow.state.messages[-1].content == "worked"

    def test_handle_turn_reruns_graph_after_prior_turn_completed(self) -> None:
        """Multi-turn must not flip ``_is_execution_resuming`` and short-circuit.

        ``Flow.kickoff`` with persistence enabled treats ``inputs={"id": ...}``
        as a checkpoint restore, so it skips clearing ``_completed_methods``.
        Without ``ConversationalFlow.kickoff`` resetting that state, turn 2+
        sees every method as already-completed, short-circuits to
        ``_method_outputs[-1]``, and returns the previous turn's output.
        """

        class Route(BaseModel):
            intent: Literal["RESEARCH", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.side_effect = [
            Route(intent="converse"),
            Route(intent="RESEARCH"),
        ]
        chat_llm = MagicMock()
        chat_llm.call.return_value = "general help"

        @ConversationConfig(
            llm=chat_llm,
            router=RouterConfig(
                response_format=Route,
                llm=router_llm,
                routes=["RESEARCH"],
            ),
        )
        class DemoFlow(ConversationalFlow):
            @listen("RESEARCH")
            def handle_research(self) -> str:
                self.append_assistant_message("fresh research")
                return "fresh research"

        flow = DemoFlow()
        from crewai.flow.persistence import SQLiteFlowPersistence

        import tempfile
        from pathlib import Path

        flow.persistence = SQLiteFlowPersistence(
            str(Path(tempfile.mkdtemp()) / "regression.db")
        )

        out1 = flow.handle_turn("tell me what you can do")
        out2 = flow.handle_turn("now do research")

        assert out1 == "general help"
        assert out2 == "fresh research"
        assert chat_llm.call.call_count == 1
        assert router_llm.call.call_count == 2
        assert flow.state.messages[-1].content == "fresh research"
        assert flow._is_execution_resuming is False

    def test_route_catalog_combines_docstrings_builtins_and_overrides(self) -> None:
        """Catalog precedence: route_descriptions > built-in > docstring."""

        @ConversationConfig(
            router=RouterConfig(
                routes=["RESEARCH", "ORDER"],
                route_descriptions={"ORDER": "explicit override for order route"},
            )
        )
        class CatalogFlow(ConversationalFlow):
            @listen("RESEARCH")
            def handle_research(self) -> str:
                """Fresh web research, current news, real-time lookups."""
                return "researched"

            @listen("ORDER")
            def handle_order(self) -> str:
                """This docstring should NOT win — override takes priority."""
                return "ordered"

        flow = CatalogFlow()
        catalog = flow._build_route_catalog(flow.conversational_config.router)

        assert catalog["RESEARCH"] == (
            "Fresh web research, current news, real-time lookups."
        )
        assert catalog["ORDER"] == "explicit override for order route"
        # Built-in routes get framework-canned descriptions.
        assert "Ordinary chat" in catalog["converse"]
        assert "finished" in catalog["end"]

    def test_route_catalog_falls_back_to_empty_when_no_docstring(self) -> None:
        @ConversationConfig(router=RouterConfig(routes=["BARE"]))
        class BareFlow(ConversationalFlow):
            @listen("BARE")
            def handle_bare(self) -> str:
                return "bare"

        flow = BareFlow()
        catalog = flow._build_route_catalog(flow.conversational_config.router)

        assert catalog["BARE"] == ""

    def test_router_messages_include_route_catalog(self) -> None:
        """The router system prompt must enumerate routes with descriptions."""

        class Route(BaseModel):
            intent: Literal["RESEARCH", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = Route(intent="RESEARCH")

        @ConversationConfig(
            router=RouterConfig(
                prompt="A research-focused assistant.",
                response_format=Route,
                llm=router_llm,
                routes=["RESEARCH"],
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("RESEARCH")
            def handle_research(self) -> str:
                """Fresh web research and current news."""
                self.append_assistant_message("researched")
                return "researched"

        flow = RoutedFlow()
        flow.handle_turn("research today's AI news")

        system_message = router_llm.call.call_args.kwargs["messages"][0]["content"]
        assert "Routes:" in system_message
        assert "- RESEARCH: Fresh web research and current news." in system_message
        assert "- converse: Ordinary chat" in system_message
        assert system_message.startswith("A research-focused assistant.")

    def test_router_decision_persists_last_intent_and_passes_it_next_turn(
        self,
    ) -> None:
        """Router must record its decision so the next turn's router LLM sees it."""

        class Route(BaseModel):
            intent: Literal["research", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.side_effect = [
            Route(intent="research"),
            Route(intent="converse"),
        ]
        chat_llm = MagicMock()
        chat_llm.call.return_value = "follow-up reply"

        @ConversationConfig(
            llm=chat_llm,
            router=RouterConfig(
                response_format=Route,
                llm=router_llm,
                routes=["research"],
            ),
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_assistant_message("researched")
                return "researched"

        flow = RoutedFlow()

        flow.handle_turn("research CrewAI")
        assert flow.state.last_intent == "research"

        flow.handle_turn("tell me more about that")
        assert flow.state.last_intent == "converse"

        # Turn 2's router LLM must have seen last_intent='research' in its context.
        second_call_user_content = router_llm.call.call_args_list[1].kwargs["messages"][1][
            "content"
        ]
        assert '"last_intent": "research"' in second_call_user_content

    def test_custom_route_still_runs_with_builtin_routes(self) -> None:
        class ResearchRoute(BaseModel):
            intent: Literal["research", "converse", "end"]

        router_llm = MagicMock()
        router_llm.call.return_value = ResearchRoute(intent="research")

        @ConversationConfig(
            router=RouterConfig(
                prompt="Classify.",
                response_format=ResearchRoute,
                llm=router_llm,
                routes=["research"],
                default_intent="converse",
            )
        )
        class RoutedFlow(ConversationalFlow):
            @listen("research")
            def run_research(self) -> str:
                self.append_agent_result("researcher", "researched", visibility="public")
                return "researched"

        flow = RoutedFlow()
        result = flow.handle_turn("research CrewAI")

        assert result == "researched"
        assert flow.state.messages[-1].content == "researched"

    def test_conversational_flow_auto_defaults_to_conversation_state(self) -> None:
        """``class C(Flow): conversational = True`` resolves state to ConversationState.

        Pins the auto-default in ``_create_initial_state``: when the user opts
        into conversational mode without an explicit ``Flow[...]`` type
        parameter or ``initial_state``, state is a ``ConversationState`` with
        the chat-shaped fields ready to use.
        """

        class BareChat(Flow):
            conversational = True

        flow = BareChat()
        assert isinstance(flow._state, ConversationState)
        assert flow.state.messages == []
        assert flow.state.current_user_message is None
        assert flow.state.session_ready is False

    def test_mixin_handle_turn_resolves_on_flow_subclass(self) -> None:
        """``Flow`` mixes in ``_ConversationalMixin`` — opt-in subclasses get its methods.

        The conversational graph + ``handle_turn`` live on the mixin in
        ``crewai.experimental.conversational_mixin``; this test confirms
        MRO resolution wires them onto a ``Flow`` subclass that opts in.
        """
        from crewai.experimental.conversational_mixin import _ConversationalMixin

        @ConversationConfig()
        class MyChat(Flow):
            conversational = True

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

        flow = MyChat()
        assert isinstance(flow, _ConversationalMixin)
        assert callable(getattr(flow, "handle_turn", None))
        assert callable(getattr(flow, "finalize_session_traces", None))
        assert callable(getattr(flow, "append_assistant_message", None))

        # Driving the mixin's handle_turn through to the listener proves
        # the wiring is end-to-end, not just attribute presence.
        flow.handle_turn("anything")
        assert flow.state.messages[-1].content == "worked"

    def test_chat_runs_repl_over_handle_turn_and_finalizes(self) -> None:
        @ConversationConfig(defer_trace_finalization=False)
        class MyChat(ConversationalFlow):
            turns: int = 0

            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.turns += 1
                reply = f"worked: {self.state.current_user_message}"
                self.append_assistant_message(reply)
                return reply

        flow = MyChat()
        inputs = iter(["first", "", "second", "quit"])
        prompts: list[str] = []
        outputs: list[str] = []

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return next(inputs)

        with patch.object(flow, "finalize_session_traces") as mock_finalize:
            flow.chat(
                session_id="session-1",
                input_fn=input_fn,
                output_fn=outputs.append,
            )

        assert flow.turns == 2
        assert prompts == ["\nYou: ", "\nYou: ", "\nYou: ", "\nYou: "]
        assert outputs == [
            "\nAssistant: worked: first",
            "\nAssistant: worked: second",
        ]
        mock_finalize.assert_called_once_with()
        assert flow.defer_trace_finalization is False

    def test_chat_stringifies_repl_output_like_conversation_helpers(self) -> None:
        class RawResult:
            raw = "raw assistant output"

        @ConversationConfig(defer_trace_finalization=False)
        class MyChat(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> RawResult:
                return RawResult()

        flow = MyChat()
        inputs = iter(["first", "quit"])
        outputs: list[str] = []

        with patch.object(flow, "finalize_session_traces"):
            flow.chat(
                input_fn=lambda _: next(inputs),
                output_fn=outputs.append,
            )

        assert outputs == ["\nAssistant: raw assistant output"]

    def test_chat_rejects_non_conversational_flows(self) -> None:
        class PlainFlow(Flow):
            @start()
            def begin(self) -> str:
                return "done"

        flow = PlainFlow()

        try:
            flow.chat(input_fn=lambda _: "quit")
        except ValueError as exc:
            assert "conversational flows" in str(exc)
        else:
            raise AssertionError("Flow.chat() should reject regular flows")

    def test_defer_trace_finalization_skips_per_turn_finalize(self) -> None:
        """``defer_trace_finalization = True`` suppresses per-turn ``finalize_batch``.

        Without deferral, each ``handle_turn()`` ends with a trace-batch
        finalize. With deferral on, the framework defers until
        ``finalize_session_traces()`` is called at session end.
        """

        @ConversationConfig()
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        flow.defer_trace_finalization = True

        listener = TraceCollectionListener()
        with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
            flow.handle_turn("turn 1")
            flow.handle_turn("turn 2")
            flow.handle_turn("turn 3")

        assert mock_finalize.call_count == 0, (
            "defer_trace_finalization=True must skip per-turn finalize"
        )

    def test_deferred_conversation_emits_one_flow_started(self) -> None:
        """Deferred conversational sessions emit one flow_started for the session."""
        from crewai.events.types.flow_events import FlowStartedEvent

        @ConversationConfig(defer_trace_finalization=True)
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        observed_events: list[str] = []
        started_events: list[FlowStartedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowStartedEvent)
            def capture(_: Any, event: FlowStartedEvent) -> None:
                observed_events.append(event.type)
                started_events.append(event)

            @crewai_event_bus.on(ConversationMessageAddedEvent)
            def capture_message(
                _: Any, event: ConversationMessageAddedEvent
            ) -> None:
                if event.role == "user":
                    observed_events.append(event.type)

            flow.handle_turn("turn 1")
            flow.handle_turn("turn 2")
            flow.handle_turn("turn 3")
            crewai_event_bus.flush()

        assert len(started_events) == 1, (
            "deferred conversational traces should emit one session-level "
            "flow_started event, not one per turn"
        )
        assert observed_events[0] == "flow_started"
        assert observed_events[1] == "conversation_message_added"

    def test_handle_turn_emits_started_and_completed_for_each_conversational_turn(
        self,
    ) -> None:
        """Each ``handle_turn()`` emits paired turn lifecycle events."""

        @ConversationConfig(defer_trace_finalization=True)
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        default_session_id = flow.state.id
        turn_events: list[
            ConversationTurnStartedEvent | ConversationTurnCompletedEvent
        ] = []

        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(
                event, (ConversationTurnStartedEvent, ConversationTurnCompletedEvent)
            ):
                turn_events.append(event)
            return original_emit(source, event)

        with patch.object(crewai_event_bus, "emit", side_effect=capture_emit):
            flow.handle_turn("turn 1")
            flow.handle_turn("turn 2", session_id="custom-session")
            crewai_event_bus.flush()

        assert [event.type for event in turn_events] == [
            "conversation_turn_started",
            "conversation_turn_completed",
            "conversation_turn_started",
            "conversation_turn_completed",
        ]
        assert turn_events[0].session_id == default_session_id
        assert turn_events[1].session_id == default_session_id
        assert turn_events[2].session_id == "custom-session"
        assert turn_events[3].session_id == "custom-session"

    def test_handle_turn_emits_failed_instead_of_completed_when_turn_raises(
        self,
    ) -> None:
        """Failed turns emit a terminal failure event without completion."""

        @ConversationConfig(defer_trace_finalization=True)
        class FailingFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                raise RuntimeError("turn exploded")

        flow = FailingFlow()
        turn_events: list[
            ConversationTurnStartedEvent
            | ConversationTurnCompletedEvent
            | ConversationTurnFailedEvent
        ] = []
        handled_failed_events: list[ConversationTurnFailedEvent] = []
        original_emit = crewai_event_bus.emit

        def capture_emit(source: Any, event: Any) -> Any:
            if isinstance(
                event,
                (
                    ConversationTurnStartedEvent,
                    ConversationTurnCompletedEvent,
                    ConversationTurnFailedEvent,
                ),
            ):
                turn_events.append(event)
            return original_emit(source, event)

        with (
            crewai_event_bus.scoped_handlers(),
            patch.object(crewai_event_bus, "emit", side_effect=capture_emit),
        ):

            @crewai_event_bus.on(ConversationTurnFailedEvent)
            def capture_failed(
                _: Any, event: ConversationTurnFailedEvent
            ) -> None:
                handled_failed_events.append(event)

            with pytest.raises(RuntimeError, match="turn exploded"):
                flow.handle_turn("turn 1")

        assert [event.type for event in turn_events] == [
            "conversation_turn_started",
            "conversation_turn_failed",
        ]
        assert turn_events[0].session_id == flow.state.id
        failed_event = turn_events[1]
        assert isinstance(failed_event, ConversationTurnFailedEvent)
        assert failed_event.session_id == flow.state.id
        assert str(failed_event.error) == "turn exploded"
        assert handled_failed_events == [failed_event]

    def test_conversation_turn_completed_tracks_feature_usage(self) -> None:
        """Completed conversation turns count conversational Flow usage."""
        from crewai.events.event_listener import event_listener

        @ConversationConfig(defer_trace_finalization=True)
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()

        with (
            crewai_event_bus.scoped_handlers(),
            patch.object(
                event_listener._telemetry,
                "feature_usage_span",
            ) as feature_usage_span,
        ):
            event_listener.setup_listeners(crewai_event_bus)
            flow.handle_turn("turn 1")

        feature_usage_span.assert_any_call("flow:conversation_turn")

    def test_route_event_uses_no_message_index_for_empty_transcript(self) -> None:
        """Route events do not reference index zero when no message exists."""

        @ConversationConfig()
        class DeferredFlow(ConversationalFlow):
            pass

        flow = DeferredFlow()
        route_events: list[ConversationRouteSelectedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ConversationRouteSelectedEvent)
            def capture(_: Any, event: ConversationRouteSelectedEvent) -> None:
                route_events.append(event)

            flow._emit_conversation_route_selected("converse")
            crewai_event_bus.flush()

        assert len(route_events) == 1
        assert route_events[0].message_index is None

    def test_finalize_session_traces_emits_finished_and_finalizes_batch(self) -> None:
        """``finalize_session_traces()`` emits one ``FlowFinishedEvent`` + one ``finalize_batch``.

        Pairs with the deferral above: after N turns with deferral on, a
        single ``finalize_session_traces()`` closes the whole session as
        one trace batch with one terminal event.
        """
        from crewai.events.types.flow_events import FlowFinishedEvent

        @ConversationConfig()
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        flow.defer_trace_finalization = True

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.first_time_handler.is_first_time = False

        finished_events: list[FlowFinishedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture(_: Any, event: FlowFinishedEvent) -> None:
                finished_events.append(event)

            with patch.object(
                listener.batch_manager, "finalize_batch"
            ) as mock_finalize:
                flow.handle_turn("turn 1")
                crewai_event_bus.flush()
                flow.handle_turn("turn 2")
                crewai_event_bus.flush()
                # No flow_finished or finalize_batch yet — deferred.
                assert finished_events == []
                assert mock_finalize.call_count == 0

                flow.finalize_session_traces()
                crewai_event_bus.flush()

                assert len(finished_events) == 1, (
                    "finalize_session_traces must emit exactly one FlowFinishedEvent"
                )
                assert mock_finalize.call_count == 1, (
                    "finalize_session_traces must finalize the trace batch once"
                )

    def test_deferred_resume_skips_per_resume_flow_finished_event(self) -> None:
        """Deferred sessions do not emit terminal events while resuming."""
        from crewai.events.types.flow_events import FlowFinishedEvent
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        class DeferredResumeFlow(Flow[ChatState]):
            defer_trace_finalization = True

            @start()
            def begin(self) -> str:
                return "started"

        flow = DeferredResumeFlow()
        flow._pending_feedback_context = PendingFeedbackContext(
            flow_id=flow.flow_id,
            flow_class="DeferredResumeFlow",
            method_name="begin",
            method_output="started",
            message="Review",
        )

        finished_events: list[FlowFinishedEvent] = []

        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture(_: Any, event: FlowFinishedEvent) -> None:
                finished_events.append(event)

            flow.resume("approved")
            crewai_event_bus.flush()

        assert finished_events == []

    def test_finalize_session_traces_restores_event_scope(self, capsys) -> None:
        """No ``empty scope stack`` warning when deferred ``flow_finished`` fires.

        The first turn's ``flow_started`` event id is stashed on the flow
        so ``finalize_session_traces`` can restore the scope before emitting
        ``flow_finished``. Without this, the event bus prints
        ``Warning: Ending event 'flow_finished' emitted with empty scope stack``.
        """

        @ConversationConfig()
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        flow.defer_trace_finalization = True

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.first_time_handler.is_first_time = False

        with patch.object(listener.batch_manager, "finalize_batch"):
            flow.handle_turn("hi")
            flow.finalize_session_traces()

        captured = capsys.readouterr()
        assert "Missing starting event" not in (captured.out + captured.err), (
            "finalize_session_traces should restore the flow_started scope so "
            "the event bus pairs flow_finished with its opener"
        )

    def test_finalize_session_traces_is_noop_when_not_deferred(self) -> None:
        """Without deferral, ``finalize_session_traces()`` must not re-emit.

        Each per-turn ``handle_turn()`` already emits its own
        ``flow_finished``; a defensive ``try/finally`` call to
        ``finalize_session_traces()`` at session end must not emit a second,
        unpaired session-end event (which would confuse tracing).
        """
        from crewai.events.types.flow_events import FlowFinishedEvent

        @ConversationConfig(defer_trace_finalization=False)
        class PlainFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = PlainFlow()  # finalization NOT deferred

        # A non-deferred turn closes itself (no flow_started stashed for later).
        flow.handle_turn("turn 1")
        crewai_event_bus.flush()
        assert getattr(flow, "_deferred_flow_started_event_id", None) is None

        # Capture only what finalize_session_traces emits.
        finished_events: list[FlowFinishedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture(_: Any, event: FlowFinishedEvent) -> None:
                finished_events.append(event)

            flow.finalize_session_traces()
            crewai_event_bus.flush()

        assert finished_events == [], (
            "finalize_session_traces must be a no-op when finalization was not "
            "deferred — it should not emit a duplicate flow_finished"
        )


class TestHandleTurnReplyFallback:
    """Regression tests for EPD-181: ``handle_turn()`` decided "did the
    handler append its reply?" by comparing assistant-message counts. A
    handler that appends its reply AND trims history to a cap left the count
    unchanged, so the fallback appended the reply a second time — every turn,
    once trimming engaged. The check now uses an explicit appended-this-turn
    flag.
    """

    MAX_MESSAGES = 4

    def _make_bot(self) -> ConversationalFlow:
        max_messages = self.MAX_MESSAGES

        class EchoBot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "ECHO"

            @listen("ECHO")
            def echo(self) -> str:
                reply = f"echo: {self.state.current_user_message or ''}"
                self.append_assistant_message(reply)  # handler DOES append
                if len(self.state.messages) > max_messages:  # ...and trims
                    self.state.messages = self.state.messages[-max_messages:]
                return reply

        return EchoBot()

    def test_no_duplicate_reply_when_handler_trims_history(self) -> None:
        bot = self._make_bot()
        for i in range(1, 5):
            bot.handle_turn(f"message {i}")
            contents = [message.content for message in bot.state.messages]
            assert len(contents) == len(set(contents)), (
                f"duplicate reply on turn {i}: {contents}"
            )

        # The capped window holds the last two full turns, in order.
        assert [message.content for message in bot.state.messages] == [
            "message 3",
            "echo: message 3",
            "message 4",
            "echo: message 4",
        ]

    def test_fallback_still_appends_when_handler_does_not_reply(self) -> None:
        class SilentBot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "WORK"

            @listen("WORK")
            def work(self) -> str:
                return "computed reply"  # returns without appending

        bot = SilentBot()
        bot.handle_turn("hello")

        assistant_messages = [
            message.content
            for message in bot.state.messages
            if message.role == "assistant"
        ]
        assert assistant_messages == ["computed reply"]


class TestPersistCustomListenReplies:
    """Custom ``@listen`` returns must land in ``@persist`` snapshots.

    The fallback appends after ``kickoff()``, so the per-method snapshot is
    user-only unless we persist again. Fresh Flow instances restore the
    latest row; without that second snapshot the assistant turn disappears.
    """

    SESSION = "persist-custom-listen-session"

    @staticmethod
    def _roles(messages: Any) -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for message in messages:
            if hasattr(message, "role"):
                rows.append((message.role, message.content))
            else:
                rows.append((message["role"], message["content"]))
        return rows

    def test_custom_listen_return_persists_across_fresh_instances(
        self, tmp_path: Any
    ) -> None:
        store = SQLiteFlowPersistence(str(tmp_path / "custom-listen.db"))

        @persist(store)
        class ResearchBot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "research"

            @listen("research")
            def run_research(self) -> str:
                return f"researched: {self.state.current_user_message}"

        bot_a = ResearchBot()
        bot_a.handle_turn("what is CrewAI?", session_id=self.SESSION)

        saved = store.load_state(self.SESSION)
        assert saved is not None
        assert self._roles(saved["messages"]) == [
            ("user", "what is CrewAI?"),
            ("assistant", "researched: what is CrewAI?"),
        ]

        bot_b = ResearchBot()
        bot_b.handle_turn("tell me more", session_id=self.SESSION)

        assert self._roles(bot_b.state.messages) == [
            ("user", "what is CrewAI?"),
            ("assistant", "researched: what is CrewAI?"),
            ("user", "tell me more"),
            ("assistant", "researched: tell me more"),
        ]

    def test_builtin_converse_does_not_double_append_with_persist(
        self, tmp_path: Any
    ) -> None:
        store = SQLiteFlowPersistence(str(tmp_path / "converse.db"))
        chat_llm = MagicMock()
        chat_llm.call.return_value = "hello back"

        @ConversationConfig(llm=chat_llm)
        @persist(store)
        class ChatBot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return None

        bot_a = ChatBot()
        bot_a.handle_turn("hi", session_id=self.SESSION)

        assert self._roles(bot_a.state.messages) == [
            ("user", "hi"),
            ("assistant", "hello back"),
        ]

        bot_b = ChatBot()
        bot_b.handle_turn("again", session_id=self.SESSION)

        assert self._roles(bot_b.state.messages) == [
            ("user", "hi"),
            ("assistant", "hello back"),
            ("user", "again"),
            ("assistant", "hello back"),
        ]

    def test_custom_listen_return_emits_one_user_and_one_assistant_message_event(
        self, tmp_path: Any
    ) -> None:
        store = SQLiteFlowPersistence(str(tmp_path / "trace.db"))

        @persist(store)
        class ResearchBot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "research"

            @listen("research")
            def run_research(self) -> str:
                return "researched"

        events: list[ConversationMessageAddedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ConversationMessageAddedEvent)
            def capture(_: Any, event: ConversationMessageAddedEvent) -> None:
                events.append(event)

            ResearchBot().handle_turn("hello", session_id=self.SESSION)
            crewai_event_bus.flush()

        assert [(event.role, event.content) for event in events] == [
            ("user", "hello"),
            ("assistant", "researched"),
        ]


class TestFalsyRouteTurnFallback:
    """A falsy ``route_turn()`` must never replay a previous turn's intent.

    Regression tests for EPD-176: an overridden ``route_turn()`` returning
    ``None`` on an unhandled input used to silently reuse the sticky
    ``state.last_intent`` from the *previous* turn, running the wrong handler
    with no error or warning.
    """

    def test_falsy_route_turn_does_not_replay_previous_turns_intent(self) -> None:
        ran: list[str] = []

        class Bot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                message = context.get("current_user_message") or ""
                if "hello" in message.lower():
                    return "GREETING"
                return None  # unhandled input -> falsy return

            @listen("GREETING")
            def greeting(self) -> str:
                ran.append("GREETING")
                reply = "Hi! I only do greetings."
                self.append_assistant_message(reply)
                return reply

            @listen("WEATHER")
            def weather(self) -> str:
                ran.append("WEATHER")
                reply = "It is sunny."
                self.append_assistant_message(reply)
                return reply

        flow = Bot()
        flow.handle_turn("hello there")
        assert ran == ["GREETING"]
        assert flow.state.last_intent == "GREETING"

        flow.handle_turn("what is the meaning of life?")
        assert ran == ["GREETING"], (
            "an unhandled turn must not re-run the previous turn's handler"
        )
        # With no routing decision the turn falls through to the built-in
        # 'converse' default instead of replaying the stale intent.
        assert flow.state.last_intent == "converse"
        assert flow.state.messages[-1].content != "Hi! I only do greetings."

    def test_stale_intent_ignored_but_route_selected_event_still_emitted(
        self,
    ) -> None:
        class Bot(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                message = context.get("current_user_message") or ""
                return "work" if "work" in message else None

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = Bot()
        routes: list[ConversationRouteSelectedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(ConversationRouteSelectedEvent)
            def capture(_: Any, event: ConversationRouteSelectedEvent) -> None:
                routes.append(event)

            flow.handle_turn("work please")
            flow.handle_turn("something unrelated")
            crewai_event_bus.flush()

        assert [event.route for event in routes] == ["work", "converse"]
        # The fallback decision still reports the prior intent for visibility.
        assert routes[1].previous_intent == "work"

    def test_fresh_intent_classified_this_turn_still_routes(self) -> None:
        """The legacy ``default_intents`` path classifies per turn and must
        keep routing on the freshly classified intent — including when the
        intent changes between turns."""
        ran: list[str] = []

        @ConversationConfig(
            default_intents=["search", "weather"], intent_llm="gpt-4o-mini"
        )
        class LegacyFlow(ConversationalFlow):
            @listen("search")
            def handle_search(self) -> str:
                ran.append("search")
                self.append_assistant_message("searched")
                return "searched"

            @listen("weather")
            def handle_weather(self) -> str:
                ran.append("weather")
                self.append_assistant_message("sunny")
                return "sunny"

        flow = LegacyFlow()
        with patch.object(
            flow, "_collapse_to_outcome", side_effect=["search", "weather"]
        ):
            flow.handle_turn("look up crewai")
            flow.handle_turn("how is the weather?")

        assert ran == ["search", "weather"]
        assert flow.state.last_intent == "weather"


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

    def test_method_execution_suppressed_when_flow_events_suppressed(self) -> None:
        """``suppress_flow_events=True`` silences MethodExecution events so
        infrastructure flows (AgentExecutor, memory) don't emit one trace span
        per internal control-flow method."""

        class QuietFlow(Flow[ChatState]):
            suppress_flow_events = True

            @start()
            def begin(self) -> str:
                return "ok"

        started: list[str] = []
        finished: list[str] = []
        original_emit = crewai_event_bus.emit

        def track_emit(source: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(event, MethodExecutionStartedEvent):
                started.append(event.method_name)
            if isinstance(event, MethodExecutionFinishedEvent):
                finished.append(event.method_name)
            return original_emit(source, event, *args, **kwargs)

        with patch.object(crewai_event_bus, "emit", side_effect=track_emit):
            QuietFlow().kickoff()

        assert started == []
        assert finished == []

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
    def test_bare_conversational_flow_defers_by_default(self) -> None:
        class BareChat(ConversationalFlow):
            pass

        assert BareChat()._should_defer_trace_finalization() is True

    def test_conversation_config_drives_defer_flag(self) -> None:
        """``ConversationConfig(defer_trace_finalization=...)`` controls whether
        a conversational subclass defers per-turn trace finalization."""

        @ConversationConfig(defer_trace_finalization=True)
        class DeferOn(ConversationalFlow):
            pass

        @ConversationConfig(defer_trace_finalization=False)
        class DeferOff(ConversationalFlow):
            pass

        assert DeferOn()._should_defer_trace_finalization() is True
        assert DeferOff()._should_defer_trace_finalization() is False



class TestDeferredFlowLifecycleEvents:
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

    def test_finalize_session_traces_is_idempotent(self) -> None:
        """Calling ``finalize_session_traces()`` twice emits flow_finished once.

        The stashed ``_deferred_flow_started_event_id`` is cleared after the
        first call, so a second call (e.g. a defensive ``try/finally``) does
        not re-emit a session-end event.
        """
        from crewai.events.types.flow_events import FlowFinishedEvent

        @ConversationConfig(defer_trace_finalization=True)
        class DeferredFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("worked")
                return "worked"

        flow = DeferredFlow()
        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.first_time_handler.is_first_time = False

        finished: list[FlowFinishedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowFinishedEvent)
            def capture(_: Any, event: FlowFinishedEvent) -> None:
                finished.append(event)

            with patch.object(listener.batch_manager, "finalize_batch"):
                flow.handle_turn("hi")
                crewai_event_bus.flush()
                flow.finalize_session_traces()
                flow.finalize_session_traces()  # second call must be a no-op
                crewai_event_bus.flush()

        assert len(finished) == 1, (
            "finalize_session_traces must emit flow_finished exactly once, even "
            "when called more than once"
        )

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

    def test_deferred_flow_kickoff_marks_trace_manager_session_deferred(
        self,
    ) -> None:
        class DeferredTraceFlow(Flow[ChatState]):
            @start()
            def begin(self) -> str:
                return "done"

        listener = TraceCollectionListener()
        listener.batch_manager.defer_session_finalization = False

        flow = DeferredTraceFlow()
        flow.defer_trace_finalization = True

        with patch.object(listener.batch_manager, "finalize_batch"):
            flow.kickoff()

        assert listener.batch_manager.defer_session_finalization is True

        flow.finalize_session_traces()

        assert listener.batch_manager.defer_session_finalization is False

    def test_non_deferred_flow_kickoff_clears_stale_trace_manager_flag(
        self,
    ) -> None:
        class PlainTraceFlow(Flow[ChatState]):
            @start()
            def begin(self) -> str:
                return "done"

        listener = TraceCollectionListener()
        listener.batch_manager.defer_session_finalization = True

        PlainTraceFlow().kickoff()

        assert listener.batch_manager.defer_session_finalization is False


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

    def test_lazy_flow_batch_from_context_preserves_deferred_parent(self) -> None:
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )

        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = None
        listener.batch_manager.batch_owner_type = None
        listener.batch_manager.batch_owner_id = None
        listener.batch_manager.defer_session_finalization = False
        listener.batch_manager.event_buffer.clear()

        flow_id_token = current_flow_id.set("parent-flow-id")
        flow_name_token = current_flow_name.set("ParentChatFlow")
        defer_token = current_flow_defer_trace_finalization.set(True)
        try:
            initialized = listener._try_initialize_flow_batch_from_context(
                type("Event", (), {"timestamp": None})()
            )

            assert initialized is True
            assert listener.batch_manager.batch_owner_type == "flow"
            assert listener.batch_manager.batch_owner_id == "parent-flow-id"
            assert listener.batch_manager.defer_session_finalization is True
            assert listener.batch_manager.current_batch is not None
            assert (
                listener.batch_manager.current_batch.execution_metadata[
                    "execution_type"
                ]
                == "flow"
            )
            assert (
                listener.batch_manager.current_batch.execution_metadata["flow_name"]
                == "ParentChatFlow"
            )
        finally:
            current_flow_defer_trace_finalization.reset(defer_token)
            current_flow_name.reset(flow_name_token)
            current_flow_id.reset(flow_id_token)
            listener.batch_manager.current_batch = None
            listener.batch_manager.batch_owner_type = None
            listener.batch_manager.batch_owner_id = None
            listener.batch_manager.trace_batch_id = None
            listener.batch_manager.defer_session_finalization = False
            listener.batch_manager.event_buffer.clear()

    def test_nested_agent_executor_flow_does_not_finalize_parent_batch(
        self,
    ) -> None:
        from crewai import Agent, Crew, Task
        from crewai.llms.base_llm import BaseLLM

        class StaticLLM(BaseLLM):
            def __init__(self) -> None:
                super().__init__(model="debug-static-llm", provider="debug")

            def call(
                self,
                messages: Any,
                tools: Any = None,
                callbacks: Any = None,
                available_functions: Any = None,
                from_task: Any = None,
                from_agent: Any = None,
                response_model: Any = None,
            ) -> str:
                return (
                    "Thought: I can answer directly.\n"
                    "Final Answer: nested crew result"
                )

        class NestedCrewFlow(Flow[ChatState]):
            defer_trace_finalization = True
            tracing = True

            @start()
            def begin(self) -> str:
                return "run_nested_crew"

            @listen(begin)
            def run_nested_crew(self, _: str) -> str:
                agent = Agent(
                    role="Debug Agent",
                    goal="Return a short deterministic result",
                    backstory="Used only for trace finalization debugging.",
                    llm=StaticLLM(),
                    verbose=False,
                )
                task = Task(
                    description="Return the deterministic nested crew result.",
                    expected_output="nested crew result",
                    agent=agent,
                )
                return Crew(agents=[agent], tasks=[task], verbose=False).kickoff().raw

        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = None
        listener.batch_manager.batch_owner_type = None
        listener.batch_manager.batch_owner_id = None
        listener.batch_manager.trace_batch_id = None
        listener.batch_manager.defer_session_finalization = False
        listener.batch_manager.event_buffer.clear()
        listener.first_time_handler.is_first_time = False

        def initialize_backend_batch(*_: Any, **__: Any) -> None:
            listener.batch_manager.trace_batch_id = "debug-trace-batch"

        flow = NestedCrewFlow()

        with (
            patch.object(
                listener.batch_manager,
                "_initialize_backend_batch",
                side_effect=initialize_backend_batch,
            ),
            patch.object(listener.batch_manager, "finalize_batch") as mock_finalize,
        ):
            flow.kickoff()
            crewai_event_bus.flush()
            flow.kickoff()
            crewai_event_bus.flush()

            assert mock_finalize.call_count == 0, (
                "nested AgentExecutor flows inside a deferred parent Flow must "
                "not finalize the parent trace batch"
            )


class TestConversationalOptIn:
    """``@ConversationConfig`` opts a Flow into conversational mode."""

    def test_decorator_alone_enables_conversational_mode(self) -> None:
        @ConversationConfig(llm="gpt-4o-mini")
        class DecoratedFlow(Flow[ConversationState]):
            @listen("order")
            def handle_order(self) -> str:
                """Order status questions."""
                return "on the way"

        definition = DecoratedFlow.flow_definition()

        assert definition.conversational is not None
        assert definition.conversational.enabled is True
        assert definition.conversational.llm == "gpt-4o-mini"

    def test_decorator_alone_registers_the_builtin_methods(self) -> None:
        @ConversationConfig()
        class DecoratedFlow(Flow[ConversationState]):
            pass

        methods = DecoratedFlow.flow_definition().methods

        assert "route_conversation" in methods
        assert methods["route_conversation"].start is True
        assert methods["route_conversation"].router is True
        assert "converse_turn" in methods
        assert "end_conversation" in methods

    def test_decorator_alone_runs_a_turn(self) -> None:
        @ConversationConfig()
        class DecoratedFlow(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> str:
                return "handled"

        flow = DecoratedFlow()

        assert flow.handle_turn("hello") == "handled"
        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "hello"),
            ("assistant", "handled"),
        ]

    def test_decorator_alone_defers_trace_finalization(self) -> None:
        @ConversationConfig()
        class DecoratedFlow(Flow[ConversationState]):
            pass

        assert DecoratedFlow()._should_defer_trace_finalization() is True

    def test_explicit_flag_without_a_config_still_opts_in(self) -> None:
        class FlagOnlyFlow(Flow[ConversationState]):
            conversational = True

        definition = FlagOnlyFlow.flow_definition()

        assert definition.conversational is not None
        assert definition.conversational.enabled is True
        assert FlagOnlyFlow()._is_conversational_enabled() is True

    def test_flow_with_neither_stays_non_conversational(self) -> None:
        class PlainFlow(Flow):
            @start()
            def begin(self) -> str:
                return "begin"

        definition = PlainFlow.flow_definition()

        assert definition.conversational is None
        assert set(definition.methods) == {"begin"}
        assert PlainFlow.conversational is False

    def test_decorator_does_not_leak_the_flag_onto_other_flows(self) -> None:
        @ConversationConfig()
        class DecoratedFlow(Flow[ConversationState]):
            pass

        class LaterPlainFlow(Flow):
            @start()
            def begin(self) -> str:
                return "begin"

        assert DecoratedFlow.conversational is True
        assert LaterPlainFlow.conversational is False
        assert Flow.conversational is False
        assert LaterPlainFlow.flow_definition().conversational is None


class TestHandleTurnGuard:
    """``handle_turn`` fails loudly on a non-conversational Flow."""

    def test_handle_turn_rejects_non_conversational_flows(self) -> None:
        class PlainFlow(Flow[ConversationState]):
            @start()
            def begin(self) -> str:
                return "begin"

        with pytest.raises(
            ValueError,
            match="Flow.handle_turn\\(\\) is only available on conversational flows",
        ):
            PlainFlow().handle_turn("hello")

    def test_handle_turn_guard_matches_chat_and_stream_turn(self) -> None:
        class PlainFlow(Flow[ConversationState]):
            @start()
            def begin(self) -> str:
                return "begin"

        flow = PlainFlow()

        for call in (
            lambda: flow.handle_turn("hi"),
            lambda: flow.stream_turn("hi"),
            flow.chat,
        ):
            with pytest.raises(
                ValueError, match="only available on conversational flows"
            ):
                call()

    def test_guard_does_not_fire_on_a_conversational_flow(self) -> None:
        @ConversationConfig()
        class DecoratedFlow(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> str:
                return "ok"

        assert DecoratedFlow().handle_turn("hi") == "ok"


_MIXIN = "crewai.experimental.conversational_mixin:_ConversationalMixin"


class DeclaredChatState(ConversationState):
    """Module-level so ``state.ref`` can import it; a locals-scoped class cannot."""

    ticket_id: str | None = None


class _ScriptedLLM(BaseLLM):
    """Fake LLM returning queued responses; records the messages it saw."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake")
        object.__setattr__(self, "_responses", list(responses or []))
        object.__setattr__(self, "seen", [])

    def call(self, messages, **kwargs) -> str:  # type: ignore[no-untyped-def]
        self.seen.append(messages)
        return self._responses.pop(0) if self._responses else "fallback"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


class _Outer:
    """Holds a nested model, so its qualname is dotted without `<locals>`."""

    class Route(BaseModel):
        intent: str


def _conversational_declaration(**overrides: Any) -> dict[str, Any]:
    """A declaration naming the built-in methods explicitly.

    Synthesizing these entries from ``conversational.enabled`` is a follow-up;
    until then a declaration has to name them.
    """
    declaration: dict[str, Any] = {
        "schema": "crewai.flow/v1",
        "name": "DeclaredChat",
        "state": {
            "type": "pydantic",
            "ref": "crewai.experimental.conversational:ConversationState",
        },
        "conversational": {},
        "methods": {
            "route_conversation": {
                "do": {"call": "code", "ref": f"{_MIXIN}.route_conversation"},
                "start": True,
                "router": True,
            },
            "converse_turn": {
                "do": {"call": "code", "ref": f"{_MIXIN}.converse_turn"},
                "listen": "converse",
            },
            "end_conversation": {
                "do": {"call": "code", "ref": f"{_MIXIN}.end_conversation"},
                "listen": "end",
            },
        },
    }
    declaration.update(overrides)
    return declaration


class TestDeclarativeConversationalFlow:
    """A declaration's ``conversational`` block drives the runtime."""

    def test_declaration_enables_conversational_mode(self) -> None:
        flow = Flow.from_declaration(contents=_conversational_declaration())

        assert flow._is_conversational_enabled() is True

    def test_declaration_with_enabled_false_stays_non_conversational(self) -> None:
        flow = Flow.from_declaration(
            contents=_conversational_declaration(conversational={"enabled": False})
        )

        assert flow._is_conversational_enabled() is False

    def test_declared_route_labels_reach_the_router_catalog(self) -> None:
        declaration = _conversational_declaration()
        declaration["methods"]["handle_order"] = {
            "do": {"call": "expression", "expr": "'shipped'"},
            "listen": "order",
            "description": "Order status questions.",
        }

        flow = Flow.from_declaration(contents=declaration)

        assert "order" in flow._valid_route_labels()
        assert "order" in flow._effective_routes()

    def test_declaration_runs_a_turn_and_accumulates_history(self) -> None:
        declaration = _conversational_declaration(
            conversational={"system_prompt": "You are terse."}
        )
        chat = _ScriptedLLM(["Hi there.", "Anything else?"])
        flow = Flow.from_declaration(contents=declaration)
        flow._conversation_config.llm = chat

        assert flow.handle_turn("hello") == "Hi there."
        assert flow.handle_turn("thanks") == "Anything else?"
        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "hello"),
            ("assistant", "Hi there."),
            ("user", "thanks"),
            ("assistant", "Anything else?"),
        ]
        assert chat.seen[-1][0] == {"role": "system", "content": "You are terse."}

    def test_declaration_drives_deferred_trace_finalization(self) -> None:
        deferred = Flow.from_declaration(contents=_conversational_declaration())
        not_deferred = Flow.from_declaration(
            contents=_conversational_declaration(
                conversational={"defer_trace_finalization": False}
            )
        )

        assert deferred._should_defer_trace_finalization() is True
        assert not_deferred._should_defer_trace_finalization() is False

    def test_declaration_orders_the_router_last_and_sequentially(self) -> None:
        flow = Flow.from_declaration(contents=_conversational_declaration())

        ordered, sequential = flow._order_start_methods_for_kickoff(
            ["bootstrap", "route_conversation"]
        )

        assert ordered == ["bootstrap", "route_conversation"]
        assert sequential is True

    def test_router_response_format_takes_a_python_ref_not_a_module_ref(self) -> None:
        """The contract declares the same `{"python": ...}` shape crews use.

        A bare `module:qualname` ref used to be accepted and silently dropped;
        it is now a load-time error, so a typo cannot look like it worked.
        """
        with pytest.raises(ValidationError, match="response_format"):
            Flow.from_declaration(
                contents=_conversational_declaration(
                    conversational={
                        "router": {"response_format": {"ref": "some.module:Schema"}}
                    }
                )
            )


class TestDeclaredRouterResponseFormat:
    """A declaration can name the model the routing decision is parsed into."""

    ROUTE_MODULE = (
        "from typing import Literal\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class ConversationRoute(BaseModel):\n"
        "    intent: Literal['order', 'converse']\n"
    )

    @staticmethod
    def _declaration(router: dict[str, Any]) -> dict[str, Any]:
        return _conversational_declaration(conversational={"router": router})

    def test_a_declared_python_ref_is_resolved_to_the_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "routes.py").write_text(self.ROUTE_MODULE, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))

        flow = Flow.from_declaration(
            contents=self._declaration(
                {"response_format": {"python": "routes.ConversationRoute"}}
            )
        )
        resolved = flow._conversation_config.router.response_format

        assert resolved is not None
        assert resolved.__name__ == "ConversationRoute"
        assert "intent" in resolved.model_fields
        assert flow._router_response_format(flow._conversation_config.router) is resolved

    def test_a_ref_resolves_next_to_the_declaration_not_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flow loaded by path finds the model sitting beside its YAML.

        Resolution used to fall back to ``Path.cwd()``, so the same project
        loaded from another directory could not import its own route model.
        """
        project = tmp_path / "project"
        project.mkdir()
        (project / "chat_routes.py").write_text(self.ROUTE_MODULE, encoding="utf-8")
        declaration = project / "chat.yaml"
        declaration.write_text(
            yaml.safe_dump(
                self._declaration(
                    {"response_format": {"python": "chat_routes.ConversationRoute"}}
                )
            ),
            encoding="utf-8",
        )

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delitem(sys.modules, "chat_routes", raising=False)

        try:
            flow = Flow.from_declaration(path=declaration)
            resolved = flow._conversation_config.router.response_format
        finally:
            sys.modules.pop("chat_routes", None)

        assert resolved is not None
        assert resolved.__name__ == "ConversationRoute"
        assert "intent" in resolved.model_fields

    def test_omitting_it_still_synthesizes_one(self) -> None:
        flow = Flow.from_declaration(contents=self._declaration({}))

        synthesized = flow._router_response_format(flow._conversation_config.router)

        assert flow._conversation_config.router.response_format is None
        assert list(synthesized.model_fields) == ["intent"]

    def test_a_ref_outside_the_project_root_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declarations may not reach outside the project to import code."""
        monkeypatch.chdir(tmp_path)
        flow = Flow.from_declaration(
            contents=self._declaration(
                {"response_format": {"python": "os.path.basename"}}
            )
        )

        # Resolution is lazy: the declaration loads, the import is refused.
        with pytest.raises(Exception, match="inside the project root"):
            _ = flow._conversation_config

    def test_a_ref_without_a_dot_is_rejected_at_load(self) -> None:
        with pytest.raises(ValidationError):
            Flow.from_declaration(
                contents=self._declaration({"response_format": {"python": "nodots"}})
            )

    def test_a_live_class_on_a_python_flow_is_untouched(self) -> None:
        class MyRoute(BaseModel):
            intent: str

        @ConversationConfig(router=RouterConfig(response_format=MyRoute))
        class ClassChat(Flow[ConversationState]):
            pass

        assert ClassChat()._conversation_config.router.response_format is MyRoute

    def test_a_function_local_model_is_omitted_from_the_projection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A path through `<locals>` cannot be imported back, so never emit it."""

        class LocalRoute(BaseModel):
            intent: str

        @ConversationConfig(router=RouterConfig(response_format=LocalRoute))
        class ClassChat(Flow[ConversationState]):
            pass

        with caplog.at_level(logging.WARNING, logger="crewai.flow.dsl._utils"):
            definition = ClassChat.flow_definition()

        assert definition.conversational.router.response_format is None
        # Silently dropping it would leave the author guessing, so warn.
        assert "cannot be imported by path" in caplog.text
        # The live class still drives the running flow; only the projection drops it.
        assert ClassChat()._conversation_config.router.response_format is LocalRoute

    def test_a_non_model_response_format_is_omitted_from_the_projection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class NotAModel:
            pass

        @ConversationConfig(router=RouterConfig(response_format=NotAModel))
        class ClassChat(Flow[ConversationState]):
            pass

        with caplog.at_level(logging.WARNING, logger="crewai.flow.dsl._utils"):
            definition = ClassChat.flow_definition()

        assert definition.conversational.router.response_format is None
        assert "is not a Pydantic model class" in caplog.text

    def test_a_nested_model_is_omitted_from_the_projection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`module.Outer.Route` reloads `module.Outer` as a module that is absent."""

        @ConversationConfig(router=RouterConfig(response_format=_Outer.Route))
        class ClassChat(Flow[ConversationState]):
            pass

        with caplog.at_level(logging.WARNING, logger="crewai.flow.dsl._utils"):
            definition = ClassChat.flow_definition()

        assert definition.conversational.router.response_format is None
        assert "cannot be imported by its module path" in caplog.text
        assert ClassChat()._conversation_config.router.response_format is _Outer.Route

    def test_an_unbound_generated_model_is_omitted_from_the_projection(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `create_model()` class no module attribute names cannot be reloaded."""
        generated = create_model("GeneratedRoute", intent=(str, ...))

        @ConversationConfig(router=RouterConfig(response_format=generated))
        class ClassChat(Flow[ConversationState]):
            pass

        with caplog.at_level(logging.WARNING, logger="crewai.flow.dsl._utils"):
            definition = ClassChat.flow_definition()

        assert definition.conversational.router.response_format is None
        assert "cannot be imported by its module path" in caplog.text

        reloaded = Flow.from_declaration(contents=definition.to_dict())
        synthesized = reloaded._router_response_format(
            reloaded._conversation_config.router
        )

        assert synthesized is not generated
        assert list(synthesized.model_fields) == ["intent"]

    def test_a_dropped_projection_reloads_with_the_synthesized_model(self) -> None:
        class LocalRoute(BaseModel):
            intent: str

        @ConversationConfig(router=RouterConfig(response_format=LocalRoute))
        class ClassChat(Flow[ConversationState]):
            pass

        reloaded = Flow.from_declaration(
            contents=ClassChat.flow_definition().to_dict()
        )
        synthesized = reloaded._router_response_format(
            reloaded._conversation_config.router
        )

        assert list(synthesized.model_fields) == ["intent"]

    def test_a_live_class_projects_as_a_python_ref(self) -> None:
        @ConversationConfig(router=RouterConfig(response_format=ConversationState))
        class ClassChat(Flow[ConversationState]):
            pass

        projected = ClassChat.flow_definition().conversational.router.response_format

        assert projected is not None
        assert projected.python == (
            "crewai.experimental.conversational.ConversationState"
        )


class DeclaredSchemaChatState(ConversationState):
    """A ConversationState subclass, i.e. the already-supported state shape."""

    ticket_id: str | None = None


class TestDeclaredConversationalState:
    """Any declared state shape works for a chat flow, keeping its own fields."""

    @staticmethod
    def _flow(state: dict[str, Any] | None) -> Flow[Any]:
        declaration = _conversational_declaration(conversational={})
        if state is None:
            declaration.pop("state", None)
        else:
            declaration["state"] = state
        return Flow.from_declaration(contents=declaration)

    CONVERSATIONAL_FIELDS = (
        "messages",
        "current_user_message",
        "last_intent",
        "ended",
        "events",
        "agent_threads",
    )

    def test_inline_json_schema_state_gains_the_conversational_fields(self) -> None:
        flow = self._flow(
            {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "ticket_id": {"type": "string"},
                        "turns": {"type": "integer"},
                    },
                },
                "default": {"ticket_id": "T-1"},
            }
        )
        fields = type(flow.state).model_fields

        for name in self.CONVERSATIONAL_FIELDS:
            assert name in fields, name
        assert "ticket_id" in fields and "turns" in fields
        assert flow.state.ticket_id == "T-1"
        assert flow.state.id

    def test_a_pydantic_ref_that_is_not_a_conversation_state_is_composed(self) -> None:
        flow = self._flow({"type": "pydantic", "ref": "crewai.flow:ChatState"})
        fields = type(flow.state).model_fields

        for name in self.CONVERSATIONAL_FIELDS:
            assert name in fields, name
        assert "session_ready" in fields

    def test_a_conversation_state_subclass_is_not_composed_twice(self) -> None:
        flow = self._flow(
            {"type": "pydantic", "ref": f"{__name__}:DeclaredSchemaChatState"}
        )
        mro = [cls.__name__ for cls in type(flow.state).__mro__]

        assert mro.count("ConversationState") == 1
        assert "ticket_id" in type(flow.state).model_fields

    def test_dict_state_is_given_the_real_shape(self) -> None:
        """A dict cannot carry the fields, so supply them rather than die."""
        flow = self._flow({"type": "dict", "default": {"last_intent": "order"}})

        assert isinstance(flow.state, ConversationState)
        assert flow.state.last_intent == "order"

    def test_dict_state_keeps_declared_defaults_it_does_not_own(self) -> None:
        """Dropping them would break an action reading `state.topic`."""
        flow = self._flow({"type": "dict", "default": {"topic": "ai", "limit": 3}})

        assert flow.state.topic == "ai"
        assert flow.state.limit == 3
        assert isinstance(flow.state, ConversationState)

    def test_unknown_state_is_treated_like_dict(self) -> None:
        flow = self._flow({"type": "unknown", "ref": "x:Y", "default": {"topic": "ai"}})

        assert flow.state.topic == "ai"
        assert isinstance(flow.state, ConversationState)

    def test_an_unbuildable_declared_model_still_gets_the_chat_shape(self) -> None:
        """A bad ref fell back to a plain dict, so the turn died on `state.id`."""
        flow = self._flow({"type": "pydantic", "ref": "no.such.module:Nope"})

        assert isinstance(flow.state, ConversationState)
        assert flow.state.id

    def test_declared_state_still_runs_a_turn(self) -> None:
        flow = self._flow(
            {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                },
            }
        )
        flow._conversation_config.llm = _ScriptedLLM(["Hello."])

        assert flow.handle_turn("hi") == "Hello."
        assert [m.role for m in flow.state.messages] == ["user", "assistant"]

    def test_non_conversational_declaration_keeps_its_plain_state(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "Plain",
                "state": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                    },
                    "default": {"topic": "ai"},
                },
                "methods": {
                    "begin": {
                        "do": {"call": "expression", "expr": "state.topic"},
                        "start": True,
                    }
                },
            }
        )

        assert "messages" not in type(flow.state).model_fields
        assert flow.state.topic == "ai"

class TestDeclaredConversationalLLM:
    """A conversational block accepts the shapes a crew agent's `llm` accepts."""

    @staticmethod
    def _resolved(llm: Any) -> Any:
        flow = Flow.from_declaration(
            contents=_conversational_declaration(conversational={"llm": llm})
        )
        return flow._coerce_llm(flow._conversation_config.llm)

    def test_model_id_string(self) -> None:
        assert self._resolved("gpt-4o-mini").model == "gpt-4o-mini"

    def test_config_mapping_carries_provider_settings(self) -> None:
        resolved = self._resolved({"model": "openai/gpt-4o-mini", "max_tokens": 512})

        assert resolved.model == "gpt-4o-mini"
        assert resolved.max_tokens == 512

    def test_config_mapping_passes_through_extra_settings(self) -> None:
        resolved = self._resolved({"model": "openai/gpt-4o-mini", "temperature": 0.2})

        assert resolved.temperature == 0.2

    def test_a_live_llm_object_is_passed_through_untouched(self) -> None:
        llm = _ScriptedLLM(["hi"])

        @ConversationConfig(llm=llm)
        class ClassChat(Flow[ConversationState]):
            pass

        assert ClassChat()._coerce_llm(llm) is llm

    def test_a_mapping_without_a_model_key_says_so(self) -> None:
        with pytest.raises(ValueError, match="must include 'model'"):
            self._resolved({"max_tokens": 512})

    def test_a_wrong_type_fails_now_not_at_call_time(self) -> None:
        """`create_llm` would take an int as a model name and fail later."""
        with pytest.raises(ValueError, match="expected a model-id string"):
            self._resolved(1234)

    def test_an_llm_definition_resolves_with_its_settings(self) -> None:
        from crewai.project.crew_definition import LLMDefinition

        flow = Flow.from_declaration(contents=_conversational_declaration())
        resolved = flow._coerce_llm(
            LLMDefinition(model="openai/gpt-4o-mini", max_tokens=256)
        )

        assert resolved.model == "gpt-4o-mini"
        assert resolved.max_tokens == 256

    def test_a_declared_mapping_reaches_create_llm_during_a_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Swapping the config out before the turn would not prove the path."""
        declared = {"model": "openai/gpt-4o-mini", "max_tokens": 128}
        scripted = _ScriptedLLM(["Hello."])
        seen: list[Any] = []

        def fake_create_llm(value: Any) -> Any:
            seen.append(value)
            return scripted

        monkeypatch.setattr(
            "crewai.utilities.llm_utils.create_llm", fake_create_llm
        )
        flow = Flow.from_declaration(
            contents=_conversational_declaration(conversational={"llm": declared})
        )

        assert flow.handle_turn("hi") == "Hello."
        assert declared in seen

    def test_a_declared_intent_llm_mapping_is_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_collapse_to_outcome` takes only str | BaseLLM, so it must be coerced.

        Swapping the mapping for an LLM before the turn would skip the coercion
        entirely, so the mapping stays declared and ``create_llm`` is patched.
        """
        declared = {"model": "openai/gpt-4o-mini", "max_tokens": 64}
        intent_llm = _ScriptedLLM(['{"outcome": "order"}'])
        converse_llm = _ScriptedLLM(["ok"])
        seen: list[Any] = []

        def fake_create_llm(value: Any) -> Any:
            seen.append(value)
            return intent_llm if value == declared else converse_llm

        monkeypatch.setattr("crewai.utilities.llm_utils.create_llm", fake_create_llm)
        flow = Flow.from_declaration(
            contents=_conversational_declaration(
                conversational={"default_intents": ["order"], "intent_llm": declared}
            )
        )

        # Before coercion this raised "Invalid llm type: <class 'dict'>".
        flow.handle_turn("where is my order?")

        assert declared in seen
        assert [m.role for m in flow.state.messages][0] == "user"

class TestRoutingArtefactLabels:
    """A route label echoed by a handler is a routing artefact, not a reply."""

    def test_class_based_set_matches_the_framework_labels(self) -> None:
        @ConversationConfig()
        class ClassChat(Flow[ConversationState]):
            pass

        assert ClassChat()._routing_artefact_labels() == {
            "conversation",
            "converse",
            "end",
            "answer_from_history",
            "route_to_flow",
        }

    def test_a_declared_extra_builtin_route_is_covered(self) -> None:
        """A literal list would miss it; deriving from the routes does not."""
        flow = Flow.from_declaration(
            contents=_conversational_declaration(
                conversational={"builtin_routes": ["converse", "end", "greet"]}
            )
        )

        assert "greet" in flow._routing_artefact_labels()
        assert flow._is_public_turn_result("greet") is False

    def test_ordinary_text_is_still_a_reply(self) -> None:
        flow = Flow.from_declaration(contents=_conversational_declaration())

        assert flow._is_public_turn_result("Your order shipped.") is True


class TestConversationalCapabilityAttribute:
    """A declarative chat flow reports itself conversational to outside callers.

    Consumers outside this package capability-check the ``conversational``
    attribute, so it must agree with ``_is_conversational_enabled()``.
    """

    def test_declaration_marks_the_instance_conversational(self) -> None:
        flow = Flow.from_declaration(contents=_conversational_declaration())

        assert getattr(flow, "conversational") is True
        assert flow._is_conversational_enabled() is True
        assert callable(getattr(flow, "stream_turn", None))

    def test_the_flag_does_not_leak_onto_the_class(self) -> None:
        """The DSL projection reads it off the class to decide what to emit."""
        Flow.from_declaration(contents=_conversational_declaration())

        assert Flow.conversational is False

        class LaterPlainFlow(Flow):
            @start()
            def begin(self) -> str:
                return "begin"

        assert LaterPlainFlow.conversational is False
        assert LaterPlainFlow.flow_definition().conversational is None

    def test_disabled_declaration_is_not_marked(self) -> None:
        flow = Flow.from_declaration(
            contents=_conversational_declaration(conversational={"enabled": False})
        )

        assert getattr(flow, "conversational") is False

    def test_non_conversational_declaration_is_not_marked(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "Plain",
                "methods": {
                    "begin": {
                        "do": {"call": "expression", "expr": "'x'"},
                        "start": True,
                    }
                },
            }
        )

        assert getattr(flow, "conversational") is False

    def test_class_based_flow_keeps_its_own_flag(self) -> None:
        @ConversationConfig()
        class ClassChat(Flow[ConversationState]):
            pass

        assert ClassChat.conversational is True
        assert getattr(ClassChat(), "conversational") is True


class TestBuiltinRouteResolution:
    """``route_turn`` and ``_effective_routes`` agree on what is built in."""

    @staticmethod
    def _flow(builtin: list[str]) -> Flow[Any]:
        return Flow.from_declaration(
            contents=_conversational_declaration(
                conversational={"builtin_routes": builtin}
            )
        )

    def test_declared_builtin_routes_are_not_mistaken_for_custom_ones(self) -> None:
        """A declaration adding a builtin route must not auto-enable the router.

        ``route_turn`` used to subtract the class attribute while
        ``_effective_routes`` used the declaration, so an added builtin looked
        like a custom route and turned the LLM router on for every turn.
        """
        flow = self._flow(["converse", "end", "greet"])

        assert flow._effective_builtin_routes() == {"converse", "end", "greet"}
        assert flow.route_turn(flow.build_router_context()) is None

    def test_a_real_custom_route_still_auto_enables_the_router(self) -> None:
        declaration = _conversational_declaration(conversational={})
        declaration["methods"]["handle_order"] = {
            "do": {"call": "expression", "expr": "'shipped'"},
            "listen": "order",
        }
        flow = Flow.from_declaration(contents=declaration)
        flow._conversation_config.llm = _ScriptedLLM(['{"intent": "order"}'])

        assert "order" in flow._effective_routes(None) - flow._effective_builtin_routes()
        assert flow.route_turn(flow.build_router_context()) == "order"

    def test_class_based_flow_falls_back_to_its_class_attributes(self) -> None:
        @ConversationConfig()
        class ClassChat(Flow[ConversationState]):
            pass

        flow = ClassChat()

        assert flow._effective_builtin_routes() == set(flow.builtin_routes)
        assert flow._effective_internal_routes() == set(flow.internal_routes)


class TestConversationalStatePrecedence:
    """A declared ``state:`` block is never replaced by the default."""

    def test_declared_state_survives_on_both_paths(self) -> None:
        class SubclassChat(Flow):
            conversational = True

        declaration = _conversational_declaration(
            state={"type": "pydantic", "ref": f"{__name__}:DeclaredChatState"}
        )

        for cls in (Flow, SubclassChat):
            state = cls.from_declaration(contents=declaration).state

            assert "ticket_id" in type(state).model_fields, cls.__name__
            assert "messages" in type(state).model_fields, cls.__name__

    def test_conversation_state_is_implied_when_none_declared(self) -> None:
        declaration = _conversational_declaration()
        del declaration["state"]

        flow = Flow.from_declaration(contents=declaration)

        assert isinstance(flow.state, ConversationState)

    def test_non_conversational_declaration_keeps_its_state(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "Plain",
                "state": {"type": "dict", "default": {"topic": "ai"}},
                "methods": {
                    "begin": {
                        "do": {"call": "expression", "expr": "state.topic"},
                        "start": True,
                    }
                },
            }
        )

        assert flow.state["topic"] == "ai"


class TestClassConfigStillWins:
    """Existing decorated Python flows keep their live objects."""

    def test_live_llm_object_is_not_downgraded_by_the_definition(self) -> None:
        llm = _ScriptedLLM(["from the live object"])

        @ConversationConfig(llm=llm)
        class LiveLLMFlow(ConversationalFlow):
            pass

        flow = LiveLLMFlow()

        assert flow._conversation_config.llm is llm
        assert flow.handle_turn("hello") == "from the live object"

    def test_live_router_response_format_is_not_downgraded(self) -> None:
        class MyRoute(BaseModel):
            intent: str

        @ConversationConfig(router=RouterConfig(response_format=MyRoute))
        class LiveFormatFlow(ConversationalFlow):
            pass

        assert LiveFormatFlow()._conversation_config.router.response_format is MyRoute

    def test_defer_trace_finalization_follows_the_class_config(self) -> None:
        """Deferral is a behavior knob, so it follows the same precedence.

        The class config and the declaration can disagree on the hybrid path;
        every other setting follows the class config, and this must too.
        """

        @ConversationConfig(defer_trace_finalization=False)
        class ConfiguredChat(Flow):
            conversational = True

        flow = ConfiguredChat.from_declaration(
            contents=_conversational_declaration(conversational={})
        )

        assert flow._conversation_definition.defer_trace_finalization is True
        assert flow._conversation_config.defer_trace_finalization is False
        assert flow._should_defer_trace_finalization() is False

    def test_instance_flag_still_forces_deferral(self) -> None:
        @ConversationConfig(defer_trace_finalization=False)
        class ConfiguredChat(Flow):
            conversational = True

        flow = ConfiguredChat()
        assert flow._should_defer_trace_finalization() is False

        flow.defer_trace_finalization = True
        assert flow._should_defer_trace_finalization() is True

    def test_non_conversational_flow_never_defers_from_config(self) -> None:
        class PlainFlow(Flow):
            @start()
            def begin(self) -> str:
                return "begin"

        assert PlainFlow()._should_defer_trace_finalization() is False

    def test_class_config_wins_over_a_declaration_block(self) -> None:
        llm = _ScriptedLLM(["from the class config"])

        @ConversationConfig(llm=llm, system_prompt="From the class.")
        class ConfiguredChat(Flow):
            conversational = True

        flow = ConfiguredChat.from_declaration(
            contents=_conversational_declaration(
                conversational={"system_prompt": "From the declaration."}
            )
        )

        assert flow._conversation_config.llm is llm
        assert flow._conversation_config.system_prompt == "From the class."


class TestMinimalDeclarativeChat:
    """A declaration only has to name its own routes."""

    @staticmethod
    def _declaration(**overrides: Any) -> dict[str, Any]:
        declaration: dict[str, Any] = {
            "schema": "crewai.flow/v1",
            "name": "MinimalChat",
            "conversational": {},
            "methods": {
                "handle_order": {
                    "do": {"call": "expression", "expr": "'Your order shipped.'"},
                    "listen": "order",
                    "description": "Order status questions.",
                }
            },
        }
        declaration.update(overrides)
        return declaration

    def test_minimal_declaration_runs_a_conversational_turn(self) -> None:
        # No custom routes, so the router does not auto-enable and the turn
        # goes straight to the built-in converse handler.
        declaration = self._declaration(methods={})
        flow = Flow.from_declaration(contents=declaration)
        flow._conversation_config.llm = _ScriptedLLM(["Hello there."])

        assert flow.handle_turn("hi") == "Hello there."
        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "hi"),
            ("assistant", "Hello there."),
        ]

    def test_minimal_declaration_implies_conversation_state(self) -> None:
        flow = Flow.from_declaration(contents=self._declaration())

        assert isinstance(flow.state, ConversationState)

    def test_minimal_declaration_routes_to_a_declared_handler(self) -> None:
        flow = Flow.from_declaration(contents=self._declaration())
        flow._conversation_config.router = RouterConfig(
            llm=_ScriptedLLM(['{"intent": "order"}'])
        )

        assert flow.handle_turn("where is my order?") == "Your order shipped."
        assert flow.state.last_intent == "order"

    def test_method_description_reaches_the_router_catalog(self) -> None:
        flow = Flow.from_declaration(contents=self._declaration())

        catalog = flow._build_route_catalog(RouterConfig(routes=["order"]))

        assert catalog["order"] == "Order status questions."


class TestBuiltinMethodSynthesis:
    """A declaration does not have to name the built-in methods."""

    BUILTIN_METHODS = {
        "route_conversation",
        "converse_turn",
        "end_conversation",
        "answer_from_history_turn",
    }

    @staticmethod
    def _declaration(**overrides: Any) -> dict[str, Any]:
        declaration: dict[str, Any] = {
            "schema": "crewai.flow/v1",
            "name": "SynthChat",
            "conversational": {},
            "methods": {
                "handle_order": {
                    "do": {"call": "expression", "expr": "'shipped'"},
                    "listen": "order",
                }
            },
        }
        declaration.update(overrides)
        return declaration

    def test_builtin_methods_are_synthesized_with_their_roles(self) -> None:
        flow = Flow.from_declaration(contents=self._declaration())
        methods = flow._definition.methods

        assert self.BUILTIN_METHODS <= set(methods)
        assert methods["route_conversation"].start is True
        assert methods["route_conversation"].router is True
        assert methods["converse_turn"].listen == "converse"
        assert methods["end_conversation"].listen == "end"
        assert methods["answer_from_history_turn"].listen == "answer_from_history"

    def test_synthesized_refs_match_the_python_projection(self) -> None:
        class ProjectedChat(Flow):
            conversational = True

        projected = ProjectedChat.flow_definition().methods
        synthesized = Flow.from_declaration(
            contents=self._declaration()
        )._definition.methods

        for name in self.BUILTIN_METHODS:
            assert synthesized[name].do == projected[name].do

    def test_author_supplied_entry_is_not_overridden(self) -> None:
        declaration = self._declaration()
        declaration["methods"]["converse_turn"] = {
            "do": {"call": "expression", "expr": "'mine'"},
            "listen": "converse",
        }

        flow = Flow.from_declaration(contents=declaration)

        assert flow._definition.methods["converse_turn"].do.expr == "'mine'"

    def test_disabled_block_synthesizes_nothing(self) -> None:
        flow = Flow.from_declaration(
            contents=self._declaration(conversational={"enabled": False})
        )

        assert set(flow._definition.methods) == {"handle_order"}

    def test_non_conversational_flow_synthesizes_nothing(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "Plain",
                "methods": {
                    "begin": {
                        "do": {"call": "expression", "expr": "'x'"},
                        "start": True,
                    }
                },
            }
        )

        assert set(flow._definition.methods) == {"begin"}

    def test_the_loaded_definition_itself_is_left_alone(self) -> None:
        """Synthesis is a runtime concern; the contract stays as authored."""
        from crewai.flow.flow_definition import FlowDefinition

        definition = FlowDefinition.from_declaration(contents=self._declaration())

        assert set(definition.methods) == {"handle_order"}


class TestRouteDescriptions:
    """Route descriptions survive into a declaration."""

    def test_python_handler_docstring_is_projected(self) -> None:
        class DocumentedChat(Flow):
            conversational = True

            @listen("research")
            def handle_research(self) -> str:
                """Fresh web research and current news."""
                return "researched"

        definition = DocumentedChat.flow_definition()

        assert (
            definition.methods["handle_research"].description
            == "Fresh web research and current news."
        )

    def test_declared_description_reaches_the_catalog(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "DescribedChat",
                "conversational": {},
                "methods": {
                    "handle_order": {
                        "do": {"call": "expression", "expr": "'shipped'"},
                        "listen": "order",
                        "description": "Order status questions.",
                    }
                },
            }
        )

        catalog = flow._build_route_catalog(RouterConfig(routes=["order"]))

        assert catalog["order"] == "Order status questions."

    def test_missing_description_is_empty_not_nonetype_docstring(self) -> None:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "UndescribedChat",
                "conversational": {},
                "methods": {
                    "handle_order": {
                        "do": {"call": "expression", "expr": "'shipped'"},
                        "listen": "order",
                    }
                },
            }
        )

        catalog = flow._build_route_catalog(RouterConfig(routes=["order"]))

        assert catalog["order"] == ""

    def test_python_docstring_still_describes_the_route(self) -> None:
        class DocumentedChat(Flow):
            conversational = True

            @listen("research")
            def handle_research(self) -> str:
                """Fresh web research and current news."""
                return "researched"

        catalog = DocumentedChat()._build_route_catalog(
            RouterConfig(routes=["research"])
        )

        assert catalog["research"] == "Fresh web research and current news."


class TestDeclarativeTurnMatrix:
    """A declaration-built flow across the turn entry points."""

    @staticmethod
    def _flow(reply: str = "Declared reply.") -> Flow[Any]:
        flow = Flow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "MatrixChat",
                "conversational": {},
                "methods": {},
            }
        )
        flow._conversation_config.llm = _ScriptedLLM([reply])
        return flow

    def test_sync_turn(self) -> None:
        assert self._flow().handle_turn("hi") == "Declared reply."

    def test_stream_turn_frames_match_the_class_based_path(self) -> None:
        """A chat UI must see the same frame sequence either way."""

        @ConversationConfig(llm=_ScriptedLLM(["streamed reply"]))
        class ClassChat(Flow[ConversationState]):
            pass

        class_frames = [f.type for f in ClassChat().stream_turn("hi").events]

        flow = self._flow("streamed reply")
        stream = flow.stream_turn("hi", session_id="session-1")
        declared_frames = [f.type for f in stream.events]

        assert declared_frames == class_frames
        assert declared_frames[0] == "conversation_turn_started"
        assert declared_frames[-1] == "conversation_turn_completed"
        assert "conversation_message_added" in declared_frames
        assert stream.result == "streamed reply"
        assert flow.state.messages[-1].content == "streamed reply"

    def test_turn_inside_a_running_event_loop(self) -> None:
        """``kickoff`` takes its thread-pool path when a loop is already running."""
        import asyncio

        flow = self._flow("reply from the loop")

        async def run() -> Any:
            return flow.handle_turn("hi")

        assert asyncio.run(run()) == "reply from the loop"
        assert flow.state.messages[-1].content == "reply from the loop"

    def test_follow_up_turn_keeps_history(self) -> None:
        flow = self._flow()
        flow._conversation_config.llm = _ScriptedLLM(["first", "second"])

        flow.handle_turn("one")
        flow.handle_turn("two")

        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "one"),
            ("assistant", "first"),
            ("user", "two"),
            ("assistant", "second"),
        ]

    def test_chat_repl_drives_declared_turns(self) -> None:
        flow = self._flow()
        flow._conversation_config.llm = _ScriptedLLM(["hello", "goodbye"])
        prompts: list[str] = []
        outputs: list[str] = []

        replies = iter(["hi", "bye", "exit"])

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return next(replies)

        flow.chat(input_fn=input_fn, output_fn=outputs.append)

        assert [m.content for m in flow.state.messages if m.role == "user"] == [
            "hi",
            "bye",
        ]
        assert len(outputs) == 2


class TestPrivateAgentResultsStayPrivate:
    """Unwrapping ``.raw`` must not defeat ``visible_agent_outputs``."""

    class _Out:
        """Shape of ``LiteAgentOutput`` / ``CrewOutput``."""

        def __init__(self, raw: str) -> None:
            self.raw = raw

    def test_privately_recorded_result_is_not_republished(self) -> None:
        """A handler that records privately and hands the object back.

        ``append_agent_result`` defaults to private and does not set the
        reply flag, so before this guard the end-of-turn fallback promoted the
        very object the handler had just asked to keep out of the transcript.
        """
        recorded = self._Out("SECRET scratch work")

        @ConversationConfig()
        class PrivateChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                self.append_agent_result("researcher", recorded)
                return recorded

        flow = PrivateChat()
        flow.handle_turn("look something up")

        assert [m.role for m in flow.state.messages] == ["user"]
        assert flow.state.agent_threads["researcher"][0].content == (
            "SECRET scratch work"
        )

    def test_a_summary_returned_alongside_a_private_result_is_published(self) -> None:
        """Identity, not content: a different return value is still promoted."""

        @ConversationConfig()
        class SummarisingChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                self.append_agent_result(
                    "researcher", TestPrivateAgentResultsStayPrivate._Out("SECRET")
                )
                return "Here is the summary."

        flow = SummarisingChat()
        flow.handle_turn("look something up")

        assert flow.state.messages[-1].content == "Here is the summary."

    def test_visible_agent_outputs_still_publishes(self) -> None:
        result = self._Out("PUBLIC RESULT")

        @ConversationConfig(visible_agent_outputs="all")
        class VisibleChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                self.append_agent_result("researcher", result)
                return result

        flow = VisibleChat()
        flow.handle_turn("go")

        assert flow.state.messages[-1].content == "PUBLIC RESULT"
        assert [m.role for m in flow.state.messages].count("assistant") == 1

    def test_explicit_public_visibility_still_publishes(self) -> None:
        result = self._Out("PUBLIC RESULT")

        @ConversationConfig()
        class PublicChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                self.append_agent_result("researcher", result, visibility="public")
                return result

        flow = PublicChat()
        flow.handle_turn("go")

        assert flow.state.messages[-1].content == "PUBLIC RESULT"
        assert [m.role for m in flow.state.messages].count("assistant") == 1

    def test_recorded_results_do_not_leak_across_turns(self) -> None:
        """The per-turn list resets, so turn 2 is judged on its own."""
        shared = self._Out("reply text")

        @ConversationConfig()
        class TwoTurnChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                return shared

        flow = TwoTurnChat()
        flow.handle_turn("one")
        flow.handle_turn("two")

        assert [m.content for m in flow.state.messages if m.role == "assistant"] == [
            "reply text",
            "reply text",
        ]


class TestHandlerReplyPromotion:
    """An agent or crew handler's reply reaches the transcript."""

    class _AgentOutput:
        """Shape of ``LiteAgentOutput`` / ``CrewOutput``: text lives on ``.raw``."""

        def __init__(self, raw: str) -> None:
            self.raw = raw

        def __str__(self) -> str:
            return f"<output {self.raw!r}>"

    @staticmethod
    def _chat(handler_result: Any) -> Flow[Any]:
        @ConversationConfig()
        class HandlerChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                return handler_result

        return HandlerChat()

    def test_output_object_reply_reaches_history(self) -> None:
        flow = self._chat(self._AgentOutput("Your order shipped Tuesday."))

        flow.handle_turn("where is my order?")

        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "where is my order?"),
            ("assistant", "Your order shipped Tuesday."),
        ]

    def test_string_reply_still_reaches_history(self) -> None:
        flow = self._chat("A plain string reply.")

        flow.handle_turn("hi")

        assert flow.state.messages[-1].content == "A plain string reply."

    def test_route_label_output_is_not_promoted(self) -> None:
        flow = self._chat(self._AgentOutput("converse"))

        flow.handle_turn("hi")

        assert [m.role for m in flow.state.messages] == ["user"]

    def test_output_matching_this_turns_intent_is_not_promoted(self) -> None:
        @ConversationConfig()
        class RoutedChat(Flow[ConversationState]):
            def route_turn(self, context: dict[str, Any]) -> str:
                return "order"

            @listen("order")
            def handle_order(self) -> Any:
                # Echoing the route label is a routing artefact, not a reply.
                return TestHandlerReplyPromotion._AgentOutput("order")

        flow = RoutedChat()
        flow.handle_turn("where is my order?")

        assert flow.state.last_intent == "order"
        assert [m.role for m in flow.state.messages] == ["user"]

    def test_non_text_output_is_not_promoted(self) -> None:
        flow = self._chat(self._AgentOutput(raw=None))  # type: ignore[arg-type]

        flow.handle_turn("hi")

        assert [m.role for m in flow.state.messages] == ["user"]

    def test_handler_that_already_replied_is_not_double_promoted(self) -> None:
        @ConversationConfig()
        class ExplicitReplyChat(Flow[ConversationState]):
            @listen("converse")
            def converse_turn(self) -> Any:
                self.append_assistant_message("explicit")
                return TestHandlerReplyPromotion._AgentOutput("returned")

        flow = ExplicitReplyChat()
        flow.handle_turn("hi")

        assert [(m.role, m.content) for m in flow.state.messages] == [
            ("user", "hi"),
            ("assistant", "explicit"),
        ]
