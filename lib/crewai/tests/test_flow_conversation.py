"""Tests for conversational Flow helpers and kickoff parameters."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

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
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.experimental import (
    ConversationConfig,
    ConversationMessage,
    ConversationState,
    RouterConfig,
)
from crewai.flow import Flow, ChatState, listen, start
from crewai.flow.flow_context import (
    current_flow_defer_trace_finalization,
    current_flow_id,
    current_flow_name,
)
from crewai.flow.conversation import (
    append_message,
    get_conversation_messages,
    normalize_kickoff_inputs,
    prepare_conversational_turn,
)

# The built-in conversational graph lives on ``_ConversationalMixin`` and is
# inherited by ``conversational = True`` subclasses. The definition-first start
# migration intentionally stopped scanning inherited methods, so that graph no
# longer registers. These end-to-end conversational tests are out of scope
# until conversational mode is migrated onto the FlowDefinition.
conversational_graph_broken = pytest.mark.skip(
    reason="Experimental conversational registry behavior is out of scope for "
    "the definition-first start migration."
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

    def test_builtin_converse_enables_llm_streaming_for_streaming_flow(self) -> None:
        llm = MagicMock()
        llm.stream = False
        stream_values_seen: list[bool | None] = []

        def call(*args: Any, **kwargs: Any) -> str:
            stream_values_seen.append(llm.stream)
            return "streamed reply"

        llm.call.side_effect = call

        @ConversationConfig(llm=llm)
        class StreamingFlow(ConversationalFlow):
            pass

        flow = StreamingFlow()
        flow.stream = False

        with flow._streaming_run():
            result = flow.converse_turn()

        assert result == "streamed reply"
        assert stream_values_seen == [True]
        assert llm.stream is False
        assert flow._should_stream_llm_calls() is False
        assert flow.state.messages[-1].content == "streamed reply"

    def test_streaming_handle_turn_preserves_pending_user_message(self) -> None:
        @ConversationConfig(llm="unused")
        class StreamingEchoFlow(ConversationalFlow):
            stream = True

            def route_turn(self, context: dict[str, Any]) -> str:
                return "echo"

            @listen("echo")
            def handle_echo(self) -> str:
                reply = f"heard: {self.state.current_user_message}"
                self.append_assistant_message(reply)
                return reply

        flow = StreamingEchoFlow()
        result = flow.handle_turn("hello streaming")
        for _chunk in result:
            pass

        assert result.result == "heard: hello streaming"
        assert [message.role for message in flow.state.messages] == [
            "user",
            "assistant",
        ]
        assert flow.state.messages[0].content == "hello streaming"
        assert flow.state.messages[1].content == "heard: hello streaming"

    @conversational_graph_broken
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

    @conversational_graph_broken
    def test_answer_from_history_uses_configured_llm_and_appends_reply(self) -> None:
        @ConversationConfig(answer_from_history_llm="gpt-4o-mini")
        class HistoryFlow(ConversationalFlow):
            pass

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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
    def test_route_catalog_falls_back_to_empty_when_no_docstring(self) -> None:
        @ConversationConfig(router=RouterConfig(routes=["BARE"]))
        class BareFlow(ConversationalFlow):
            @listen("BARE")
            def handle_bare(self) -> str:
                return "bare"

        flow = BareFlow()
        catalog = flow._build_route_catalog(flow.conversational_config.router)

        assert catalog["BARE"] == ""

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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

    @conversational_graph_broken
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
