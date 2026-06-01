"""Tests for conversational Flow helpers and kickoff parameters."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.types.flow_events import (
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.experimental import (
    ConversationConfig,
    ConversationMessage,
    ConversationState,
    ConversationalFlow,
    RouterConfig,
)
from crewai.flow import Flow, ChatState, listen, start
from crewai.flow.flow_context import current_flow_id, current_flow_name
from crewai.flow.conversation import (
    ConversationalConfig,
    append_message,
    get_conversation_messages,
    normalize_kickoff_inputs,
    prepare_conversational_turn,
)
from crewai.state import CheckpointConfig
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


class TestConversationalFlow:
    def test_deferred_multi_turn_trace_keeps_event_sequence_continuous(
        self,
    ) -> None:
        @ConversationConfig()
        class TraceFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                reply = f"worked: {self.state.current_user_message}"
                self.append_assistant_message(reply)
                return reply

        events: list[Any] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(FlowStartedEvent)
            def capture_flow_started(source: Any, event: Any) -> None:
                events.append(event)

            @crewai_event_bus.on(MethodExecutionStartedEvent)
            def capture_method_started(source: Any, event: Any) -> None:
                events.append(event)

            @crewai_event_bus.on(MethodExecutionFinishedEvent)
            def capture_method_finished(source: Any, event: Any) -> None:
                events.append(event)

            flow = TraceFlow()
            flow.handle_turn("research apple stock")
            flow.handle_turn("research google stock")
            flow.finalize_session_traces()
            crewai_event_bus.flush()

        flow_started_events = [
            event for event in events if isinstance(event, FlowStartedEvent)
        ]
        method_events = [
            event
            for event in events
            if isinstance(
                event, MethodExecutionStartedEvent | MethodExecutionFinishedEvent
            )
        ]
        sequences = [
            event.emission_sequence
            for event in events
            if event.emission_sequence is not None
        ]
        assert len(flow_started_events) == 1
        assert len(sequences) == len(set(sequences))
        assert all(
            event.parent_event_id == flow_started_events[0].event_id
            for event in method_events
        )

    def test_handle_turn_defers_trace_until_session_finalize(self) -> None:
        from crewai.events.listeners.tracing.trace_batch_manager import TraceBatch

        @ConversationConfig()
        class TraceFlow(ConversationalFlow):
            def route_turn(self, context: dict[str, Any]) -> str | None:
                return "work"

            @listen("work")
            def do_work(self) -> str:
                self.append_assistant_message("done")
                return "done"

        flow = TraceFlow()
        listener = TraceCollectionListener()
        listener.batch_manager.current_batch = TraceBatch()
        listener.batch_manager.batch_owner_type = "flow"
        listener.batch_manager._batch_finalized = False
        try:
            with patch.object(flow, "_finalize_flow_trace_batch") as mock_finalize:
                flow.handle_turn("hello")

                assert flow.defer_trace_finalization is True
                assert flow._should_defer_trace_finalization() is True
                mock_finalize.assert_called_once_with()

                flow.finalize_session_traces()

            assert mock_finalize.call_args_list[-1] == ((), {"force": True})
        finally:
            listener.batch_manager.current_batch = None
            listener.batch_manager.batch_owner_type = None
            listener.batch_manager.defer_session_finalization = False
            listener.batch_manager._batch_finalized = False

    def test_handle_turn_delegates_to_restored_checkpoint_flow(self) -> None:
        class CheckpointFlow(ConversationalFlow):
            pass

        flow = CheckpointFlow()
        mock_restored = MagicMock(spec=CheckpointFlow)
        mock_restored.kickoff.return_value = "restored reply"

        cfg = CheckpointConfig(restore_from="/path/to/conversation_cp.json")
        with patch.object(CheckpointFlow, "from_checkpoint", return_value=mock_restored):
            result = flow.handle_turn("resume this chat", from_checkpoint=cfg)

        mock_restored.kickoff.assert_called_once_with(
            inputs={
                "id": flow.state.id,
                "user_message": "resume this chat",
            },
            input_files=None,
            user_message="resume this chat",
            session_id=flow.state.id,
        )
        assert mock_restored.checkpoint.restore_from is None
        assert result == "restored reply"

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
        assert "conversation_start" in ResearchFlow._start_methods
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
        which would race with ``conversation_start`` and let the router fire
        before user setup finished. In conversational mode the framework runs
        them sequentially, with ``conversation_start`` last.
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

    def test_subclass_can_override_conversation_start_without_redecorating(
        self,
    ) -> None:
        """Overriding an inherited ``@start`` method must not unregister it.

        Before the metaclass fix, subclasses had to re-apply ``@start()`` on
        every override or the parent's ``conversation_start`` would silently
        drop out of ``_start_methods`` — leaving the flow with nothing to fire.
        """

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
        assert "conversation_start" in flow._start_methods

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

    def test_method_execution_emitted_when_panel_events_suppressed(self) -> None:
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

        assert started == ["begin"]
        assert finished == ["begin"]

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

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.first_time_handler.is_first_time = False

        with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
            flow._finalize_flow_trace_batch()
            mock_finalize.assert_not_called()

            flow._finalize_flow_trace_batch(force=True)
            mock_finalize.assert_called_once()


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

    def test_finalize_flow_trace_batch_respects_defer_session_flag(self) -> None:
        """Nested Flow kickoffs (e.g. AgentExecutor) must not finalize a deferred session batch."""

        class InnerFlow(Flow[ChatState]):
            @start()
            def begin(self) -> str:
                return "ok"

        listener = TraceCollectionListener()
        listener.batch_manager.batch_owner_type = "flow"
        listener.batch_manager.defer_session_finalization = True
        listener.first_time_handler.is_first_time = False

        inner = InnerFlow()
        with patch.object(listener.batch_manager, "finalize_batch") as mock_finalize:
            inner._finalize_flow_trace_batch()
        mock_finalize.assert_not_called()

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
