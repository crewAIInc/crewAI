"""Conversational graph + helpers as an experimental Flow extension.

The conversational chat surface remains experimental and may change before the
v2 graduation path. It lives here so ``crewai.flow.runtime`` can stay focused
on the execution engine. ``crewai.flow.flow`` composes this mixin onto the
public ``Flow`` class for backwards compatibility.

The built-in conversational graph only registers for subclasses that opt in
with ``conversational = True``. Static conversational metadata is projected
into ``FlowDefinition.conversational`` via the Python DSL builder.

Import surface:
    - :class:`_ConversationalMixin` — internal; the public ``Flow`` class
      composes it in. Users don't import it directly.
    - The data types this mixin uses live in
      :mod:`crewai.experimental.conversational`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    ConversationMessageAddedEvent,
    ConversationRouteSelectedEvent,
    ConversationTurnCompletedEvent,
    ConversationTurnFailedEvent,
    ConversationTurnStartedEvent,
)
from crewai.experimental.conversational import (
    AgentMessage,
    ConversationConfig,
    ConversationEvent,
    ConversationMessage,
    ConversationState,
    RouterConfig,
    _conversational_only,
    message_to_llm_dict,
)
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.conversation import (
    append_message as _append_conversation_message,
    get_conversation_messages,
    receive_user_message as _receive_user_message,
)
from crewai.flow.conversational_definition import (
    FlowConversationalDefinition,
    FlowConversationalRouterDefinition,
)
from crewai.flow.dsl import listen, start
from crewai.flow.dsl._utils import _method_action, _set_flow_method_definition
from crewai.flow.flow_definition import FlowDefinition, FlowMethodDefinition
from crewai.project.crew_definition import PythonReferenceDefinition
from crewai.flow.types import FlowMethodName
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM


logger = logging.getLogger(__name__)


def _iter_condition_labels(condition: Any) -> set[str]:
    if isinstance(condition, str):
        return {condition}
    if isinstance(condition, dict):
        labels: set[str] = set()
        for value in condition.values():
            if isinstance(value, list):
                for item in value:
                    labels.update(_iter_condition_labels(item))
            else:
                labels.update(_iter_condition_labels(value))
        return labels
    return set()


def _resolve_router_response_format(
    declared: PythonReferenceDefinition | None,
    project_root: Path | None,
) -> type[BaseModel] | None:
    """Import the model class a declaration names for the routing decision.

    ``_router_response_format`` hands its result straight to
    ``llm.call(response_format=...)``, which needs a real class. Resolved the
    same way a crew agent's ``response_format`` is, so the declaration shape
    and its failure messages match -- including resolving the ref against the
    declaration's own directory, so a flow loaded by path finds the model
    sitting next to it rather than under whatever the cwd happens to be.
    """
    if declared is None:
        return None

    from crewai.project.json_loader import _resolve_model_class

    return _resolve_model_class(
        declared.model_dump(),
        "conversational.router.response_format",
        project_root,
    )


def _router_config_from_definition(
    definition: FlowConversationalRouterDefinition,
    project_root: Path | None,
) -> RouterConfig:
    """Build a live ``RouterConfig`` from its serializable form."""
    response_format = _resolve_router_response_format(
        definition.response_format, project_root
    )

    return RouterConfig(
        prompt=definition.prompt,
        response_format=response_format,
        llm=definition.llm,
        routes=definition.routes,
        route_descriptions=definition.route_descriptions,
        default_intent=definition.default_intent,
        fallback_intent=definition.fallback_intent,
        intent_field=definition.intent_field,
    )


def _config_from_definition(
    definition: FlowConversationalDefinition,
    project_root: Path | None,
) -> ConversationConfig:
    """Build a live ``ConversationConfig`` from its serializable form.

    Used for flows built from a declaration, which have no class-level
    ``conversational_config`` to read. ``project_root`` is the declaration
    file's directory, used to resolve the Python refs it names.
    """
    return ConversationConfig(
        system_prompt=definition.system_prompt,
        llm=definition.llm,
        router=(
            _router_config_from_definition(definition.router, project_root)
            if definition.router is not None
            else None
        ),
        answer_from_history_prompt=definition.answer_from_history_prompt,
        default_intents=definition.default_intents,
        intent_llm=definition.intent_llm,
        answer_from_history_llm=definition.answer_from_history_llm,
        visible_agent_outputs=definition.visible_agent_outputs,
        defer_trace_finalization=definition.defer_trace_finalization,
    )


def _conversation_state_with_defaults(
    defaults: dict[str, Any] | None,
) -> ConversationState:
    """Build the chat state, keeping declared defaults the shape cannot hold.

    A ``dict`` state's defaults are arbitrary keys, and dropping the ones that
    are not conversational fields would break an action reading
    ``state.topic``. They are carried as extras instead.
    """
    if not defaults:
        return ConversationState()

    known = {
        name: value
        for name, value in defaults.items()
        if name in ConversationState.model_fields
    }
    extra = {
        name: value
        for name, value in defaults.items()
        if name not in ConversationState.model_fields
    }
    if not extra:
        return ConversationState(**known)

    class ConversationStateWithDeclaredDefaults(ConversationState):
        model_config = ConfigDict(extra="allow")

    return ConversationStateWithDeclaredDefaults(**known, **extra)


def _builtin_method(handler: Callable[..., Any], **roles: Any) -> FlowMethodDefinition:
    """One built-in conversational method, as the code ref the DSL already emits."""
    return FlowMethodDefinition(do=_method_action(handler), **roles)


def _builtin_methods() -> dict[str, FlowMethodDefinition]:
    """The built-in methods a conversational flow needs to run a turn.

    A Python flow inherits these methods; a declaration has nothing to inherit
    from, so without them ``conversational: {}`` loads clean and then runs zero
    methods and returns ``None``.
    """
    return {
        "route_conversation": _builtin_method(
            _ConversationalMixin.route_conversation, start=True, router=True
        ),
        "converse_turn": _builtin_method(
            _ConversationalMixin.converse_turn, listen="converse"
        ),
        "end_conversation": _builtin_method(
            _ConversationalMixin.end_conversation, listen="end"
        ),
        "answer_from_history_turn": _builtin_method(
            _ConversationalMixin.answer_from_history_turn,
            listen="answer_from_history",
        ),
    }


def _conversation_start_router(func: Callable[..., Any]) -> Any:
    wrapper = start()(func)
    _set_flow_method_definition(
        cast(Any, wrapper),
        FlowMethodDefinition(do=_method_action(func), start=True, router=True),
    )
    return wrapper


class _ConversationalMixin:
    """Experimental conversational graph for ``Flow``.

    This mixin owns chat behavior and runtime hooks. Non-chat flows see these
    methods as inert attributes unless they opt in with ``conversational = True``.
    """

    # === EXPERIMENTAL: conversational mode ===
    # When ``conversational = True`` on a Flow subclass, this mixin's built-in
    # graph registers and ``handle_turn`` / ``chat`` become chat entry points.
    conversational: ClassVar[bool] = False
    conversational_config: ClassVar[ConversationConfig | None] = None
    builtin_routes: ClassVar[tuple[str, ...]] = ("converse", "end")
    internal_routes: ClassVar[tuple[str, ...]] = ("answer_from_history",)
    builtin_route_descriptions: ClassVar[dict[str, str]] = {
        "converse": (
            "Ordinary chat, follow-ups, summaries, clarifications, and "
            "questions answerable from prior conversation history."
        ),
        "end": ("User signals the conversation is finished (goodbye, exit, done)."),
        "answer_from_history": (
            "Answer directly from prior conversation history without invoking "
            "tools, agents, or custom routes."
        ),
    }

    # The metaclass + state attributes referenced below live on ``Flow`` —
    # this mixin is never instantiated standalone. These type-only
    # declarations exist so static analyzers don't flag attribute access.
    # Class-level slots use ``ClassVar`` to match Flow's actual declarations
    # (otherwise mypy flags "Cannot override instance variable with class
    # variable" when Flow declares them as ``ClassVar``).
    if TYPE_CHECKING:
        # Instance attrs from ``Flow``.
        state: Any
        name: str | None
        _completed_methods: set[Any]
        _method_outputs: list[Any]
        _pending_events: dict[Any, Any]
        _method_call_counts: dict[Any, int]
        _is_execution_resuming: bool
        _conversation_messages: list[LLMMessage]
        _pending_user_message: str | dict[str, Any] | None
        _pending_intents: Sequence[str] | None
        _pending_intent_llm: str | BaseLLM | None
        _turn_classified_intent: str | None
        _assistant_reply_appended: bool
        _turn_recorded_results: list[Any]
        _resolved_conversation_config: ConversationConfig | None

        def _clear_or_listeners(self) -> None:
            pass

        def _collapse_to_outcome(
            self,
            feedback: str,
            outcomes: Sequence[str],
            llm: str | BaseLLM,
        ) -> str:
            pass

        def _copy_and_serialize_state(self) -> dict[str, Any]:
            pass

        def kickoff(self, *args: Any, **kwargs: Any) -> Any:
            pass

        def _persist_method_completion(self, method_name: FlowMethodName) -> None:
            pass

        @property
        def method_outputs(self) -> list[Any]:
            pass

    def conversation_start(self) -> str | None:
        """Return the current user message for conversational route selection.

        This remains as a plain overridable helper for compatibility. It is not
        registered as a Flow method; ``route_conversation`` is the synthetic
        built-in start/router that begins a conversational turn.
        """
        state = cast(ConversationState, self.state)
        return state.current_user_message

    @_conversation_start_router
    @_conversational_only
    def route_conversation(self) -> str:
        """Route the current turn to a listener label."""
        if "conversation_start" not in {
            str(method_name) for method_name in self._completed_methods
        }:
            self.conversation_start()
        state = cast(ConversationState, self.state)
        context = self.build_router_context()
        previous_intent = state.last_intent
        configured_route = self.route_turn(context)
        if configured_route:
            state.last_intent = configured_route
            self._emit_conversation_route_selected(
                configured_route,
                previous_intent=previous_intent,
            )
            return configured_route

        turn_intent = self._turn_classified_intent
        if turn_intent:
            state.last_intent = turn_intent
            self._emit_conversation_route_selected(
                turn_intent,
                previous_intent=previous_intent,
            )
            return turn_intent

        if previous_intent:
            logger.debug(
                "route_turn() returned no route and no intent was classified "
                "this turn; ignoring stale last_intent=%r from a previous turn "
                "and falling back to built-in routing",
                previous_intent,
            )

        if self.can_answer_from_history(context):
            state.last_intent = "answer_from_history"
            self._emit_conversation_route_selected(
                "answer_from_history",
                previous_intent=previous_intent,
            )
            return "answer_from_history"

        state.last_intent = "converse"
        self._emit_conversation_route_selected(
            "converse",
            previous_intent=previous_intent,
        )
        return "converse"

    @listen("converse")
    @_conversational_only
    def converse_turn(self) -> str:
        """Built-in chat handler over canonical conversation history."""
        llm = self._default_conversation_llm()
        if llm is None:
            content = "I can continue the conversation once an LLM is configured."
            self.append_assistant_message(content)
            return content

        messages: list[LLMMessage] = []
        system_prompt = self._resolve_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_messages)

        llm_instance = self._coerce_llm(llm)
        with self._conversation_streaming_enabled(llm_instance):
            response = llm_instance.call(messages=messages)
        content = self._stringify_result(response)
        self.append_assistant_message(content)
        return content

    @listen("end")
    @_conversational_only
    def end_conversation(self) -> str:
        """Built-in conversation terminator."""
        cast(ConversationState, self.state).ended = True
        content = "Conversation ended."
        self.append_assistant_message(content)
        return content

    @listen("answer_from_history")
    @_conversational_only
    def answer_from_history_turn(self) -> str | None:
        """Deprecated history-only handler retained for compatibility.

        Use the built-in ``converse`` route, which already receives canonical
        conversation history, or override ``converse_turn``.
        """
        config = self._conversation_config
        if config is None:
            return None
        llm = config.answer_from_history_llm
        if llm is None:
            return None

        llm_instance = self._coerce_llm(llm)
        messages: list[LLMMessage] = [
            {
                "role": "system",
                "content": self._resolve_answer_from_history_prompt(),
            },
            *self.build_agent_context("answer_from_history"),
        ]
        with self._conversation_streaming_enabled(llm_instance):
            response = llm_instance.call(messages=messages)
        content = self._stringify_result(response)
        self.append_assistant_message(content)
        return content

    def handle_turn(
        self,
        message: str,
        *,
        session_id: str | None = None,
        intents: Sequence[str] | None = None,
        intent_llm: str | BaseLLM | None = None,
        **kickoff_kwargs: Any,
    ) -> Any:
        """Append a user message, run one conversational turn, and return output.

        .. warning::

           **EXPERIMENTAL.** This is the public entry point for the
           conversational ``Flow``. Signature and semantics may change before
           the feature graduates from ``crewai.experimental``.

        Available only when ``conversational = True`` is set on the subclass
        (``@ConversationConfig`` sets it for you). Stashes the message +
        session_id as pending turn state, runs kickoff (which restores from
        persist and then applies the pending turn), and promotes the result to
        an assistant message when the handler didn't.
        If that fallback appends, state is snapshotted again so ``@persist``
        restores the assistant reply on a fresh Flow instance.

        Raises:
            ValueError: If the flow is not conversational. Without this the
                turn is a silent no-op: no route handlers are registered, the
                message is never appended, and ``None`` comes back.
        """
        if not self._is_conversational_enabled():
            raise ValueError(
                "Flow.handle_turn() is only available on conversational flows"
            )

        state = cast(ConversationState, self.state)
        sid = session_id or state.id
        crewai_event_bus.emit(
            self,
            ConversationTurnStartedEvent(
                type="conversation_turn_started",
                flow_name=self.name or self.__class__.__name__,
                session_id=sid,
            ),
        )

        # Stash the pending turn so the kickoff extension hook picks it up
        # after persist restore.
        self._pending_user_message = message
        self._pending_intents = list(intents) if intents else None
        self._pending_intent_llm = intent_llm

        failed_event: ConversationTurnFailedEvent | None = None
        try:
            # Each turn is a fresh execution; clear graph tracking so the second
            # turn re-runs instead of being treated as a checkpoint restore.
            if "from_checkpoint" not in kickoff_kwargs:
                self._reset_turn_execution_state()

            object.__setattr__(self, "_assistant_reply_appended", False)
            object.__setattr__(self, "_turn_recorded_results", [])
            result = self.kickoff(inputs={"id": sid}, **kickoff_kwargs)
            self._append_public_turn_result_and_persist(result)
        except Exception as exc:
            failed_event = ConversationTurnFailedEvent(
                type="conversation_turn_failed",
                flow_name=self.name or self.__class__.__name__,
                session_id=sid,
                error=exc,
            )
            raise
        finally:
            self._pending_user_message = None
            self._pending_intents = None
            self._pending_intent_llm = None
            if failed_event is not None:
                self._emit_terminal_conversation_turn_event(failed_event)

        self._emit_terminal_conversation_turn_event(
            ConversationTurnCompletedEvent(
                type="conversation_turn_completed",
                flow_name=self.name or self.__class__.__name__,
                session_id=sid,
            ),
        )
        return result

    def stream_turn(
        self,
        message: str,
        *,
        session_id: str | None = None,
        intents: Sequence[str] | None = None,
        intent_llm: str | BaseLLM | None = None,
        **kickoff_kwargs: Any,
    ) -> Any:
        """Append a user message and stream one conversational turn as frames."""
        if not self._is_conversational_enabled():
            raise ValueError(
                "Flow.stream_turn() is only available on conversational flows"
            )

        from crewai.types.streaming import StreamSession
        from crewai.utilities.streaming import (
            create_frame_generator,
            create_frame_streaming_state,
        )

        state = cast(ConversationState, self.state)
        sid = session_id or state.id
        result_holder: list[Any] = []
        frame_state = create_frame_streaming_state(result_holder, use_async=False)
        output_holder: list[StreamSession[Any]] = []

        def run_turn() -> Any:
            crewai_event_bus.emit(
                self,
                ConversationTurnStartedEvent(
                    type="conversation_turn_started",
                    flow_name=self.name or self.__class__.__name__,
                    session_id=sid,
                ),
            )

            self._pending_user_message = message
            self._pending_intents = list(intents) if intents else None
            self._pending_intent_llm = intent_llm

            try:
                if "from_checkpoint" not in kickoff_kwargs:
                    self._reset_turn_execution_state()

                object.__setattr__(self, "_assistant_reply_appended", False)
                object.__setattr__(self, "_turn_recorded_results", [])
                original_stream = bool(getattr(self, "stream", False))
                original_streaming_turn = getattr(
                    self, "_streaming_conversation_turn", False
                )
                try:
                    object.__setattr__(self, "stream", False)
                    object.__setattr__(self, "_streaming_conversation_turn", True)
                    result = self.kickoff(inputs={"id": sid}, **kickoff_kwargs)
                finally:
                    object.__setattr__(self, "stream", original_stream)
                    object.__setattr__(
                        self, "_streaming_conversation_turn", original_streaming_turn
                    )
                self._append_public_turn_result_and_persist(result)
            except HumanFeedbackPending as exc:
                return exc
            except Exception as exc:
                failed_event = ConversationTurnFailedEvent(
                    type="conversation_turn_failed",
                    flow_name=self.name or self.__class__.__name__,
                    session_id=sid,
                    error=exc,
                )
                self._emit_terminal_conversation_turn_event(failed_event)
                raise
            finally:
                self._pending_user_message = None
                self._pending_intents = None
                self._pending_intent_llm = None

            self._emit_terminal_conversation_turn_event(
                ConversationTurnCompletedEvent(
                    type="conversation_turn_completed",
                    flow_name=self.name or self.__class__.__name__,
                    session_id=sid,
                ),
            )
            return result

        stream_session: StreamSession[Any] = StreamSession(
            sync_iterator=create_frame_generator(frame_state, run_turn, output_holder)
        )
        output_holder.append(stream_session)
        return stream_session

    def _emit_terminal_conversation_turn_event(
        self,
        event: ConversationTurnCompletedEvent | ConversationTurnFailedEvent,
    ) -> None:
        """Emit a terminal turn event and wait for its own handlers."""
        future = crewai_event_bus.emit(self, event)
        if future is None:
            return
        try:
            future.result(timeout=30)
        except Exception:
            logger.warning(
                "%s handler failed or timed out",
                event.__class__.__name__,
                exc_info=True,
            )

    def chat(
        self,
        *,
        session_id: str | None = None,
        prompt: str = "\nYou: ",
        assistant_prefix: str = "\nAssistant: ",
        exit_commands: Sequence[str] = ("exit", "quit"),
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        skip_empty: bool = True,
        defer_trace_finalization: bool = True,
        **handle_turn_kwargs: Any,
    ) -> None:
        """Run an interactive terminal chat loop for a conversational Flow.

        ``chat()`` is a convenience wrapper around ``handle_turn()`` for local
        REPLs. For web apps, tests, and custom transports, call
        ``handle_turn()`` directly. The input/output callables are injectable so
        callers can customize prompts or exercise the loop without patching
        builtins.
        """
        if not self._is_conversational_enabled():
            raise ValueError("Flow.chat() is only available on conversational flows")

        exit_set = {command.lower() for command in exit_commands}
        previous_defer = getattr(self, "defer_trace_finalization", False)
        if defer_trace_finalization:
            self.defer_trace_finalization = True

        try:
            while True:
                try:
                    message = input_fn(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    output_fn("")
                    break

                if message.lower() in exit_set:
                    break
                if skip_empty and not message:
                    continue

                result = self.handle_turn(
                    message,
                    session_id=session_id,
                    **handle_turn_kwargs,
                )
                output_fn(f"{assistant_prefix}{self._stringify_result(result)}")
        finally:
            self.finalize_session_traces()
            if defer_trace_finalization:
                self.defer_trace_finalization = previous_defer

    def build_router_context(self) -> dict[str, Any]:
        """Build context used by the routing policy for the current turn."""
        state = cast(ConversationState, self.state)
        return {
            "system_prompt": self._resolve_system_prompt(),
            "current_user_message": state.current_user_message,
            "message_history": self.conversation_messages,
            "events": [event.model_dump() for event in state.events],
            "last_intent": state.last_intent,
        }

    def build_agent_context(self, agent_name: str) -> list[LLMMessage]:
        """Build canonical message context for an agent or direct LLM call."""
        state = cast(ConversationState, self.state)
        messages = list(self.conversation_messages)
        thread = state.agent_threads.get(agent_name, [])
        messages.extend(
            cast(
                LLMMessage,
                {
                    "role": msg.role,
                    "content": self._stringify_result(msg.content),
                },
            )
            for msg in thread
        )
        return messages

    def route_turn(self, context: dict[str, Any]) -> str | None:
        """Route the current turn via the LLM router.

        When ``ConversationConfig.router`` is omitted, the router is
        auto-enabled with default settings as long as the flow declares
        custom ``@listen`` handlers (anything beyond the built-in
        ``converse`` / ``end`` routes). ``@ConversationConfig(llm=ROUTER_LLM)``
        is enough to dispatch to your custom handlers — no explicit
        ``RouterConfig()`` needed.

        Pass an explicit ``RouterConfig`` only to override the routing prompt,
        supply per-route descriptions, or change the default/fallback intent.
        Override this method to bypass the LLM router entirely (e.g.,
        permission gates before the LLM decision).

        Returning a falsy value means "no routing decision": the turn falls
        through to the deprecated ``answer_from_history`` compatibility path
        when configured, then ``converse``. It never replays a previous turn's
        intent.
        """
        config = self._conversation_config
        if config is None:
            return None

        router_config = config.router
        if router_config is None:
            if config.default_intents:
                return None
            custom_routes = (
                self._effective_routes(None) - self._effective_builtin_routes()
            )
            if not custom_routes:
                return None
            router_config = RouterConfig()

        return self._route_with_config(router_config, context)

    def can_answer_from_history(self, context: dict[str, Any]) -> bool:
        """Return whether the deprecated history-only route can answer."""
        config = self._conversation_config
        if config is None or config.answer_from_history_llm is None:
            return False
        if len(self.conversation_messages) < 2:
            return False

        feedback = (
            f"{self._resolve_answer_from_history_prompt()}\n\n"
            f"Current user message: {context.get('current_user_message')}\n\n"
            f"Message history:\n{self._format_messages(self.conversation_messages)}"
        )
        outcome = self._collapse_to_outcome(
            feedback,
            ("answer_from_history", "route_to_flow"),
            self._coerce_llm(config.answer_from_history_llm),
        )
        return outcome == "answer_from_history"

    def append_agent_result(
        self,
        agent_name: str,
        result: Any,
        *,
        visibility: Literal["private", "public"] = "private",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent result, optionally making it visible to the user."""
        content = self._stringify_result(result)
        event_visibility = self._resolve_visibility(agent_name, visibility)
        event = ConversationEvent(
            type="agent_result",
            agent_name=agent_name,
            visibility=event_visibility,
            payload={"content": content, **(metadata or {})},
        )
        state = cast(ConversationState, self.state)
        state.events.append(event)
        state.agent_threads.setdefault(agent_name, []).append(
            AgentMessage(content=content, metadata=metadata or {})
        )
        # Remember the object itself so the end-of-turn fallback does not
        # re-publish a result the handler deliberately kept private. Identity,
        # not content: a handler that records scratch work privately and then
        # returns a user-facing summary still gets that summary promoted.
        self._turn_recorded_results.append(result)
        if event_visibility == "public":
            self.append_assistant_message(content)

    def append_assistant_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a final user-visible assistant message."""
        # Explicit signal for handle_turn's "did the handler reply?" check.
        # A count heuristic breaks when handlers trim history mid-turn.
        object.__setattr__(self, "_assistant_reply_appended", True)
        state = cast(ConversationState, self.state)
        state.messages.append(
            ConversationMessage(
                role="assistant",
                content=content,
                metadata=metadata or {},
            )
        )
        self._emit_conversation_message_added(
            role="assistant",
            content=content,
            message_index=len(state.messages) - 1,
        )

    def _emit_conversation_message_added(
        self,
        *,
        role: Literal["user", "assistant", "system", "tool"],
        content: Any,
        message_index: int,
    ) -> None:
        """Emit a compact transcript event for conversational trace views."""
        state = cast(ConversationState, self.state)
        crewai_event_bus.emit(
            self,
            ConversationMessageAddedEvent(
                type="conversation_message_added",
                flow_name=self.name or self.__class__.__name__,
                session_id=state.id,
                role=role,
                content=content,
                message_index=message_index,
            ),
        )

    def _emit_conversation_route_selected(
        self,
        route: str,
        *,
        previous_intent: str | None = None,
    ) -> None:
        """Emit the conversational routing decision for the current turn."""
        state = cast(ConversationState, self.state)
        crewai_event_bus.emit(
            self,
            ConversationRouteSelectedEvent(
                type="conversation_route_selected",
                flow_name=self.name or self.__class__.__name__,
                session_id=state.id,
                route=route,
                user_message=state.current_user_message,
                message_index=(len(state.messages) - 1) if state.messages else None,
                previous_intent=previous_intent,
            ),
        )

    def append_message(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        **extra: Any,
    ) -> None:
        """Append a message to conversation history (legacy ChatState path)."""
        _append_conversation_message(cast(Any, self), role, content, **extra)

    @property
    def conversation_messages(self) -> list[LLMMessage]:
        """Message history from state, coerced to LLM-shaped dicts."""
        return [
            message_to_llm_dict(message)
            for message in get_conversation_messages(cast(Any, self))
        ]

    def receive_user_message(
        self,
        text: str,
        *,
        outcomes: Sequence[str] | None = None,
        llm: str | BaseLLM | None = None,
    ) -> str:
        """Append a user message and optionally classify intent.

        Conversational flows push a ``ConversationMessage`` onto
        ``state.messages`` and preserve ``last_intent`` across turns.
        Non-conversational flows fall through to the legacy helper.
        """
        if self._is_conversational_enabled():
            state = cast(ConversationState, self.state)
            state.messages.append(ConversationMessage(role="user", content=text))
            self._emit_conversation_message_added(
                role="user",
                content=text,
                message_index=len(state.messages) - 1,
            )
            state.current_user_message = text
            state.last_user_message = text
            if outcomes and llm is not None:
                intent = self.classify_intent(
                    text,
                    outcomes,
                    llm=llm,
                    context=self.conversation_messages,
                )
                state.last_intent = intent
                object.__setattr__(self, "_turn_classified_intent", intent)
                return intent
            return text

        return _receive_user_message(cast(Any, self), text, outcomes=outcomes, llm=llm)

    def classify_intent(
        self,
        text: str,
        outcomes: Sequence[str],
        *,
        llm: str | BaseLLM,
        context: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Map user text to one of the given outcomes using an LLM."""
        if context:
            context_blob = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in context
            )
            feedback = f"{context_blob}\n\nLatest user message: {text}"
        else:
            feedback = text
        # ``_collapse_to_outcome`` takes only ``str | BaseLLM``, so a declared
        # config mapping has to be resolved before it gets there.
        return self._collapse_to_outcome(
            feedback, tuple(outcomes), self._coerce_llm(llm)
        )

    @property
    def _conversation_config(self) -> ConversationConfig | None:
        """Resolve the live conversational configuration for this flow.

        The class-level ``conversational_config`` wins when present: it can
        hold live objects — a configured ``LLM``, a custom ``BaseLLM``, a
        ``response_format`` model class — that the serializable definition
        degrades to a config dict or a ``module:qualname`` ref. Reading the
        definition first would silently downgrade every decorated Python flow.

        A flow built from a declaration has no class config, so its
        ``conversational`` block supplies one.
        """
        config: ConversationConfig | None = getattr(
            type(self), "conversational_config", None
        )
        if config is not None:
            return config

        # Cached so the resolved config has stable identity, matching the
        # class-config path. Callers read it many times per turn and a fresh
        # object each time would make `config.llm is llm` false.
        resolved: ConversationConfig | None = getattr(
            self, "_resolved_conversation_config", None
        )
        if resolved is not None:
            return resolved

        flow_definition = self._conversation_flow_definition()
        definition = flow_definition.conversational
        if definition is None or not definition.enabled:
            return None

        resolved = _config_from_definition(definition, flow_definition.source_dir)
        object.__setattr__(self, "_resolved_conversation_config", resolved)
        return resolved

    @property
    def _conversation_definition(self) -> FlowConversationalDefinition | None:
        return self._conversation_flow_definition().conversational

    def _conversation_flow_definition(self) -> FlowDefinition:
        """Return the definition that describes this flow's methods.

        Prefers the instance definition, which is the loaded declaration for a
        flow built with ``from_declaration`` and the class projection
        otherwise. Falls back to the class projection while ``_definition`` is
        still unset — ``_initialize_runtime_extension_attrs`` runs one step
        before it is assigned in ``_flow_post_init``.
        """
        definition: FlowDefinition | None = getattr(self, "_definition", None)
        if definition is not None:
            return definition

        flow_definition = getattr(type(self), "flow_definition", None)
        if not callable(flow_definition):
            raise AttributeError(
                f"{type(self).__name__} does not expose flow_definition()"
            )
        return cast(FlowDefinition, flow_definition())

    def _is_conversational_enabled(self) -> bool:
        definition = self._conversation_definition
        return bool(definition and definition.enabled)

    def _initialize_runtime_extension_attrs(self) -> None:
        if not isinstance(getattr(self, "_conversation_messages", None), list):
            object.__setattr__(self, "_conversation_messages", [])
        if not hasattr(self, "_pending_user_message"):
            object.__setattr__(self, "_pending_user_message", None)
        if not hasattr(self, "_pending_intents"):
            object.__setattr__(self, "_pending_intents", None)
        if not hasattr(self, "_pending_intent_llm"):
            object.__setattr__(self, "_pending_intent_llm", None)
        if not hasattr(self, "_streaming_conversation_turn"):
            object.__setattr__(self, "_streaming_conversation_turn", False)
        if not hasattr(self, "_turn_classified_intent"):
            object.__setattr__(self, "_turn_classified_intent", None)
        if not hasattr(self, "_assistant_reply_appended"):
            object.__setattr__(self, "_assistant_reply_appended", False)
        if not isinstance(getattr(self, "_turn_recorded_results", None), list):
            object.__setattr__(self, "_turn_recorded_results", [])
        if not hasattr(self, "_resolved_conversation_config"):
            object.__setattr__(self, "_resolved_conversation_config", None)

    def _extend_definition(self, definition: FlowDefinition) -> FlowDefinition:
        """Add the built-in conversational methods a declaration cannot inherit.

        An entry the author supplied under the same name always wins, and a
        Python flow already carries all four from the MRO walk, so this only
        fills gaps.
        """
        conversational = definition.conversational
        if conversational is None or not conversational.enabled:
            return definition

        # A declaration enables chat without the ``conversational = True``
        # class attribute, so mark the instance. Callers outside this package
        # capability-check that attribute, and it should agree with
        # ``_is_conversational_enabled()``. Instance-only on purpose: the DSL
        # projection reads it off the *class* to decide whether to emit a
        # conversational block, and setting it there would make every later
        # subclass look conversational.
        object.__setattr__(self, "conversational", True)

        missing = {
            name: method
            for name, method in _builtin_methods().items()
            if name not in definition.methods
        }
        if not missing:
            return definition
        return definition.model_copy(
            update={"methods": {**definition.methods, **missing}}
        )

    def _compose_extension_state_model(
        self, model_class: type[BaseModel]
    ) -> type[BaseModel]:
        """Add the conversational fields to a declared state model.

        A turn reads ``messages``, ``current_user_message``, ``last_intent``,
        ``ended``, ``events`` and ``agent_threads`` off state, so a declared
        model that lacks them fails on the first turn. Composing rather than
        replacing keeps every field the declaration asked for. A model that
        already extends ``ConversationState`` is returned untouched.
        """
        if not self._is_conversational_enabled():
            return model_class
        if issubclass(model_class, ConversationState):
            return model_class

        class ConversationalState(ConversationState, model_class):  # type: ignore[misc, valid-type]
            pass

        return ConversationalState

    def _create_default_extension_state(
        self, *, ignore_declared_state: bool = False
    ) -> ConversationState | None:
        """Supply ``ConversationState`` when the declaration cannot carry it.

        A declared ``pydantic`` or ``json_schema`` state is composed with the
        conversational fields instead (see
        ``_compose_extension_state_model``), so it keeps everything it asked
        for. ``dict`` and ``unknown`` state cannot carry them at all, so this
        supplies the real shape rather than letting the turn die on
        ``state.id`` -- seeded from the declared defaults where they fit.
        """
        if not self._is_conversational_enabled():
            return None

        state_definition = self._conversation_flow_definition().state
        if state_definition is None:
            return ConversationState()
        if not ignore_declared_state and state_definition.type in (
            "pydantic",
            "json_schema",
        ):
            # Composed with the declared model instead; see
            # ``_compose_extension_state_model``.
            return None
        return _conversation_state_with_defaults(state_definition.default)

    def _should_apply_pending_kickoff_context(self) -> bool:
        return (
            self._is_conversational_enabled() and self._pending_user_message is not None
        )

    def _apply_pending_kickoff_context(self) -> None:
        self._apply_pending_conversational_turn()

    def _order_start_methods_for_kickoff(
        self,
        start_methods: list[Any],
    ) -> tuple[list[Any], bool]:
        if not self._is_conversational_enabled():
            return start_methods, False

        route_conversation = "route_conversation"
        if route_conversation not in {str(method) for method in start_methods}:
            return start_methods, False

        ordered_starts = [
            method for method in start_methods if str(method) != route_conversation
        ]
        ordered_starts.append(
            next(
                method for method in start_methods if str(method) == route_conversation
            )
        )
        return ordered_starts, True

    def _should_defer_trace_finalization(self) -> bool:
        """Whether per-turn ``FlowFinished`` + ``finalize_batch`` should be skipped.

        True when either:
          - ``flow.defer_trace_finalization`` is set on the instance, OR
          - the resolved conversational configuration enables it.

        Either source enables the deferred-session pattern. The caller
        eventually invokes ``finalize_session_traces()`` to close the batch.

        Read through ``_conversation_config`` rather than the definition so
        deferral follows the same precedence as every other behavior knob. A
        class config and a declaration can disagree on the hybrid
        ``ClassWithConfig.from_declaration(...)`` path, and deferral must not
        be the one setting that follows the other source.
        """
        if getattr(self, "defer_trace_finalization", False):
            return True
        if not self._is_conversational_enabled():
            return False
        config = self._conversation_config
        return bool(config and config.defer_trace_finalization)

    def _reset_turn_execution_state(self) -> None:
        """Clear per-execution tracking so the next turn re-runs the graph."""
        self._completed_methods.clear()
        self._method_outputs.clear()
        self._pending_events.clear()
        self._method_call_counts.clear()
        self._clear_or_listeners()
        self._is_execution_resuming = False
        object.__setattr__(self, "_turn_classified_intent", None)

    def _apply_pending_conversational_turn(self) -> None:
        """Drain the stashed user message + classify if intents configured.

        Called from ``Flow.kickoff_async`` AFTER persist state restore so
        the appended message survives ``self.persistence.load_state(...)``.
        """
        object.__setattr__(self, "_turn_classified_intent", None)
        if self._pending_user_message is None:
            return

        text = self._coerce_user_message_text(self._pending_user_message)
        if not text.strip():
            return

        cfg = self._conversation_config
        outcomes = self._pending_intents
        if outcomes is None and cfg is not None:
            outcomes = cfg.default_intents
        llm = self._pending_intent_llm
        if llm is None and cfg is not None:
            llm = cfg.intent_llm

        if outcomes:
            if llm is None:
                raise ValueError("intent_llm is required when intents are provided")
            self.receive_user_message(text, outcomes=outcomes, llm=llm)
        else:
            self.receive_user_message(text)

    def _resolve_system_prompt(self) -> str | None:
        """Return the effective conversational system prompt."""
        from crewai.utilities.i18n import I18N_DEFAULT

        config = self._conversation_config
        if config is None or config.system_prompt is None:
            return I18N_DEFAULT.slice("conversational_system_prompt")
        return config.system_prompt or None

    def _resolve_answer_from_history_prompt(self) -> str:
        """Return the deprecated history-only route's effective prompt."""
        from crewai.utilities.i18n import I18N_DEFAULT

        config = self._conversation_config
        if config is None or not config.answer_from_history_prompt:
            return I18N_DEFAULT.slice("conversational_answer_from_history_prompt")
        return config.answer_from_history_prompt

    def _route_with_config(
        self,
        router_config: RouterConfig,
        context: dict[str, Any],
    ) -> str | None:
        router_llm = self._default_router_llm(router_config)
        if router_llm is None:
            return router_config.default_intent

        try:
            llm = self._coerce_llm(router_llm)
            response = self._call_router_llm(
                llm,
                messages=self._build_router_messages(router_config, context),
                response_format=self._router_response_format(router_config),
            )
            intent = self._extract_router_intent(response, router_config.intent_field)
        except Exception:
            return router_config.fallback_intent or router_config.default_intent

        if intent is None:
            return router_config.fallback_intent or router_config.default_intent

        valid_labels = self._effective_routes(router_config)
        if valid_labels and intent not in valid_labels:
            return router_config.fallback_intent or router_config.default_intent

        return intent

    def _default_router_llm(self, router_config: RouterConfig) -> Any | None:
        config = self._conversation_config
        return (
            router_config.llm
            or (config.intent_llm if config else None)
            or (config.llm if config else None)
        )

    def _router_response_format(
        self,
        router_config: RouterConfig,
    ) -> type[BaseModel]:
        if router_config.response_format is not None:
            return router_config.response_format

        routes = sorted(self._effective_routes(router_config))
        field_definitions: dict[str, Any] = {
            router_config.intent_field: (
                str,
                Field(description=f"One of: {', '.join(routes)}"),
            )
        }
        return cast(
            type[BaseModel],
            create_model("ConversationRoute", **field_definitions),
        )

    def _call_router_llm(
        self,
        llm: Any,
        *,
        messages: list[LLMMessage],
        response_format: type[BaseModel],
    ) -> Any:
        try:
            return llm.call(messages=messages, response_format=response_format)
        except TypeError as exc:
            if "response_format" not in str(exc):
                raise
            return llm.call(messages=messages, response_model=response_format)

    def _build_router_messages(
        self,
        router_config: RouterConfig,
        context: dict[str, Any],
    ) -> list[LLMMessage]:
        catalog = self._build_route_catalog(router_config)
        context = {**context, "available_routes": sorted(catalog.keys())}
        domain_prompt = f"{router_config.prompt}\n\n" if router_config.prompt else ""
        routes_section = "Routes:\n" + "\n".join(
            f"- {label}: {description}" if description else f"- {label}"
            for label, description in sorted(catalog.items())
        )
        routing_prompt = (
            domain_prompt
            + routes_section
            + "\n\nChoose exactly one route from the list above. Prefer "
            "'converse' for follow-ups, summaries, and clarifications about "
            "prior turns — even if they touch on a topic the user previously "
            "invoked a custom route for. Use a custom route only when the user "
            "is making a fresh request for that tool or workflow."
        )
        return [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": json.dumps(context, default=str)},
        ]

    def _build_route_catalog(
        self,
        router_config: RouterConfig | None,
    ) -> dict[str, str]:
        label_to_method: dict[str, str] = {}
        flow_definition = self._conversation_flow_definition()
        for listener_name, method_definition in flow_definition.methods.items():
            if method_definition.listen is None or method_definition.router:
                continue
            for trigger_label in _iter_condition_labels(method_definition.listen):
                label_to_method.setdefault(trigger_label, listener_name)

        routes = self._effective_routes(router_config)
        overrides = (
            router_config.route_descriptions
            if router_config and router_config.route_descriptions
            else {}
        )

        catalog: dict[str, str] = {}
        for route_label in routes:
            if route_label in overrides:
                catalog[route_label] = overrides[route_label]
                continue
            if route_label in self.builtin_route_descriptions:
                catalog[route_label] = self.builtin_route_descriptions[route_label]
                continue
            handler_name = label_to_method.get(route_label)
            catalog[route_label] = (
                self._route_description(flow_definition, handler_name)
                if handler_name
                else ""
            )

        return catalog

    def _route_description(
        self,
        flow_definition: FlowDefinition,
        handler_name: str,
    ) -> str:
        """Describe a route for the router LLM's catalog.

        The declared ``description`` comes first so a declaration can say what
        a route is for; the handler docstring backs it for Python flows whose
        definition predates the DSL filling that field.
        """
        declared = flow_definition.methods[handler_name].description
        if declared:
            return declared.strip().split("\n", 1)[0].strip()

        method = getattr(type(self), handler_name, None)
        if method is None:
            # A declarative flow has no class attribute to read, and
            # ``None.__doc__`` is NoneType's own docstring — which would be
            # handed to the router LLM as the route description.
            return ""
        doc = getattr(method, "__doc__", None)
        return doc.strip().split("\n", 1)[0].strip() if doc else ""

    def _extract_router_intent(self, response: Any, intent_field: str) -> str | None:
        if isinstance(response, BaseModel):
            value = getattr(response, intent_field, None)
        elif isinstance(response, dict):
            value = response.get(intent_field)
        elif isinstance(response, str):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                value = response.strip()
            else:
                value = parsed.get(intent_field)
        else:
            value = getattr(response, intent_field, None)

        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    def _valid_route_labels(self) -> set[str]:
        labels: set[str] = set()
        flow_definition = self._conversation_flow_definition()
        for method_definition in flow_definition.methods.values():
            if method_definition.listen is None or method_definition.router:
                continue
            labels.update(_iter_condition_labels(method_definition.listen))
        return labels

    def _effective_builtin_routes(self) -> set[str]:
        """Route labels the framework handles, from the definition when there is one.

        Shared with ``route_turn`` so both agree on what counts as a custom
        route. Reading the class attribute in one place and the definition in
        the other lets a declaration that customizes ``builtin_routes`` look
        like it declared a custom route, which auto-enables the LLM router.
        """
        definition = self._conversation_definition
        if definition is not None:
            return set(definition.builtin_routes)
        return set(self.builtin_routes)

    def _effective_internal_routes(self) -> set[str]:
        definition = self._conversation_definition
        if definition is not None:
            return set(definition.internal_routes)
        return set(self.internal_routes)

    def _effective_routes(self, router_config: RouterConfig | None = None) -> set[str]:
        custom_routes = set(router_config.routes or ()) if router_config else set()
        builtin_routes = self._effective_builtin_routes()
        if not custom_routes:
            custom_routes = (
                self._valid_route_labels()
                - builtin_routes
                - self._effective_internal_routes()
            )
        return custom_routes | builtin_routes

    def _default_conversation_llm(self) -> Any | None:
        config = self._conversation_config
        if config is None:
            return None
        if config.llm is not None:
            return config.llm
        if config.answer_from_history_llm is not None:
            return config.answer_from_history_llm
        if config.router is not None:
            return config.router.llm
        return config.intent_llm

    def _resolve_visibility(
        self,
        agent_name: str,
        visibility: Literal["private", "public"],
    ) -> Literal["private", "public"]:
        if visibility == "public":
            return "public"
        config = self._conversation_config
        visible = config.visible_agent_outputs if config else None
        if visible == "all" or (visible is not None and agent_name in visible):
            return "public"
        return "private"

    def _append_public_turn_result_and_persist(self, result: Any) -> None:
        """Copy a public handler return into history, then snapshot if needed.

        Built-in routes already call ``append_assistant_message`` inside the
        method, so this is a no-op for them. Custom ``@listen`` handlers that
        only ``return`` a public string hit this path after ``kickoff()``,
        which is after the per-method ``@persist`` snapshot. Persist again so
        a fresh Flow instance restores the assistant reply.
        """
        if (
            result is None
            or self._assistant_reply_appended
            or not self._is_public_turn_result(result)
        ):
            return
        self.append_assistant_message(self._stringify_result(result))
        self._persist_fallback_assistant_reply()

    def _persist_fallback_assistant_reply(self) -> None:
        """Write the current state using the same persist path as kickoff."""
        last = self._method_outputs[-1] if self._method_outputs else None
        method_name = last.get("method") if isinstance(last, dict) else None
        if not isinstance(method_name, str) or not method_name:
            return
        self._persist_method_completion(FlowMethodName(method_name))

    def _is_public_turn_result(self, result: Any) -> bool:
        """Whether a handler's return value should become the assistant reply.

        Declarative ``agent`` and ``crew`` actions return ``LiteAgentOutput`` /
        ``CrewOutput`` rather than a string, so the text lives on ``.raw``.
        Without unwrapping it here their reply never reaches the transcript.
        """
        if any(result is recorded for recorded in self._turn_recorded_results):
            # Already routed by ``append_agent_result``, which honours
            # ``visible_agent_outputs``. Promoting it here would publish a
            # result the handler asked to keep private.
            return False

        text = result
        raw = getattr(result, "raw", None)
        if not isinstance(text, str) and isinstance(raw, str):
            text = raw
        if not isinstance(text, str):
            return False
        if text in self._routing_artefact_labels():
            return False
        return text != cast(ConversationState, self.state).last_intent

    def _routing_artefact_labels(self) -> set[str]:
        """Strings that are a routing decision rather than something to say.

        Derived from the effective routes so a declaration that customizes
        ``builtin_routes`` is covered too; a literal list here would miss the
        added labels. ``conversation`` and ``route_to_flow`` are not routes:
        the first is a legacy label and the second is an outcome of the
        answer-from-history check.
        """
        return (
            self._effective_builtin_routes()
            | self._effective_internal_routes()
            | {"conversation", "route_to_flow"}
        )

    @staticmethod
    def _coerce_user_message_text(user_message: str | dict[str, Any] | Any) -> str:
        if isinstance(user_message, str):
            return user_message
        if isinstance(user_message, dict) and user_message.get("content") is not None:
            return str(user_message["content"])
        return str(user_message)

    @staticmethod
    def _stringify_result(result: Any) -> str:
        if hasattr(result, "raw"):
            return str(result.raw)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)

    @staticmethod
    def _format_messages(messages: Sequence[Mapping[str, Any]]) -> str:
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )

    @staticmethod
    def _coerce_llm(llm: str | BaseLLM | Any) -> Any:
        """Resolve a declared or live LLM the way a crew agent's ``llm`` resolves.

        ``create_llm`` is the shared helper the crew/agent declaration layer
        uses, so a conversational block accepts the same shapes: a model-id
        string, a config mapping (``{model, max_tokens, ...}``), an
        ``LLMDefinition``, or a live ``LLM``/``BaseLLM`` passed straight
        through.
        """
        from crewai.llms.base_llm import BaseLLM as BaseLLMClass
        from crewai.project.crew_definition import LLMDefinition
        from crewai.utilities.llm_utils import create_llm

        if isinstance(llm, BaseLLMClass) or callable(getattr(llm, "call", None)):
            return llm
        if isinstance(llm, LLMDefinition):
            llm = llm.model_dump(exclude_none=True)
        if not isinstance(llm, (str, Mapping)):
            # ``create_llm`` would take an int or a list as a model name and
            # build an LLM that only fails later, at call time, with a
            # provider error. A declaration is hand-written, so a typo here
            # should say so now.
            raise ValueError(
                f"Invalid llm {llm!r}: expected a model-id string, a config "
                "mapping with a 'model' key, or an LLM instance."
            )

        return create_llm(dict(llm) if isinstance(llm, Mapping) else llm)

    @contextmanager
    def _conversation_streaming_enabled(self, llm: Any) -> Iterator[None]:
        if not getattr(self, "_streaming_conversation_turn", False) or not hasattr(
            llm, "stream"
        ):
            yield
            return

        from crewai.llms.base_llm import call_stream_override

        with call_stream_override(llm, True):
            yield

    def finalize_session_traces(self) -> None:
        """Emit a final ``FlowFinishedEvent`` and finalize the trace batch.

        Pairs with ``flow.defer_trace_finalization = True`` (or
        ``ConversationConfig(defer_trace_finalization=True)``): per-turn
        ``handle_turn()`` skips the close, then a single call here at
        session end emits one ``FlowFinishedEvent`` + ``finalize_batch()``
        so the whole conversation lands as one trace.

        Safe to call when not deferring — it's a no-op if the trace batch
        was already finalized per-turn or never started.
        """
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.event_context import restore_event_scope
        from crewai.events.listeners.tracing.trace_listener import (
            TraceCollectionListener,
        )
        from crewai.events.types.flow_events import FlowFinishedEvent

        # Background memory saves must finish (and emit their completed/failed
        # events) before the session-end flow_finished / batch finalization
        # below tears down listeners, mirroring the non-deferred kickoff path.
        # The flush then waits for those events' async bus handlers.
        drain_memory_writes = getattr(self, "_drain_memory_writes", None)
        if callable(drain_memory_writes):
            drain_memory_writes()
        crewai_event_bus.flush()

        # Only emit the session-end event when a deferred flow_started is
        # actually pending. ``_deferred_flow_started_event_id`` is set only by
        # deferred kickoffs; when finalization was not deferred, each per-turn
        # kickoff already emitted its own flow_finished, so emitting here would
        # duplicate the session-end event and confuse tracing. Restoring the
        # stashed scope also pairs this flow_finished with its opener instead
        # of warning about an empty scope stack.
        started_id = getattr(self, "_deferred_flow_started_event_id", None)
        if started_id:
            method_outputs = self.method_outputs
            last_output = method_outputs[-1] if method_outputs else None
            restore_event_scope(((started_id, "flow_started"),))
            try:
                crewai_event_bus.emit(
                    self,
                    FlowFinishedEvent(
                        type="flow_finished",
                        flow_name=self.name or self.__class__.__name__,
                        result=last_output,
                        state=self._copy_and_serialize_state(),
                    ),
                )
            except Exception:
                logger.warning(
                    "FlowFinishedEvent emission failed during finalize_session_traces",
                    exc_info=True,
                )
            finally:
                restore_event_scope(())
                object.__setattr__(self, "_deferred_flow_started_event_id", None)

        trace_listener = TraceCollectionListener()
        batch_manager = trace_listener.batch_manager
        try:
            if batch_manager.batch_owner_type == "flow":
                if trace_listener.first_time_handler.is_first_time:
                    trace_listener.first_time_handler.mark_events_collected()
                    trace_listener.first_time_handler.handle_execution_completion()
                else:
                    batch_manager.finalize_batch()
        finally:
            batch_manager.defer_session_finalization = False


__all__ = ["_ConversationalMixin"]
