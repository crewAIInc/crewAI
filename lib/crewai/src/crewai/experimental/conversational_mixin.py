"""Conversational graph + helpers as a mixin for ``Flow`` (experimental).

The experimental conversational chat surface lives here as a mixin so that
``crewai.flow.runtime`` stays focused on the execution engine. ``Flow``
inherits from ``_ConversationalMixin``; the methods only register on
subclasses that opt in via ``conversational = True`` (enforced by the
``_conversational_only`` marker + ``FlowMeta`` gating in
``crewai.flow.runtime``).

Import surface:
    - :class:`_ConversationalMixin` — internal; ``Flow`` mixes it in. Users
      don't import it directly.
    - The data types this mixin uses live in
      :mod:`crewai.experimental.conversational`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import BaseModel, Field, create_model

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
from crewai.flow.conversation import (
    append_message as _append_conversation_message,
    get_conversation_messages,
    receive_user_message as _receive_user_message,
)
from crewai.flow.dsl import listen, router, start
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow
    from crewai.llms.base_llm import BaseLLM


logger = logging.getLogger(__name__)


class _ConversationalMixin:
    """Built-in conversational graph for ``Flow`` (gated on ``conversational``).

    Mixed into ``Flow`` so its execution engine (``runtime.py``) stays focused
    on running graphs. The methods here only register on subclasses that set
    ``conversational = True``; non-chat flows see them as inert attributes.
    """

    # The metaclass + state attributes referenced below live on ``Flow`` —
    # this mixin is never instantiated standalone. These type-only
    # declarations exist so static analyzers don't flag attribute access.
    # Class-level slots use ``ClassVar`` to match Flow's actual declarations
    # (otherwise mypy flags "Cannot override instance variable with class
    # variable" when Flow declares them as ``ClassVar``).
    if TYPE_CHECKING:
        conversational: ClassVar[bool]
        conversational_config: ClassVar[ConversationConfig | None]
        builtin_routes: ClassVar[tuple[str, ...]]
        internal_routes: ClassVar[tuple[str, ...]]
        builtin_route_descriptions: ClassVar[dict[str, str]]
        # Registry ClassVars populated by ``FlowMeta`` at class creation.
        _listeners: ClassVar[dict[Any, Any]]

        # Instance attrs from ``Flow``.
        state: Any
        name: str | None
        _completed_methods: set[Any]
        _method_outputs: list[Any]
        _pending_and_listeners: dict[Any, Any]
        _method_call_counts: dict[Any, int]
        _is_execution_resuming: bool
        _pending_user_message: str | dict[str, Any] | None
        _pending_intents: Sequence[str] | None
        _pending_intent_llm: str | BaseLLM | None

        def _clear_or_listeners(self) -> None:
            pass

        def _collapse_to_outcome(
            self,
            feedback: str,
            outcomes: tuple[str, ...],
            llm: str | BaseLLM | Any,
        ) -> str:
            pass

        def _copy_and_serialize_state(self) -> dict[str, Any]:
            pass

        def kickoff(self, *args: Any, **kwargs: Any) -> Any:
            pass

    @start()
    @_conversational_only
    def conversation_start(self) -> str | None:
        """Internal Flow entrypoint that hands the user message to the router.

        In conversational mode, ``Flow.kickoff_async`` runs all ``@start``
        methods sequentially and this one is registered last, so any user
        ``@start`` methods (e.g. permission loading) have already finished
        before the returned value triggers ``route_conversation``.
        """
        state = cast(ConversationState, self.state)
        return state.current_user_message

    @router(conversation_start)
    @_conversational_only
    def route_conversation(self) -> str:
        """Route the current turn to a listener label."""
        state = cast(ConversationState, self.state)
        context = self.build_router_context()
        configured_route = self.route_turn(context)
        if configured_route:
            state.last_intent = configured_route
            return configured_route

        if state.last_intent:
            return state.last_intent

        if self.can_answer_from_history(context):
            state.last_intent = "answer_from_history"
            return "answer_from_history"

        state.last_intent = "converse"
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

        response = self._coerce_llm(llm).call(messages=messages)
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
        """Answer directly from canonical conversation history when configured."""
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

        Available only when ``conversational = True`` is set on the subclass.
        Stashes the message + session_id as pending turn state, runs kickoff
        (which restores from persist and then applies the pending turn), and
        promotes the result to an assistant message when the handler didn't.
        """
        state = cast(ConversationState, self.state)
        sid = session_id or state.id

        # Stash the pending turn so ``_apply_pending_conversational_turn``
        # picks it up AFTER persist restore.
        self._pending_user_message = message
        self._pending_intents = list(intents) if intents else None
        self._pending_intent_llm = intent_llm

        # Each turn is a fresh execution; clear graph tracking so the second
        # turn re-runs instead of being treated as a checkpoint restore.
        if "from_checkpoint" not in kickoff_kwargs:
            self._reset_turn_execution_state()

        assistant_count = self._assistant_message_count()
        try:
            result = self.kickoff(inputs={"id": sid}, **kickoff_kwargs)
        finally:
            self._pending_user_message = None
            self._pending_intents = None
            self._pending_intent_llm = None

        if (
            result is not None
            and self._assistant_message_count() == assistant_count
            and self._is_public_turn_result(result)
        ):
            self.append_assistant_message(self._stringify_result(result))
        return result

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
        if not getattr(type(self), "conversational", False):
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
                output_fn(f"{assistant_prefix}{result}")
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
        """
        config = self._conversation_config
        if config is None:
            return None

        router_config = config.router
        if router_config is None:
            if config.default_intents:
                return None
            custom_routes = self._effective_routes(None) - set(self.builtin_routes)
            if not custom_routes:
                return None
            router_config = RouterConfig()

        return self._route_with_config(router_config, context)

    def can_answer_from_history(self, context: dict[str, Any]) -> bool:
        """Return whether this turn can be answered from message history."""
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
            config.answer_from_history_llm,
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
        if event_visibility == "public":
            self.append_assistant_message(content)

    def append_assistant_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a final user-visible assistant message."""
        cast(ConversationState, self.state).messages.append(
            ConversationMessage(
                role="assistant",
                content=content,
                metadata=metadata or {},
            )
        )

    def append_message(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        **extra: Any,
    ) -> None:
        """Append a message to conversation history (legacy ChatState path)."""
        _append_conversation_message(cast("Flow[Any]", self), role, content, **extra)

    @property
    def conversation_messages(self) -> list[LLMMessage]:
        """Message history from state, coerced to LLM-shaped dicts."""
        return [
            message_to_llm_dict(message)
            for message in get_conversation_messages(cast("Flow[Any]", self))
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
        if self.conversational:
            state = cast(ConversationState, self.state)
            state.messages.append(ConversationMessage(role="user", content=text))
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
                return intent
            return text

        return _receive_user_message(
            cast("Flow[Any]", self), text, outcomes=outcomes, llm=llm
        )

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
        return self._collapse_to_outcome(feedback, tuple(outcomes), llm)

    @property
    def _conversation_config(self) -> ConversationConfig | None:
        return getattr(type(self), "conversational_config", None)

    def _should_defer_trace_finalization(self) -> bool:
        """Whether per-turn ``FlowFinished`` + ``finalize_batch`` should be skipped.

        True when either:
          - ``flow.defer_trace_finalization`` is set on the instance, OR
          - the class-level ``ConversationConfig.defer_trace_finalization``
            on a conversational subclass is True.

        Either source enables the deferred-session pattern. The caller
        eventually invokes ``finalize_session_traces()`` to close the batch.
        """
        if getattr(self, "defer_trace_finalization", False):
            return True
        config = self._conversation_config
        return bool(config and config.defer_trace_finalization)

    def _reset_turn_execution_state(self) -> None:
        """Clear per-execution tracking so the next turn re-runs the graph."""
        self._completed_methods.clear()
        self._method_outputs.clear()
        self._pending_and_listeners.clear()
        self._method_call_counts.clear()
        self._clear_or_listeners()
        self._is_execution_resuming = False

    def _apply_pending_conversational_turn(self) -> None:
        """Drain the stashed user message + classify if intents configured.

        Called from ``Flow.kickoff_async`` AFTER persist state restore so
        the appended message survives ``self.persistence.load_state(...)``.
        """
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
        """Return the effective ``answer_from_history`` prompt."""
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
        for listener_name, condition in self._listeners.items():
            if isinstance(condition, tuple):
                _, trigger_labels = condition
                for trigger_label in trigger_labels:
                    label_to_method.setdefault(str(trigger_label), str(listener_name))

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
            description = ""
            if handler_name:
                method = getattr(type(self), handler_name, None)
                doc = getattr(method, "__doc__", None)
                if doc:
                    description = doc.strip().split("\n", 1)[0].strip()
            catalog[route_label] = description

        return catalog

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
        for condition in self._listeners.values():
            if isinstance(condition, tuple):
                _, methods = condition
                labels.update(str(method) for method in methods)
        return labels

    def _effective_routes(self, router_config: RouterConfig | None = None) -> set[str]:
        custom_routes = set(router_config.routes or ()) if router_config else set()
        if not custom_routes:
            custom_routes = (
                self._valid_route_labels()
                - set(self.builtin_routes)
                - set(self.internal_routes)
            )
        return custom_routes | set(self.builtin_routes)

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

    def _assistant_message_count(self) -> int:
        state = cast(ConversationState, self.state)
        return sum(1 for message in state.messages if message.role == "assistant")

    def _is_public_turn_result(self, result: Any) -> bool:
        if not isinstance(result, str):
            return False
        if result in {
            "conversation",
            "converse",
            "end",
            "answer_from_history",
            "route_to_flow",
        }:
            return False
        return result != cast(ConversationState, self.state).last_intent

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
        from crewai.llm import LLM
        from crewai.llms.base_llm import BaseLLM as BaseLLMClass

        if isinstance(llm, str):
            return LLM(model=llm)
        if isinstance(llm, BaseLLMClass) or callable(getattr(llm, "call", None)):
            return llm
        raise ValueError(f"Invalid llm type: {type(llm)}. Expected str or BaseLLM.")

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

        # Only emit the session-end event when a deferred flow_started is
        # actually pending. ``_deferred_flow_started_event_id`` is set only by
        # deferred kickoffs; when finalization was not deferred, each per-turn
        # kickoff already emitted its own flow_finished, so emitting here would
        # duplicate the session-end event and confuse tracing. Restoring the
        # stashed scope also pairs this flow_finished with its opener instead
        # of warning about an empty scope stack.
        started_id = getattr(self, "_deferred_flow_started_event_id", None)
        if started_id:
            last_output = self._method_outputs[-1] if self._method_outputs else None
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
        if batch_manager.batch_owner_type == "flow":
            if trace_listener.first_time_handler.is_first_time:
                trace_listener.first_time_handler.mark_events_collected()
                trace_listener.first_time_handler.handle_execution_completion()
            else:
                batch_manager.finalize_batch()


__all__ = ["_ConversationalMixin"]
