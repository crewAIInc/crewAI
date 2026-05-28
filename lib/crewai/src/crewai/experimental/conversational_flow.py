"""Experimental higher-level conversational Flow abstraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model

from crewai.flow.conversation import get_conversation_messages
from crewai.flow.flow import Flow, listen, router, start
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.i18n import I18N_DEFAULT
from crewai.utilities.types import LLMMessage


# Pydantic rebuilds inherited Flow annotations in this module's namespace.
# Keep the forward reference resolvable without importing crewai.context here,
# which would create a cycle while crewai.experimental is importing.
ExecutionContext = Any

ConversationMessageRole = Literal["user", "assistant", "system", "tool"]
ConversationEventVisibility = Literal["private", "public"]


@dataclass
class RouterConfig:
    """Class-level LLM router configuration for ``ConversationalFlow``.

    ``route_descriptions`` overrides the per-route descriptions used to build
    the router LLM's "available routes" catalog. Routes without an entry fall
    back to the handler's docstring first line (or, for built-in routes, the
    framework's canned description). ``prompt`` is reserved for domain
    policy/voice, not the route catalog — that's auto-built.
    """

    prompt: str | None = None
    response_format: type[BaseModel] | None = None
    llm: Any | None = None
    routes: Sequence[str] | None = None
    route_descriptions: dict[str, str] | None = None
    default_intent: str | None = "converse"
    fallback_intent: str | None = "converse"
    intent_field: str = "intent"


@dataclass
class ConversationConfig:
    """Class-level configuration for experimental conversational flows.

    ``system_prompt`` defaults to the ``slices.conversational_system_prompt``
    translation when left as ``None``. Pass an empty string to opt out of any
    system prompt for ``converse_turn``. ``answer_from_history_prompt`` falls
    back to ``slices.conversational_answer_from_history_prompt`` when ``None``.
    """

    system_prompt: str | None = None
    llm: Any | None = None
    router: RouterConfig | None = None
    answer_from_history_prompt: str | None = None
    default_intents: Sequence[str] | None = None
    intent_llm: Any | None = None
    answer_from_history_llm: Any | None = None
    visible_agent_outputs: Sequence[str] | Literal["all"] | None = None
    defer_trace_finalization: bool = True

    def __call__(self, flow_cls: type[Any]) -> type[Any]:
        """Use this config as a class decorator."""
        flow_cls.conversational_config = self
        return flow_cls


class ConversationMessage(BaseModel):
    """Canonical user-facing message shared across conversational turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: ConversationMessageRole
    content: str | list[dict[str, Any]] | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    files: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Private per-agent message or scratch result."""

    role: ConversationMessageRole | str = "assistant"
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationEvent(BaseModel):
    """Structured trace/event that is separate from user-visible messages."""

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    visibility: ConversationEventVisibility = "private"


class ConversationState(BaseModel):
    """Structured state for ``ConversationalFlow``.

    ``messages`` is the canonical user-facing history. Agent/tool scratch work
    belongs in ``events`` or ``agent_threads`` unless explicitly made public.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ConversationMessage] = Field(default_factory=list)
    current_user_message: str | None = None
    last_user_message: str | None = None
    last_intent: str | None = None
    ended: bool = False
    events: list[ConversationEvent] = Field(default_factory=list)
    agent_threads: dict[str, list[AgentMessage]] = Field(default_factory=dict)
    session_ready: bool = False


def _message_to_llm_dict(message: Any) -> LLMMessage:
    if isinstance(message, BaseModel):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {"role": "user", "content": str(message)}

    return cast(
        LLMMessage,
        {key: value for key, value in data.items() if key != "metadata"},
    )


class ConversationalFlow(Flow[ConversationState]):
    """Flow base class for turn-oriented conversational applications.

    Subclasses define normal ``@listen("intent")`` handlers. The inherited
    start/router methods turn each ``handle_turn()`` call into one Flow run.
    """

    conversational_config: ClassVar[ConversationConfig | None] = None
    builtin_routes: ClassVar[tuple[str, ...]] = ("converse", "end")
    internal_routes: ClassVar[tuple[str, ...]] = (
        "answer_from_history",
        "conversation_start",
    )
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

    @start()
    def conversation_start(self) -> str | None:
        """Internal Flow entrypoint for a single chat turn."""
        return self.state.current_user_message

    @router(conversation_start)
    def route_conversation(self) -> str:
        """Route the current turn to a listener label."""
        context = self.build_router_context()
        configured_route = self.route_turn(context)
        if configured_route:
            self.state.last_intent = configured_route
            return configured_route

        if self.state.last_intent:
            return self.state.last_intent

        if self.can_answer_from_history(context):
            self.state.last_intent = "answer_from_history"
            return "answer_from_history"

        self.state.last_intent = "converse"
        return "converse"

    @listen("converse")
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
    def end_conversation(self) -> str:
        """Built-in conversation terminator."""
        self.state.ended = True
        content = "Conversation ended."
        self.append_assistant_message(content)
        return content

    @listen("answer_from_history")
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

    def kickoff(self, *args: Any, **kwargs: Any) -> Any:
        """Run one conversational turn.

        Every call into ``ConversationalFlow`` is a turn, so reset the graph
        execution tracking before delegating. Without this, calls after the
        first see ``_completed_methods`` populated, ``Flow.kickoff_async``
        flips ``_is_execution_resuming = True``, every method short-circuits,
        and the prior turn's output is returned. The persisted Pydantic state
        (messages, current_user_message, etc.) is preserved on ``self._state``.

        Checkpoint restores deliberately keep ``_completed_methods`` populated
        so paused work resumes; skip the reset in that path.
        """
        is_checkpoint_restore = kwargs.get("from_checkpoint") is not None or (
            len(args) >= 3 and args[2] is not None
        )
        if not is_checkpoint_restore:
            self._reset_turn_execution_state()
        return super().kickoff(*args, **kwargs)

    def handle_turn(
        self,
        message: str,
        *,
        session_id: str | None = None,
        intents: Sequence[str] | None = None,
        intent_llm: str | BaseLLM | None = None,
        **kickoff_kwargs: Any,
    ) -> Any:
        """Append a user message, run one conversational turn, and return output."""
        assistant_count = self._assistant_message_count()
        result = self.kickoff(
            user_message=message,
            session_id=session_id or self.state.id,
            intents=intents,
            intent_llm=intent_llm,
            **kickoff_kwargs,
        )
        if (
            result is not None
            and self._assistant_message_count() == assistant_count
            and self._is_public_turn_result(result)
        ):
            self.append_assistant_message(self._stringify_result(result))
        return result

    def build_router_context(self) -> dict[str, Any]:
        """Build context used by the routing policy for the current turn."""
        return {
            "system_prompt": self._resolve_system_prompt(),
            "current_user_message": self.state.current_user_message,
            "message_history": self.conversation_messages,
            "events": [event.model_dump() for event in self.state.events],
            "last_intent": self.state.last_intent,
        }

    def build_agent_context(self, agent_name: str) -> list[LLMMessage]:
        """Build canonical message context for an agent or direct LLM call."""
        messages = list(self.conversation_messages)
        thread = self.state.agent_threads.get(agent_name, [])
        messages.extend(
            cast(
                LLMMessage,
                {"role": msg.role, "content": self._stringify_result(msg.content)},
            )
            for msg in thread
        )
        return messages

    def route_turn(self, context: dict[str, Any]) -> str | None:
        """Route with ``ConversationConfig.router`` when configured."""
        config = self._conversation_config
        if config is None or config.router is None:
            return None
        return self._route_with_config(config.router, context)

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
        self.state.events.append(event)
        self.state.agent_threads.setdefault(agent_name, []).append(
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
        self.state.messages.append(
            ConversationMessage(
                role="assistant",
                content=content,
                metadata=metadata or {},
            )
        )

    @property
    def conversation_messages(self) -> list[LLMMessage]:
        """Canonical user-facing message history as LLM-compatible dicts."""
        return [
            _message_to_llm_dict(message) for message in get_conversation_messages(self)
        ]

    @property
    def _conversation_config(self) -> ConversationConfig | None:
        return getattr(type(self), "conversational_config", None)

    def _reset_turn_execution_state(self) -> None:
        """Clear per-execution tracking so the next turn re-runs the graph.

        Mirrors what ``Flow.kickoff_async`` does on a non-restoring run: drops
        completed-method tracking, per-method call counts, and pending listener
        bookkeeping. ``self._state`` (messages, current_user_message, etc.) is
        deliberately untouched so the conversation continues uninterrupted.
        """
        self._completed_methods.clear()
        self._method_outputs.clear()
        self._pending_and_listeners.clear()
        self._method_call_counts.clear()
        self._clear_or_listeners()
        self._is_execution_resuming = False

    def _resolve_system_prompt(self) -> str | None:
        """Return the effective conversational system prompt.

        ``None`` on the config (the default) resolves to the i18n base prompt;
        an empty string is treated as an explicit opt-out.
        """
        config = self._conversation_config
        if config is None or config.system_prompt is None:
            return I18N_DEFAULT.slice("conversational_system_prompt")
        return config.system_prompt or None

    def _resolve_answer_from_history_prompt(self) -> str:
        """Return the effective ``answer_from_history`` prompt.

        ``None`` (the default) falls back to the i18n slice. Unlike
        ``system_prompt``, this prompt is always needed when the route runs,
        so it does not support an empty-string opt-out.
        """
        config = self._conversation_config
        if config is None or not config.answer_from_history_prompt:
            return I18N_DEFAULT.slice("conversational_answer_from_history_prompt")
        return config.answer_from_history_prompt

    def receive_user_message(
        self,
        text: str,
        *,
        outcomes: Sequence[str] | None = None,
        llm: str | BaseLLM | None = None,
    ) -> str:
        """Append a user turn and optionally classify its intent.

        ``last_intent`` is preserved across turns so the router prompt can use
        the prior turn's route as a signal (e.g., follow-up after RESEARCH
        should usually route to ``converse``). The legacy intent-classification
        path below still overwrites it when outcomes are provided, and
        ``route_conversation`` reassigns it on every router decision.
        """
        self.state.messages.append(ConversationMessage(role="user", content=text))
        self.state.current_user_message = text
        self.state.last_user_message = text

        if outcomes and llm is not None:
            intent = self.classify_intent(
                text,
                outcomes,
                llm=llm,
                context=self.conversation_messages,
            )
            self.state.last_intent = intent
            return intent

        return text

    def _apply_pending_conversational_turn(self) -> None:
        if self._pending_user_message is None:
            return

        text = self._coerce_user_message_text(self._pending_user_message)
        if not text.strip():
            return

        config = self._conversation_config
        outcomes = self._pending_intents
        if outcomes is None and config is not None:
            outcomes = config.default_intents

        llm = self._pending_intent_llm
        if llm is None and config is not None:
            llm = config.intent_llm

        if outcomes:
            if llm is None:
                raise ValueError("intent_llm is required when intents are provided")
            self.receive_user_message(text, outcomes=outcomes, llm=llm)
        else:
            self.receive_user_message(text)

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
            create_model(
                "ConversationRoute",
                **field_definitions,
            ),
        )

    def _call_router_llm(
        self,
        llm: Any,
        *,
        messages: list[LLMMessage],
        response_format: type[BaseModel],
    ) -> Any:
        """Call the router LLM with CrewAI's response_format naming.

        Older local LLM implementations may still expose ``response_model``;
        keep the compatibility fallback isolated from the public config shape.
        """
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
        context = {
            **context,
            "available_routes": sorted(catalog.keys()),
        }
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
            {
                "role": "user",
                "content": json.dumps(context, default=str),
            },
        ]

    def _build_route_catalog(
        self,
        router_config: RouterConfig | None,
    ) -> dict[str, str]:
        """Build a ``{label: description}`` catalog for the router prompt.

        Priority per route:
          1. ``router_config.route_descriptions`` override (user-provided).
          2. ``builtin_route_descriptions`` (framework-canned for converse/end/
             answer_from_history — phrased for LLM routing).
          3. First non-empty line of the ``@listen`` handler's docstring.
          4. Empty (route appears in the catalog without a description).
        """
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
        return result != self.state.last_intent

    def _assistant_message_count(self) -> int:
        return sum(1 for message in self.state.messages if message.role == "assistant")

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


__all__ = [
    "AgentMessage",
    "ConversationConfig",
    "ConversationEvent",
    "ConversationMessage",
    "ConversationState",
    "ConversationalFlow",
    "RouterConfig",
]
