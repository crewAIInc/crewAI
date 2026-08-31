"""Opt-in installer for experimental ``/btw`` conversational commands."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any, TypeVar, cast, overload

from crewai.experimental.conversation_commands.parser import parse_btw_line
from crewai.experimental.conversation_commands.steering import (
    ENABLED_ATTR,
    apply_btw_action,
    get_btw_steering,
)
from crewai.types.streaming import StreamFrame, StreamSession


T = TypeVar("T")


class _Intercept:
    __slots__ = ("consumed", "reply", "user_message")

    def __init__(
        self,
        *,
        consumed: bool,
        user_message: str | None,
        reply: str | None = None,
    ) -> None:
        self.consumed = consumed
        self.user_message = user_message
        self.reply = reply


def _intercept(flow: Any, message: str) -> _Intercept:
    parsed = parse_btw_line(message)
    if parsed.action is None:
        return _Intercept(consumed=False, user_message=parsed.user_message)

    reply = apply_btw_action(flow, parsed.action)
    if parsed.user_message:
        return _Intercept(
            consumed=False,
            user_message=parsed.user_message,
            reply=reply,
        )
    return _Intercept(consumed=True, user_message=None, reply=reply)


def _ack_stream(reply: str) -> StreamSession[str]:
    def frames() -> Iterator[StreamFrame]:
        return iter(())

    session: StreamSession[str] = StreamSession(sync_iterator=frames())
    session._set_result(reply)
    return session


def _already_enabled(target: Any) -> bool:
    if getattr(target, ENABLED_ATTR, False):
        return True
    if not isinstance(target, type) and getattr(type(target), ENABLED_ATTR, False):
        return True
    return False


def _wrap_handle_turn(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def handle_turn(self: Any, message: str, *args: Any, **kwargs: Any) -> Any:
        decision = _intercept(self, message)
        if decision.consumed:
            return decision.reply
        return original(self, decision.user_message, *args, **kwargs)

    return handle_turn


def _wrap_stream_turn(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def stream_turn(self: Any, message: str, *args: Any, **kwargs: Any) -> Any:
        decision = _intercept(self, message)
        if decision.consumed:
            return _ack_stream(decision.reply or "")
        return original(self, decision.user_message, *args, **kwargs)

    return stream_turn


def _wrap_route_turn(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def route_turn(self: Any, context: dict[str, Any]) -> str | None:
        forced = get_btw_steering(self).consume_forced_route()
        if forced:
            return forced
        return original(self, context)

    return route_turn


def _wrap_build_router_context(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def build_router_context(self: Any) -> dict[str, Any]:
        return get_btw_steering(self).apply_to_router_context(original(self))

    return build_router_context


def _wrap_resolve_system_prompt(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def _resolve_system_prompt(self: Any) -> str | None:
        return get_btw_steering(self).apply_to_system_prompt(original(self))

    return _resolve_system_prompt


def _wrap_class(cls: type[T]) -> type[T]:
    cls.handle_turn = _wrap_handle_turn(cls.handle_turn)  # type: ignore[attr-defined]
    cls.stream_turn = _wrap_stream_turn(cls.stream_turn)  # type: ignore[attr-defined]
    cls.route_turn = _wrap_route_turn(cls.route_turn)  # type: ignore[attr-defined]
    cls.build_router_context = _wrap_build_router_context(  # type: ignore[attr-defined]
        cls.build_router_context
    )
    cls._resolve_system_prompt = _wrap_resolve_system_prompt(  # type: ignore[attr-defined]
        cls._resolve_system_prompt
    )
    setattr(cls, ENABLED_ATTR, True)
    return cls


def _bind_and_wrap(
    flow: Any,
    name: str,
    wrapper: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    original = getattr(type(flow), name)
    wrapped = wrapper(original)

    @wraps(original)
    def bound(*args: Any, **kwargs: Any) -> Any:
        return wrapped(flow, *args, **kwargs)

    object.__setattr__(flow, name, bound)


def _wrap_instance(flow: T) -> T:
    target = cast(Any, flow)
    _bind_and_wrap(target, "handle_turn", _wrap_handle_turn)
    _bind_and_wrap(target, "stream_turn", _wrap_stream_turn)
    _bind_and_wrap(target, "route_turn", _wrap_route_turn)
    _bind_and_wrap(target, "build_router_context", _wrap_build_router_context)
    _bind_and_wrap(target, "_resolve_system_prompt", _wrap_resolve_system_prompt)
    object.__setattr__(target, ENABLED_ATTR, True)
    return flow


def enable_btw_commands(target: T) -> T:
    """Install ``/btw`` interjections on a Flow class or instance.

    Opt-in only: conversational flows ignore slash lines until this is
    applied. Safe to call more than once.
    """
    if _already_enabled(target):
        return target
    if isinstance(target, type):
        return _wrap_class(cast(type[T], target))
    return _wrap_instance(target)


@overload
def btw_commands(flow_cls: type[T]) -> type[T]: ...


@overload
def btw_commands(flow_cls: None = None) -> Callable[[type[T]], type[T]]: ...


def btw_commands(
    flow_cls: type[T] | None = None,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Class decorator that enables experimental ``/btw`` commands.

    Use as ``@btw_commands`` or ``@btw_commands()`` above
    ``@ConversationConfig(...)``.
    """

    def decorate(cls: type[T]) -> type[T]:
        return enable_btw_commands(cls)

    if flow_cls is not None:
        return decorate(flow_cls)
    return decorate
