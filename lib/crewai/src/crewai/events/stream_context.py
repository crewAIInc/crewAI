"""Scoped stream sinks for converting emitted events into public frames."""

from __future__ import annotations

from collections.abc import Callable
import contextvars
from typing import Any


StreamSink = Callable[[Any, Any], None]

_stream_sinks: contextvars.ContextVar[tuple[StreamSink, ...]] = contextvars.ContextVar(
    "crewai_stream_sinks", default=()
)


def add_stream_sink(sink: StreamSink) -> contextvars.Token[tuple[StreamSink, ...]]:
    """Register a sink in the current context."""
    return _stream_sinks.set((*_stream_sinks.get(), sink))


def reset_stream_sinks(token: contextvars.Token[tuple[StreamSink, ...]]) -> None:
    """Restore the stream sink context."""
    _stream_sinks.reset(token)


def publish_stream_event(source: Any, event: Any) -> None:
    """Publish a prepared event to sinks scoped to the current execution."""
    for sink in _stream_sinks.get():
        sink(source, event)
