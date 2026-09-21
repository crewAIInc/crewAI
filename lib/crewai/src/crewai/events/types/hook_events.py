from typing import Literal

from crewai.events.base_events import BaseEvent


class HookDispatchedEvent(BaseEvent):
    """Event emitted whenever an interception point dispatches to hooks.

    Only emitted when at least one hook is registered for the point, so the
    no-op fast path stays free of event overhead.
    """

    type: Literal["hook_dispatched"] = "hook_dispatched"
    interception_point: str
    outcome: Literal["proceeded", "modified", "aborted"]
    hook_count: int
    duration_ms: float
    abort_reason: str | None = None
    abort_source: str | None = None
