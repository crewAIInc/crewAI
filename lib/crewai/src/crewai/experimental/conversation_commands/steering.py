"""Session-scoped steering applied by experimental ``/btw`` commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crewai.experimental.conversation_commands.parser import (
    HELP_TEXT,
    BtwAction,
    BtwKind,
)
from crewai.flow.conversational import ConversationEvent, ConversationState


STEERING_ATTR = "_btw_steering"
ENABLED_ATTR = "_btw_commands_enabled"

_STEERING_HEADER = (
    "By-the-way steering (follow these unless they conflict with safety):"
)


@dataclass
class BtwSteering:
    """In-memory steering notes and optional forced route for one session."""

    notes: list[str] = field(default_factory=list)
    forced_route: str | None = None
    persist_route: bool = False

    def apply_note(self, note: str) -> str:
        text = note.strip()
        if not text:
            return self.show()
        if text not in self.notes:
            self.notes.append(text)
        return f"Noted. I'll keep this in mind: {text}"

    def apply_route(self, route: str, *, persist: bool) -> str:
        label = route.strip()
        self.forced_route = label
        self.persist_route = persist
        if persist:
            return f"I'll keep using route {label} until you /btw clear."
        return f"Next turn will use route {label}."

    def clear(self) -> str:
        self.notes.clear()
        self.forced_route = None
        self.persist_route = False
        return "Cleared steering notes and forced routes."

    def show(self) -> str:
        lines = ["Current /btw steering:"]
        if self.notes:
            lines.extend(f"- {note}" for note in self.notes)
        else:
            lines.append("- (no notes)")
        if self.forced_route:
            mode = "persist" if self.persist_route else "next turn"
            lines.append(f"- forced route: {self.forced_route} ({mode})")
        else:
            lines.append("- forced route: (none)")
        return "\n".join(lines)

    def consume_forced_route(self) -> str | None:
        """Return the forced route, clearing it when it was one-shot."""
        route = self.forced_route
        if route is None:
            return None
        if not self.persist_route:
            self.forced_route = None
        return route

    def apply_to_system_prompt(self, base: str | None) -> str | None:
        if not self.notes:
            return base
        steering = _STEERING_HEADER + "\n" + "\n".join(f"- {note}" for note in self.notes)
        if base:
            return f"{base}\n\n{steering}"
        return steering

    def apply_to_router_context(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.notes and not self.forced_route:
            return context
        enriched = dict(context)
        if self.notes:
            enriched["steering_notes"] = list(self.notes)
        if self.forced_route:
            enriched["forced_route"] = self.forced_route
        return enriched


def get_btw_steering(flow: Any) -> BtwSteering:
    """Return the per-instance steering store, creating it if needed."""
    steering = getattr(flow, STEERING_ATTR, None)
    if isinstance(steering, BtwSteering):
        return steering
    steering = BtwSteering()
    object.__setattr__(flow, STEERING_ATTR, steering)
    return steering


def apply_btw_action(flow: Any, action: BtwAction) -> str:
    """Mutate steering on ``flow`` and return the acknowledgement text."""
    steering = get_btw_steering(flow)
    if action.kind is BtwKind.HELP:
        reply = HELP_TEXT
    elif action.kind is BtwKind.CLEAR:
        reply = steering.clear()
    elif action.kind is BtwKind.SHOW:
        reply = steering.show()
    elif action.kind is BtwKind.ROUTE:
        reply = _apply_route(flow, steering, action)
    else:
        reply = steering.apply_note(action.argument)
    _record_btw_event(flow, action, reply)
    return reply


def _apply_route(flow: Any, steering: BtwSteering, action: BtwAction) -> str:
    label = action.argument.strip()
    available = _available_routes(flow)
    if available is not None and label not in available:
        listed = ", ".join(sorted(available)) or "(none)"
        return f"Unknown route {label!r}. Available: {listed}"
    return steering.apply_route(label, persist=action.persist_route)


def _available_routes(flow: Any) -> set[str] | None:
    resolver = getattr(flow, "_effective_routes", None)
    if not callable(resolver):
        return None
    try:
        routes = resolver()
    except Exception:
        return None
    if isinstance(routes, set):
        return routes
    return set(routes)


def _record_btw_event(flow: Any, action: BtwAction, reply: str) -> None:
    state = getattr(flow, "_state", None) or getattr(flow, "state", None)
    if not isinstance(state, ConversationState):
        return
    state.events.append(
        ConversationEvent(
            type="btw_command",
            payload={
                "kind": action.kind.value,
                "argument": action.argument,
                "persist_route": action.persist_route,
                "reply": reply,
            },
            visibility="private",
        )
    )
