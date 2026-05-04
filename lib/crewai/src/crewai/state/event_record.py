"""Directed record of execution events.

Stores events as nodes with typed edges for parent/child, causal, and
sequential relationships.  Provides O(1) lookups and traversal.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, PrivateAttr

from crewai.events.base_events import BaseEvent
from crewai.utilities.rw_lock import RWLock


_event_type_map: dict[str, type[BaseEvent]] = {}


def _resolve_event(v: Any) -> BaseEvent:
    """Validate an event value into the correct BaseEvent subclass."""
    if isinstance(v, BaseEvent):
        return v
    if not isinstance(v, dict):
        return BaseEvent.model_validate(v)
    if not _event_type_map:
        _build_event_type_map()
    event_type = v.get("type", "")
    cls = _event_type_map.get(event_type, BaseEvent)
    if cls is BaseEvent:
        return BaseEvent.model_validate(v)
    try:
        return cls.model_validate(v)
    except Exception:
        return BaseEvent.model_validate(v)


def _build_event_type_map() -> None:
    """Populate _event_type_map from all BaseEvent subclasses."""

    def _collect(cls: type[BaseEvent]) -> None:
        subclasses: list[type[BaseEvent]] = cls.__subclasses__()
        for sub in subclasses:
            type_field = sub.model_fields.get("type")
            if type_field and type_field.default:
                _event_type_map[type_field.default] = sub
            _collect(sub)

    _collect(BaseEvent)


EdgeType = Literal[
    "parent",
    "child",
    "trigger",
    "triggered_by",
    "next",
    "previous",
    "started",
    "completed_by",
]


class EventNode(BaseModel):
    """A node wrapping a single event with its adjacency lists."""

    event: Annotated[
        BaseEvent,
        BeforeValidator(_resolve_event),
        PlainSerializer(lambda v: v.model_dump()),
    ]
    edges: dict[EdgeType, list[str]] = Field(default_factory=dict)

    def add_edge(self, edge_type: EdgeType, target_id: str) -> None:
        """Add an edge from this node to another.

        Args:
            edge_type: The relationship type.
            target_id: The event_id of the target node.
        """
        self.edges.setdefault(edge_type, []).append(target_id)

    def neighbors(self, edge_type: EdgeType) -> list[str]:
        """Return neighbor IDs for a given edge type.

        Args:
            edge_type: The relationship type to query.

        Returns:
            List of event IDs connected by this edge type.
        """
        return self.edges.get(edge_type, [])


class EventRecord(BaseModel):
    """Directed record of execution events with O(1) node lookup.

    Events are added via :meth:`add` which automatically wires edges
    based on the event's relationship fields — ``parent_event_id``,
    ``triggered_by_event_id``, ``previous_event_id``, ``started_event_id``.
    """

    nodes: dict[str, EventNode] = Field(default_factory=dict)
    _lock: RWLock = PrivateAttr(default_factory=RWLock)

    def add(self, event: BaseEvent) -> EventNode:
        """Add an event to the record and wire its edges.

        Args:
            event: The event to insert.

        Returns:
            The created node.
        """
        with self._lock.w_locked():
            node = EventNode(event=event)
            self.nodes[event.event_id] = node

            if event.parent_event_id and event.parent_event_id in self.nodes:
                node.add_edge("parent", event.parent_event_id)
                self.nodes[event.parent_event_id].add_edge("child", event.event_id)

            if (
                event.triggered_by_event_id
                and event.triggered_by_event_id in self.nodes
            ):
                node.add_edge("triggered_by", event.triggered_by_event_id)
                self.nodes[event.triggered_by_event_id].add_edge(
                    "trigger", event.event_id
                )

            if event.previous_event_id and event.previous_event_id in self.nodes:
                node.add_edge("previous", event.previous_event_id)
                self.nodes[event.previous_event_id].add_edge("next", event.event_id)

            if event.started_event_id and event.started_event_id in self.nodes:
                node.add_edge("started", event.started_event_id)
                self.nodes[event.started_event_id].add_edge(
                    "completed_by", event.event_id
                )

            return node

    def get(self, event_id: str) -> EventNode | None:
        """Look up a node by event ID.

        Args:
            event_id: The event's unique identifier.

        Returns:
            The node, or None if not found.
        """
        with self._lock.r_locked():
            return self.nodes.get(event_id)

    def descendants(self, event_id: str) -> list[EventNode]:
        """Return all descendant nodes, children recursively.

        Args:
            event_id: The root event ID to start from.

        Returns:
            All descendant nodes in breadth-first order.
        """
        with self._lock.r_locked():
            result: list[EventNode] = []
            queue = [event_id]
            visited: set[str] = set()

            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                node = self.nodes.get(current_id)
                if node is None:
                    continue

                for child_id in node.neighbors("child"):
                    if child_id not in visited:
                        child_node = self.nodes.get(child_id)
                        if child_node:
                            result.append(child_node)
                            queue.append(child_id)

            return result

    def roots(self) -> list[EventNode]:
        """Return all root nodes — events with no parent.

        Returns:
            List of root event nodes.
        """
        with self._lock.r_locked():
            return [
                node for node in self.nodes.values() if not node.neighbors("parent")
            ]

    def all_nodes(self) -> list[EventNode]:
        """Return a snapshot of every node under the read lock.

        Returns:
            A list copy of the current nodes, safe to iterate without holding
            the lock.
        """
        with self._lock.r_locked():
            return list(self.nodes.values())

    def clear(self) -> None:
        """Remove all nodes from the record under the write lock."""
        with self._lock.w_locked():
            self.nodes.clear()

    def __len__(self) -> int:
        with self._lock.r_locked():
            return len(self.nodes)

    def __contains__(self, event_id: str) -> bool:
        with self._lock.r_locked():
            return event_id in self.nodes
