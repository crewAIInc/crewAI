"""Deterministic comparison of two persisted flow states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, Literal


ChangeKind = Literal["added", "removed", "changed"]

_PREVIEW_LIMIT = 80


def _canonical(value: Any) -> str:
    """Serialize a value the way the store does, so ``1``, ``1.0`` and ``true`` stay distinct."""
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _preview(value: Any) -> str:
    """Compact JSON for a value, cut to one line's worth."""
    text = _canonical(value)
    if len(text) > _PREVIEW_LIMIT:
        return text[: _PREVIEW_LIMIT - 3] + "..."
    return text


@dataclass(frozen=True)
class FlowStateChange:
    """One difference between an earlier and a later persisted state.

    Attributes:
        path: Keys from the state root down to the field, outermost first.
        kind: ``"added"`` when the key exists only in the later state,
            ``"removed"`` when it exists only in the earlier state, and
            ``"changed"`` when both states hold different values.
        before: The earlier value. Meaningless for an ``"added"`` key.
        after: The later value. Meaningless for a ``"removed"`` key.
    """

    path: tuple[str, ...]
    kind: ChangeKind
    before: Any = None
    after: Any = None

    @property
    def dotted_path(self) -> str:
        """The path joined with dots, for text output."""
        return ".".join(self.path)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form of the change.

        ``before`` and ``after`` are present only for the side on which the key
        existed, so a key holding ``null`` stays distinct from an absent key.
        """
        result: dict[str, Any] = {"path": list(self.path), "kind": self.kind}
        if self.kind != "added":
            result["before"] = self.before
        if self.kind != "removed":
            result["after"] = self.after
        return result


@dataclass(frozen=True)
class FlowStateDiff:
    """The differences between two persisted states, ordered by path."""

    changes: tuple[FlowStateChange, ...]

    @property
    def identical(self) -> bool:
        """Whether the two states hold the same data."""
        return not self.changes

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form of the comparison."""
        return {
            "identical": self.identical,
            "changes": [change.to_dict() for change in self.changes],
        }

    def render_text(self) -> str:
        """One line per change: ``+`` added, ``-`` removed, ``~`` changed."""
        if not self.changes:
            return "No differences."
        lines: list[str] = []
        for change in self.changes:
            path = change.dotted_path or "(root)"
            if change.kind == "added":
                lines.append(f"+ {path}: {_preview(change.after)}")
            elif change.kind == "removed":
                lines.append(f"- {path}: {_preview(change.before)}")
            else:
                lines.append(
                    f"~ {path}: {_preview(change.before)} -> {_preview(change.after)}"
                )
        return "\n".join(lines)


def compare_flow_states(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> FlowStateDiff:
    """Compare two persisted states of the same flow.

    Nested dictionaries are compared key by key. A list, like any other
    non-dict value, counts as one value: a changed element reports the whole
    list. A key whose value is ``None`` is present, so it differs from a key
    that is absent. Values compare by their JSON form, so ``1``, ``1.0`` and
    ``true`` are three different values. Keys are visited in sorted order at
    every level, so changes come out ordered by path and the same two states
    always produce the same output.

    Args:
        before: The earlier state, as ``load_state`` returns it.
        after: The later state.

    Returns:
        The ordered differences; empty when the states hold the same data.
    """
    changes: list[FlowStateChange] = []
    _collect_changes(before, after, (), changes)
    return FlowStateDiff(changes=tuple(changes))


def _collect_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    path: tuple[str, ...],
    changes: list[FlowStateChange],
) -> None:
    for key in sorted(set(before) | set(after), key=str):
        child = (*path, str(key))
        if key not in before:
            changes.append(FlowStateChange(child, "added", after=after[key]))
        elif key not in after:
            changes.append(FlowStateChange(child, "removed", before=before[key]))
        else:
            old, new = before[key], after[key]
            if isinstance(old, Mapping) and isinstance(new, Mapping):
                _collect_changes(old, new, child, changes)
            elif _canonical(old) != _canonical(new):
                changes.append(FlowStateChange(child, "changed", before=old, after=new))
