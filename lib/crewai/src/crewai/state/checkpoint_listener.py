"""Event listener that writes checkpoints automatically."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus
from crewai.flow.flow import Flow
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.runtime import RuntimeState, _prepare_entities
from crewai.task import Task


logger = logging.getLogger(__name__)


_SENTINEL = object()


def _resolve(value: CheckpointConfig | bool | None) -> CheckpointConfig | None | object:
    """Coerce a checkpoint field value.

    Returns:
        CheckpointConfig — use this config.
        _SENTINEL — explicit opt-out (``False``), stop walking parents.
        None — not configured, keep walking parents.
    """
    if isinstance(value, CheckpointConfig):
        return value
    if value is True:
        return CheckpointConfig()
    if value is False:
        return _SENTINEL
    return None  # None = inherit


def _find_checkpoint(source: Any) -> CheckpointConfig | None:
    """Find the CheckpointConfig for an event source.

    Walks known relationships: Task -> Agent -> Crew. Flow and Agent
    carry their own checkpoint field directly.

    A ``None`` value means "not configured, inherit from parent".
    A ``False`` value means "opt out" and stops the walk.
    """
    if isinstance(source, Flow):
        result = _resolve(source.checkpoint)
        return result if isinstance(result, CheckpointConfig) else None
    if isinstance(source, Crew):
        result = _resolve(source.checkpoint)
        return result if isinstance(result, CheckpointConfig) else None
    if isinstance(source, BaseAgent):
        result = _resolve(source.checkpoint)
        if isinstance(result, CheckpointConfig):
            return result
        if result is _SENTINEL:
            return None
        crew = source.crew
        if isinstance(crew, Crew):
            result = _resolve(crew.checkpoint)
            return result if isinstance(result, CheckpointConfig) else None
        return None
    if isinstance(source, Task):
        agent = source.agent
        if isinstance(agent, BaseAgent):
            result = _resolve(agent.checkpoint)
            if isinstance(result, CheckpointConfig):
                return result
            if result is _SENTINEL:
                return None
            crew = agent.crew
            if isinstance(crew, Crew):
                result = _resolve(crew.checkpoint)
                return result if isinstance(result, CheckpointConfig) else None
        return None
    return None


def _do_checkpoint(state: RuntimeState, cfg: CheckpointConfig) -> None:
    """Write a checkpoint synchronously and optionally prune old files."""
    _prepare_entities(state.root)
    data = state.model_dump_json()
    cfg.provider.checkpoint(data, cfg.directory)

    if cfg.max_checkpoints is not None:
        _prune(cfg.directory, cfg.max_checkpoints)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        logger.debug("Failed to remove checkpoint file %s", path, exc_info=True)


def _prune(directory: str, max_keep: int) -> None:
    """Remove oldest checkpoint files beyond *max_keep*."""
    pattern = os.path.join(directory, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    to_remove = files if max_keep == 0 else files[:-max_keep]
    for path in to_remove:
        _safe_remove(path)


def _should_checkpoint(source: Any, event: BaseEvent) -> CheckpointConfig | None:
    """Return the CheckpointConfig if this event should trigger a checkpoint."""
    cfg = _find_checkpoint(source)
    if cfg is None:
        return None
    if not cfg.trigger_all and event.type not in cfg.trigger_events:
        return None
    return cfg


def _on_any_event(source: Any, event: BaseEvent, state: Any) -> None:
    """Sync handler registered on every event class."""
    cfg = _should_checkpoint(source, event)
    if cfg is None:
        return
    try:
        _do_checkpoint(state, cfg)
    except Exception:
        logger.warning("Auto-checkpoint failed for event %s", event.type, exc_info=True)


def setup_checkpoint_handlers(event_bus: CrewAIEventsBus) -> None:
    """Register the checkpoint handler on all known event classes.

    Only the sync handler is registered. The event bus runs sync handlers
    in a ``ThreadPoolExecutor``, so blocking I/O is safe and we avoid
    writing duplicate checkpoints from both sync and async dispatch.
    """
    seen: set[type] = set()

    def _collect(cls: type[BaseEvent]) -> None:
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                type_field = sub.model_fields.get("type")
                if (
                    type_field
                    and type_field.default
                    and type_field.default != "base_event"
                ):
                    event_bus.register_handler(sub, _on_any_event)
                _collect(sub)

    _collect(BaseEvent)
