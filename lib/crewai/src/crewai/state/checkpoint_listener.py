"""Event listener that writes checkpoints automatically.

Handlers are registered lazily — only when the first ``CheckpointConfig``
is resolved (i.e. an entity actually has checkpointing enabled). This
avoids per-event overhead when no entity uses checkpointing.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import CrewAIEventsBus, crewai_event_bus
from crewai.flow.flow import Flow
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.runtime import RuntimeState, _prepare_entities
from crewai.task import Task


logger = logging.getLogger(__name__)

_handlers_registered = False
_register_lock = threading.Lock()

_SENTINEL = object()


def _ensure_handlers_registered() -> None:
    """Register checkpoint handlers on the event bus once, lazily."""
    global _handlers_registered
    if _handlers_registered:
        return
    with _register_lock:
        if _handlers_registered:
            return
        _register_all_handlers(crewai_event_bus)
        _handlers_registered = True


def _resolve(value: CheckpointConfig | bool | None) -> CheckpointConfig | None | object:
    """Coerce a checkpoint field value.

    Returns:
        CheckpointConfig — use this config.
        _SENTINEL — explicit opt-out (``False``), stop walking parents.
        None — not configured, keep walking parents.
    """
    if isinstance(value, CheckpointConfig):
        _ensure_handlers_registered()
        return value
    if value is True:
        _ensure_handlers_registered()
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


def _do_checkpoint(
    state: RuntimeState, cfg: CheckpointConfig, event: BaseEvent | None = None
) -> None:
    """Write a checkpoint and prune old ones if configured."""
    _prepare_entities(state.root)
    payload = state.model_dump(mode="json")
    if event is not None:
        payload["trigger"] = event.type
    data = json.dumps(payload)
    location = cfg.provider.checkpoint(
        data,
        cfg.location,
        parent_id=state._parent_id,
        branch=state._branch,
    )
    state._chain_lineage(cfg.provider, location)

    checkpoint_id: str = cfg.provider.extract_id(location)
    msg: str = (
        f"Checkpoint saved. Resume with: crewai checkpoint resume {checkpoint_id}"
    )
    logger.info(msg)

    if cfg.max_checkpoints is not None:
        cfg.provider.prune(cfg.location, cfg.max_checkpoints, branch=state._branch)


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
        _do_checkpoint(state, cfg, event)
    except Exception:
        logger.warning("Auto-checkpoint failed for event %s", event.type, exc_info=True)


def _register_all_handlers(event_bus: CrewAIEventsBus) -> None:
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
