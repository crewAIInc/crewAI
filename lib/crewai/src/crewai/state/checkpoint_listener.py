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


def _find_checkpoint(source: Any) -> CheckpointConfig | None:
    """Find the CheckpointConfig for an event source.

    Walks known relationships: Task -> Agent -> Crew. Flow and Agent
    carry their own checkpoint field directly.
    """
    if isinstance(source, Flow):
        return source.checkpoint
    if isinstance(source, Crew):
        return source.checkpoint
    if isinstance(source, BaseAgent):
        if source.checkpoint is not None:
            return source.checkpoint
        crew = source.crew
        if isinstance(crew, Crew):
            return crew.checkpoint
        return None
    if isinstance(source, Task):
        agent = source.agent
        if isinstance(agent, BaseAgent):
            if agent.checkpoint is not None:
                return agent.checkpoint
            crew = agent.crew
            if isinstance(crew, Crew):
                return crew.checkpoint
        return None
    return None


def _do_checkpoint(state: RuntimeState, cfg: CheckpointConfig) -> None:
    """Write a checkpoint and optionally prune old files."""
    _prepare_entities(state.root)
    data = state.model_dump_json()
    cfg.provider.checkpoint(data, cfg.directory)

    if cfg.max_checkpoints is not None:
        _prune(cfg.directory, cfg.max_checkpoints)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _prune(directory: str, max_keep: int) -> None:
    """Remove oldest checkpoint files beyond *max_keep*."""
    pattern = os.path.join(directory, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for path in files[:-max_keep]:
        _safe_remove(path)


def _on_any_event(source: Any, event: BaseEvent, state: Any) -> None:
    """Handler registered on every event class.

    Checks whether the source entity (or its parent crew/agent) has a
    ``checkpoint`` whose ``trigger_events`` includes this event's
    type string.  If so, writes a checkpoint.
    """
    cfg = _find_checkpoint(source)
    if cfg is None:
        return
    if not cfg.trigger_all and event.type not in cfg.trigger_events:
        return
    try:
        _do_checkpoint(state, cfg)
    except Exception:
        logger.warning("Auto-checkpoint failed for event %s", event.type, exc_info=True)


def setup_checkpoint_handlers(event_bus: CrewAIEventsBus) -> None:
    """Register the checkpoint handler on all known event classes."""
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
