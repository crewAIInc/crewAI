"""Unified runtime state for crewAI.

``RuntimeState`` is a ``RootModel`` whose ``model_dump_json()`` produces a
complete, self-contained snapshot of every active entity in the program.

The ``Entity`` type is resolved at import time in ``crewai/__init__.py``
via ``RuntimeState.model_rebuild()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

from pydantic import RootModel


if TYPE_CHECKING:
    pass


def _entity_discriminator(v: dict[str, Any] | object) -> str:
    if isinstance(v, dict):
        raw = v.get("entity_type", "agent")
    else:
        raw = getattr(v, "entity_type", "agent")
    return str(raw)


def _sync_checkpoint_fields(entity: object) -> None:
    """Copy private runtime attrs into checkpoint fields before serializing."""
    from crewai.crew import Crew
    from crewai.flow.flow import Flow

    if isinstance(entity, Flow):
        entity.checkpoint_completed_methods = (
            set(entity._completed_methods) if entity._completed_methods else None
        )
        entity.checkpoint_method_outputs = (
            list(entity._method_outputs) if entity._method_outputs else None
        )
        entity.checkpoint_method_counts = (
            {str(k): v for k, v in entity._method_execution_counts.items()}
            if entity._method_execution_counts
            else None
        )
        entity.checkpoint_state = (
            entity._copy_and_serialize_state() if entity._state is not None else None
        )
    if isinstance(entity, Crew):
        entity.checkpoint_inputs = entity._inputs
        entity.checkpoint_train = entity._train
        entity.checkpoint_kickoff_event_id = entity._kickoff_event_id


class RuntimeState(RootModel):  # type: ignore[type-arg]
    root: list[Entity]  # type: ignore[name-defined]  # noqa: F821

    def checkpoint(self, directory: str) -> str:
        """Write a checkpoint file to the directory."""
        from crewai.context import capture_execution_context

        for entity in self.root:
            entity.execution_context = capture_execution_context()
            _sync_checkpoint_fields(entity)

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{ts}_{uuid.uuid4().hex[:8]}.json"
        file_path = dir_path / filename
        file_path.write_text(self.model_dump_json())
        return str(file_path)
