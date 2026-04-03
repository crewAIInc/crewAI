"""Unified runtime state for crewAI.

``RuntimeState`` is a ``RootModel`` whose ``model_dump_json()`` produces a
complete, self-contained snapshot of every active entity in the program.

The ``Entity`` type is resolved at import time in ``crewai/__init__.py``
via ``RuntimeState.model_rebuild()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr, RootModel

from crewai.context import capture_execution_context
from crewai.state.provider.core import BaseProvider
from crewai.state.provider.json_provider import JsonProvider


if TYPE_CHECKING:
    from crewai import Entity


def _entity_discriminator(v: dict[str, Any] | object) -> str:
    if isinstance(v, dict):
        raw = v.get("entity_type", "agent")
    else:
        raw = getattr(v, "entity_type", "agent")
    return str(raw)


def _sync_checkpoint_fields(entity: object) -> None:
    """Copy private runtime attrs into checkpoint fields before serializing.

    Args:
        entity: The entity whose private runtime attributes will be
            copied into its public checkpoint fields.
    """
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
    root: list[Entity]
    _provider: BaseProvider = PrivateAttr(default_factory=JsonProvider)

    def checkpoint(self, directory: str) -> str:
        """Write a checkpoint file to the directory.

        Args:
            directory: Filesystem path where the checkpoint JSON will be saved.

        Returns:
            A location identifier for the saved checkpoint.
        """
        _prepare_entities(self.root)
        return self._provider.checkpoint(self.model_dump_json(), directory)

    async def acheckpoint(self, directory: str) -> str:
        """Async version of :meth:`checkpoint`.

        Args:
            directory: Filesystem path where the checkpoint JSON will be saved.

        Returns:
            A location identifier for the saved checkpoint.
        """
        _prepare_entities(self.root)
        return await self._provider.acheckpoint(self.model_dump_json(), directory)


def _prepare_entities(root: list[Entity]) -> None:
    """Capture execution context and sync checkpoint fields on each entity.

    Args:
        root: List of entities to prepare for serialization.
    """
    for entity in root:
        entity.execution_context = capture_execution_context()
        _sync_checkpoint_fields(entity)
