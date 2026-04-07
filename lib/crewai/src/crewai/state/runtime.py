"""Unified runtime state for crewAI.

``RuntimeState`` is a ``RootModel`` whose ``model_dump_json()`` produces a
complete, self-contained snapshot of every active entity in the program.

The ``Entity`` type is resolved at import time in ``crewai/__init__.py``
via ``RuntimeState.model_rebuild()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import (
    ModelWrapValidatorHandler,
    PrivateAttr,
    RootModel,
    model_serializer,
    model_validator,
)

from crewai.context import capture_execution_context
from crewai.state.event_record import EventRecord
from crewai.state.provider.core import BaseProvider
from crewai.state.provider.json_provider import JsonProvider


if TYPE_CHECKING:
    from crewai import Entity


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
    _event_record: EventRecord = PrivateAttr(default_factory=EventRecord)

    @property
    def event_record(self) -> EventRecord:
        """The execution event record."""
        return self._event_record

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {
            "entities": [e.model_dump(mode="json") for e in self.root],
            "event_record": self._event_record.model_dump(),
        }

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize(
        cls, data: Any, handler: ModelWrapValidatorHandler[RuntimeState]
    ) -> RuntimeState:
        if isinstance(data, dict) and "entities" in data:
            record_data = data.get("event_record")
            state = handler(data["entities"])
            if record_data:
                state._event_record = EventRecord.model_validate(record_data)
            return state
        return handler(data)

    def checkpoint(self, location: str) -> str:
        """Write a checkpoint.

        Args:
            location: Storage destination. For JsonProvider this is a directory
                path; for SqliteProvider it is a database file path.

        Returns:
            A location identifier for the saved checkpoint.
        """
        _prepare_entities(self.root)
        return self._provider.checkpoint(self.model_dump_json(), location)

    async def acheckpoint(self, location: str) -> str:
        """Async version of :meth:`checkpoint`.

        Args:
            location: Storage destination. For JsonProvider this is a directory
                path; for SqliteProvider it is a database file path.

        Returns:
            A location identifier for the saved checkpoint.
        """
        _prepare_entities(self.root)
        return await self._provider.acheckpoint(self.model_dump_json(), location)

    @classmethod
    def from_checkpoint(
        cls, location: str, provider: BaseProvider, **kwargs: Any
    ) -> RuntimeState:
        """Restore a RuntimeState from a checkpoint.

        Args:
            location: The identifier returned by a previous ``checkpoint`` call.
            provider: The storage backend to read from.
            **kwargs: Passed to ``model_validate_json``.

        Returns:
            A restored RuntimeState.
        """
        raw = provider.from_checkpoint(location)
        return cls.model_validate_json(raw, **kwargs)

    @classmethod
    async def afrom_checkpoint(
        cls, location: str, provider: BaseProvider, **kwargs: Any
    ) -> RuntimeState:
        """Async version of :meth:`from_checkpoint`.

        Args:
            location: The identifier returned by a previous ``acheckpoint`` call.
            provider: The storage backend to read from.
            **kwargs: Passed to ``model_validate_json``.

        Returns:
            A restored RuntimeState.
        """
        raw = await provider.afrom_checkpoint(location)
        return cls.model_validate_json(raw, **kwargs)


def _prepare_entities(root: list[Entity]) -> None:
    """Capture execution context and sync checkpoint fields on each entity.

    Args:
        root: List of entities to prepare for serialization.
    """
    for entity in root:
        entity.execution_context = capture_execution_context()
        _sync_checkpoint_fields(entity)
