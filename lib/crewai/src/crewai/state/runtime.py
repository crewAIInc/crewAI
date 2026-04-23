"""Unified runtime state for crewAI.

``RuntimeState`` is a ``RootModel`` whose ``model_dump_json()`` produces a
complete, self-contained snapshot of every active entity in the program.

The ``Entity`` type is resolved at import time in ``crewai/__init__.py``
via ``RuntimeState.model_rebuild()``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
import uuid

from packaging.version import Version
from pydantic import (
    ModelWrapValidatorHandler,
    PrivateAttr,
    RootModel,
    model_serializer,
    model_validator,
)

from crewai.context import capture_execution_context
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.checkpoint_events import (
    CheckpointCompletedEvent,
    CheckpointFailedEvent,
    CheckpointForkCompletedEvent,
    CheckpointForkStartedEvent,
    CheckpointRestoreCompletedEvent,
    CheckpointRestoreFailedEvent,
    CheckpointRestoreStartedEvent,
    CheckpointStartedEvent,
)
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.event_record import EventRecord
from crewai.state.provider.core import BaseProvider
from crewai.state.provider.json_provider import JsonProvider
from crewai.utilities.version import get_crewai_version


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from crewai import Entity


def _sync_checkpoint_fields(entity: object) -> None:
    """Copy private runtime attrs into checkpoint fields before serializing.

    Args:
        entity: The entity whose private runtime attributes will be
            copied into its public checkpoint fields.
    """
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.flow.flow import Flow

    if isinstance(entity, BaseAgent):
        entity.checkpoint_kickoff_event_id = entity._kickoff_event_id
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
        for task in entity.tasks:
            task.checkpoint_original_description = task._original_description
            task.checkpoint_original_expected_output = task._original_expected_output


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Apply version-based migrations to checkpoint data.

    Each block handles checkpoints older than a specific version,
    transforming them forward to the current format. Blocks run in
    version order so migrations compose.

    Args:
        data: The raw deserialized checkpoint dict.

    Returns:
        The migrated checkpoint dict.
    """
    raw = data.get("crewai_version")
    current = Version(get_crewai_version())
    stored = Version(raw) if raw else Version("0.0.0")

    if raw is None:
        logger.warning("Checkpoint has no crewai_version — treating as 0.0.0")
    elif stored != current:
        logger.debug(
            "Migrating checkpoint from crewAI %s to %s",
            stored,
            current,
        )

    # --- migrations in version order ---
    # if stored < Version("X.Y.Z"):
    #     data.setdefault("some_field", "default")

    return data


class RuntimeState(RootModel):  # type: ignore[type-arg]
    root: list[Entity]
    _provider: BaseProvider = PrivateAttr(default_factory=JsonProvider)
    _event_record: EventRecord = PrivateAttr(default_factory=EventRecord)
    _checkpoint_id: str | None = PrivateAttr(default=None)
    _parent_id: str | None = PrivateAttr(default=None)
    _branch: str = PrivateAttr(default="main")

    @property
    def event_record(self) -> EventRecord:
        """The execution event record."""
        return self._event_record

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {
            "crewai_version": get_crewai_version(),
            "parent_id": self._parent_id,
            "branch": self._branch,
            "entities": [e.model_dump(mode="json") for e in self.root],
            "event_record": self._event_record.model_dump(mode="json"),
        }

    @model_validator(mode="wrap")
    @classmethod
    def _deserialize(
        cls, data: Any, handler: ModelWrapValidatorHandler[RuntimeState]
    ) -> RuntimeState:
        if isinstance(data, dict) and "entities" in data:
            data = _migrate(data)
            record_data = data.get("event_record")
            state = handler(data["entities"])
            if record_data:
                state._event_record = EventRecord.model_validate(record_data)
            state._parent_id = data.get("parent_id")
            state._branch = data.get("branch", "main")
            return state
        return handler(data)

    def _chain_lineage(self, provider: BaseProvider, location: str) -> None:
        """Update lineage fields after a successful checkpoint write.

        Sets ``_checkpoint_id`` and ``_parent_id`` so the next write
        records the correct parent in the lineage chain.

        Args:
            provider: The provider that performed the write.
            location: The location string returned by the provider.
        """
        self._checkpoint_id = provider.extract_id(location)
        self._parent_id = self._checkpoint_id

    def checkpoint(self, location: str) -> str:
        """Write a checkpoint.

        Args:
            location: Storage destination. For JsonProvider this is a directory
                path; for SqliteProvider it is a database file path.

        Returns:
            A location identifier for the saved checkpoint.
        """
        provider_name: str = type(self._provider).__name__
        parent_id_snapshot: str | None = self._parent_id
        branch_snapshot: str = self._branch
        crewai_event_bus.emit(
            self,
            CheckpointStartedEvent(
                location=location,
                provider=provider_name,
                branch=branch_snapshot,
                parent_id=parent_id_snapshot,
            ),
        )
        start: float = time.perf_counter()
        try:
            _prepare_entities(self.root)
            result = self._provider.checkpoint(
                self.model_dump_json(),
                location,
                parent_id=parent_id_snapshot,
                branch=branch_snapshot,
            )
            self._chain_lineage(self._provider, result)
        except Exception as exc:
            crewai_event_bus.emit(
                self,
                CheckpointFailedEvent(
                    location=location,
                    provider=provider_name,
                    branch=branch_snapshot,
                    parent_id=parent_id_snapshot,
                    error=str(exc),
                ),
            )
            raise

        crewai_event_bus.emit(
            self,
            CheckpointCompletedEvent(
                location=result,
                provider=provider_name,
                branch=branch_snapshot,
                parent_id=parent_id_snapshot,
                checkpoint_id=self._provider.extract_id(result),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            ),
        )
        return result

    async def acheckpoint(self, location: str) -> str:
        """Async version of :meth:`checkpoint`.

        Args:
            location: Storage destination. For JsonProvider this is a directory
                path; for SqliteProvider it is a database file path.

        Returns:
            A location identifier for the saved checkpoint.
        """
        provider_name: str = type(self._provider).__name__
        parent_id_snapshot: str | None = self._parent_id
        branch_snapshot: str = self._branch
        crewai_event_bus.emit(
            self,
            CheckpointStartedEvent(
                location=location,
                provider=provider_name,
                branch=branch_snapshot,
                parent_id=parent_id_snapshot,
            ),
        )
        start: float = time.perf_counter()
        try:
            _prepare_entities(self.root)
            result = await self._provider.acheckpoint(
                self.model_dump_json(),
                location,
                parent_id=parent_id_snapshot,
                branch=branch_snapshot,
            )
            self._chain_lineage(self._provider, result)
        except Exception as exc:
            crewai_event_bus.emit(
                self,
                CheckpointFailedEvent(
                    location=location,
                    provider=provider_name,
                    branch=branch_snapshot,
                    parent_id=parent_id_snapshot,
                    error=str(exc),
                ),
            )
            raise

        crewai_event_bus.emit(
            self,
            CheckpointCompletedEvent(
                location=result,
                provider=provider_name,
                branch=branch_snapshot,
                parent_id=parent_id_snapshot,
                checkpoint_id=self._provider.extract_id(result),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            ),
        )
        return result

    def fork(self, branch: str | None = None) -> None:
        """Create a new execution branch and write an initial checkpoint.

        If this state was restored from a checkpoint, an initial checkpoint
        is written on the new branch so the fork point is recorded.

        Args:
            branch: Branch label. Auto-generated from the current checkpoint
                ID if not provided. Always unique — safe to call multiple
                times without collisions.
        """
        if branch:
            new_branch = branch
        elif self._checkpoint_id:
            new_branch = f"fork/{self._checkpoint_id}_{uuid.uuid4().hex[:6]}"
        else:
            new_branch = f"fork/{uuid.uuid4().hex[:8]}"

        parent_branch: str | None = self._branch
        parent_checkpoint_id: str | None = self._checkpoint_id

        crewai_event_bus.emit(
            self,
            CheckpointForkStartedEvent(
                branch=new_branch,
                parent_branch=parent_branch,
                parent_checkpoint_id=parent_checkpoint_id,
            ),
        )
        self._branch = new_branch
        crewai_event_bus.emit(
            self,
            CheckpointForkCompletedEvent(
                branch=new_branch,
                parent_branch=parent_branch,
                parent_checkpoint_id=parent_checkpoint_id,
            ),
        )

    @classmethod
    def from_checkpoint(cls, config: CheckpointConfig, **kwargs: Any) -> RuntimeState:
        """Restore a RuntimeState from a checkpoint.

        Args:
            config: Checkpoint configuration with ``restore_from`` set.
            **kwargs: Passed to ``model_validate_json``.

        Returns:
            A restored RuntimeState.
        """
        from crewai.state.provider.utils import detect_provider

        if config.restore_from is None:
            raise ValueError("CheckpointConfig.restore_from must be set")
        location = str(config.restore_from)

        crewai_event_bus.emit(config, CheckpointRestoreStartedEvent(location=location))
        start: float = time.perf_counter()
        provider_name: str | None = None
        try:
            provider = detect_provider(location)
            provider_name = type(provider).__name__
            raw = provider.from_checkpoint(location)
            state = cls.model_validate_json(raw, **kwargs)
            state._provider = provider
            checkpoint_id = provider.extract_id(location)
            state._checkpoint_id = checkpoint_id
            state._parent_id = checkpoint_id
        except Exception as exc:
            crewai_event_bus.emit(
                config,
                CheckpointRestoreFailedEvent(
                    location=location,
                    provider=provider_name,
                    error=str(exc),
                ),
            )
            raise

        crewai_event_bus.emit(
            config,
            CheckpointRestoreCompletedEvent(
                location=location,
                provider=provider_name,
                checkpoint_id=checkpoint_id,
                branch=state._branch,
                parent_id=state._parent_id,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            ),
        )
        return state

    @classmethod
    async def afrom_checkpoint(
        cls, config: CheckpointConfig, **kwargs: Any
    ) -> RuntimeState:
        """Async version of :meth:`from_checkpoint`.

        Args:
            config: Checkpoint configuration with ``restore_from`` set.
            **kwargs: Passed to ``model_validate_json``.

        Returns:
            A restored RuntimeState.
        """
        from crewai.state.provider.utils import detect_provider

        if config.restore_from is None:
            raise ValueError("CheckpointConfig.restore_from must be set")
        location = str(config.restore_from)

        crewai_event_bus.emit(config, CheckpointRestoreStartedEvent(location=location))
        start: float = time.perf_counter()
        provider_name: str | None = None
        try:
            provider = detect_provider(location)
            provider_name = type(provider).__name__
            raw = await provider.afrom_checkpoint(location)
            state = cls.model_validate_json(raw, **kwargs)
            state._provider = provider
            checkpoint_id = provider.extract_id(location)
            state._checkpoint_id = checkpoint_id
            state._parent_id = checkpoint_id
        except Exception as exc:
            crewai_event_bus.emit(
                config,
                CheckpointRestoreFailedEvent(
                    location=location,
                    provider=provider_name,
                    error=str(exc),
                ),
            )
            raise

        crewai_event_bus.emit(
            config,
            CheckpointRestoreCompletedEvent(
                location=location,
                provider=provider_name,
                checkpoint_id=checkpoint_id,
                branch=state._branch,
                parent_id=state._parent_id,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            ),
        )
        return state


def _prepare_entities(root: list[Entity]) -> None:
    """Capture execution context and sync checkpoint fields on each entity.

    Args:
        root: List of entities to prepare for serialization.
    """
    for entity in root:
        entity.execution_context = capture_execution_context()
        _sync_checkpoint_fields(entity)
