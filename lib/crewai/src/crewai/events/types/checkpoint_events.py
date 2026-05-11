"""Event family for automatic state checkpointing and forking."""

from typing import Literal

from crewai.events.base_events import BaseEvent


class CheckpointBaseEvent(BaseEvent):
    """Base event for checkpoint lifecycle operations."""

    type: str
    location: str
    provider: str
    trigger: str | None = None
    branch: str | None = None
    parent_id: str | None = None


class CheckpointStartedEvent(CheckpointBaseEvent):
    """Event emitted immediately before a checkpoint is written."""

    type: Literal["checkpoint_started"] = "checkpoint_started"


class CheckpointCompletedEvent(CheckpointBaseEvent):
    """Event emitted when a checkpoint has been written successfully."""

    type: Literal["checkpoint_completed"] = "checkpoint_completed"
    checkpoint_id: str
    duration_ms: float


class CheckpointFailedEvent(CheckpointBaseEvent):
    """Event emitted when a checkpoint write fails."""

    type: Literal["checkpoint_failed"] = "checkpoint_failed"
    error: str


class CheckpointPrunedEvent(CheckpointBaseEvent):
    """Event emitted after pruning old checkpoints from a branch."""

    type: Literal["checkpoint_pruned"] = "checkpoint_pruned"
    removed_count: int
    max_checkpoints: int


class CheckpointForkBaseEvent(BaseEvent):
    """Base event for fork lifecycle operations on a RuntimeState."""

    type: str
    branch: str
    parent_branch: str | None = None
    parent_checkpoint_id: str | None = None


class CheckpointForkStartedEvent(CheckpointForkBaseEvent):
    """Event emitted immediately before a fork relabels the branch."""

    type: Literal["checkpoint_fork_started"] = "checkpoint_fork_started"


class CheckpointForkCompletedEvent(CheckpointForkBaseEvent):
    """Event emitted after a fork has established the new branch."""

    type: Literal["checkpoint_fork_completed"] = "checkpoint_fork_completed"


class CheckpointRestoreBaseEvent(BaseEvent):
    """Base event for checkpoint restore lifecycle operations."""

    type: str
    location: str
    provider: str | None = None


class CheckpointRestoreStartedEvent(CheckpointRestoreBaseEvent):
    """Event emitted immediately before a checkpoint restore begins."""

    type: Literal["checkpoint_restore_started"] = "checkpoint_restore_started"


class CheckpointRestoreCompletedEvent(CheckpointRestoreBaseEvent):
    """Event emitted when a checkpoint has been restored successfully."""

    type: Literal["checkpoint_restore_completed"] = "checkpoint_restore_completed"
    checkpoint_id: str
    branch: str | None = None
    parent_id: str | None = None
    duration_ms: float


class CheckpointRestoreFailedEvent(CheckpointRestoreBaseEvent):
    """Event emitted when a checkpoint restore fails."""

    type: Literal["checkpoint_restore_failed"] = "checkpoint_restore_failed"
    error: str
