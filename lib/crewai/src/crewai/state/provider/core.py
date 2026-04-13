"""Base class for state providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseProvider(BaseModel, ABC):
    """Base class for persisting and restoring runtime state checkpoints.

    Implementations handle the storage backend — filesystem, cloud, database,
    etc. — while ``RuntimeState`` handles serialization.
    """

    provider_type: str = "base"

    @abstractmethod
    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Persist a snapshot synchronously.

        Args:
            data: The serialized string to persist.
            location: Storage destination (directory, file path, URI, etc.).
            parent_id: ID of the parent checkpoint for lineage tracking.
            branch: Branch label for this checkpoint.

        Returns:
            A location identifier for the saved checkpoint.
        """
        ...

    @abstractmethod
    async def acheckpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Persist a snapshot asynchronously.

        Args:
            data: The serialized string to persist.
            location: Storage destination (directory, file path, URI, etc.).
            parent_id: ID of the parent checkpoint for lineage tracking.
            branch: Branch label for this checkpoint.

        Returns:
            A location identifier for the saved checkpoint.
        """
        ...

    @abstractmethod
    def prune(self, location: str, max_keep: int, *, branch: str = "main") -> None:
        """Remove old checkpoints, keeping at most *max_keep* per branch.

        Args:
            location: The storage destination passed to ``checkpoint``.
            max_keep: Maximum number of checkpoints to retain.
            branch: Only prune checkpoints on this branch.
        """
        ...

    @abstractmethod
    def extract_id(self, location: str) -> str:
        """Extract the checkpoint ID from a location string.

        Args:
            location: The identifier returned by a previous ``checkpoint`` call.

        Returns:
            The checkpoint ID.
        """
        ...

    @abstractmethod
    def from_checkpoint(self, location: str) -> str:
        """Read a snapshot synchronously.

        Args:
            location: The identifier returned by a previous ``checkpoint`` call.

        Returns:
            The raw serialized string.
        """
        ...

    @abstractmethod
    async def afrom_checkpoint(self, location: str) -> str:
        """Read a snapshot asynchronously.

        Args:
            location: The identifier returned by a previous ``acheckpoint`` call.

        Returns:
            The raw serialized string.
        """
        ...
