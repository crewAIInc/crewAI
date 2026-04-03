"""Base protocol for state providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


@runtime_checkable
class BaseProvider(Protocol):
    """Interface for persisting and restoring runtime state checkpoints.

    Implementations handle the storage backend — filesystem, cloud, database,
    etc. — while ``RuntimeState`` handles serialization.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Allow Pydantic to validate any ``BaseProvider`` instance."""

        def _validate(v: Any) -> BaseProvider:
            if isinstance(v, BaseProvider):
                return v
            raise TypeError(f"Expected a BaseProvider instance, got {type(v)}")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: type(v).__name__, info_arg=False
            ),
        )

    def checkpoint(self, data: str, directory: str) -> str:
        """Persist a snapshot synchronously.

        Args:
            data: The serialized string to persist.
            directory: Logical destination: path, bucket prefix, etc.

        Returns:
            A location identifier for the saved checkpoint, such as a file path or URI.
        """
        ...

    async def acheckpoint(self, data: str, directory: str) -> str:
        """Persist a snapshot asynchronously.

        Args:
            data: The serialized string to persist.
            directory: Logical destination: path, bucket prefix, etc.

        Returns:
            A location identifier for the saved checkpoint, such as a file path or URI.
        """
        ...

    def from_checkpoint(self, location: str) -> str:
        """Read a snapshot synchronously.

        Args:
            location: The identifier returned by a previous ``checkpoint`` call.

        Returns:
            The raw serialized string.
        """
        ...

    async def afrom_checkpoint(self, location: str) -> str:
        """Read a snapshot asynchronously.

        Args:
            location: The identifier returned by a previous ``acheckpoint`` call.

        Returns:
            The raw serialized string.
        """
        ...
