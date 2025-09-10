"""Fingerprint Module

This module provides functionality for generating and validating unique identifiers
for CrewAI agents. These identifiers are used for tracking, auditing, and security.
"""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, BeforeValidator, Field, PrivateAttr
from typing_extensions import Self

from crewai.security.constants import CREW_AI_NAMESPACE


def _validate_metadata(v: Any) -> dict[str, Any]:
    """Validate that metadata is a dictionary with string keys and valid values."""
    if not isinstance(v, dict):
        raise ValueError("Metadata must be a dictionary")

    # Validate that all keys are strings
    for key, value in v.items():
        if not isinstance(key, str):
            raise ValueError(f"Metadata keys must be strings, got {type(key)}")

        # Validate nested dictionaries (prevent deeply nested structures)
        if isinstance(value, dict):
            # Check for nested dictionaries (limit depth to 1)
            for nested_key, nested_value in value.items():
                if not isinstance(nested_key, str):
                    raise ValueError(
                        f"Nested metadata keys must be strings, got {type(nested_key)}"
                    )
                if isinstance(nested_value, dict):
                    raise ValueError("Metadata can only be nested one level deep")

    # Check for maximum metadata size (prevent DoS)
    if len(str(v)) > 10_000:  # Limit metadata size to 10KB
        raise ValueError("Metadata size exceeds maximum allowed (10KB)")

    return v


class Fingerprint(BaseModel):
    """A class for generating and managing unique identifiers for agents.

    Each agent has dual identifiers:
    - Human-readable ID: For debugging and reference (derived from role if not specified)
    - Fingerprint UUID: Unique runtime identifier for tracking and auditing

    Attributes:
        uuid_str: String representation of the UUID for this fingerprint, auto-generated
        created_at: When this fingerprint was created, auto-generated
        metadata: Additional metadata associated with this fingerprint
    """

    _uuid_str: str = PrivateAttr(default_factory=lambda: str(uuid4()))
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)
    metadata: Annotated[dict[str, Any], BeforeValidator(_validate_metadata)] = Field(
        default_factory=dict
    )

    @property
    def uuid_str(self) -> str:
        """Get the string representation of the UUID for this fingerprint."""
        return self._uuid_str

    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp for this fingerprint."""
        return self._created_at

    @property
    def uuid(self) -> UUID:
        """Get the UUID object for this fingerprint."""
        return UUID(self.uuid_str)

    @classmethod
    def _generate_uuid(cls, seed: str) -> str:
        """Generate a deterministic UUID based on a seed string.

        Args:
            seed: The seed string to use for UUID generation

        Returns:
            A string representation of the UUID consistently generated from the seed
        """
        if not seed.strip():
            raise ValueError("Seed cannot be empty or whitespace")

        return str(uuid5(CREW_AI_NAMESPACE, seed))

    @classmethod
    def generate(
        cls, seed: str | None = None, metadata: dict[str, Any] | None = None
    ) -> Self:
        """Static factory method to create a new Fingerprint.

        Args:
            seed: A string to use as seed for the UUID generation.
                If None, a random UUID is generated.
            metadata: Additional metadata to store with the fingerprint.

        Returns:
            A new Fingerprint instance
        """
        fingerprint = cls(metadata=metadata or {})
        if seed:
            # For seed-based generation, we need to manually set the _uuid_str after creation
            fingerprint.__dict__["_uuid_str"] = cls._generate_uuid(seed)
        return fingerprint

    def __str__(self) -> str:
        """String representation of the fingerprint (the UUID)."""
        return self.uuid_str

    def __eq__(self, other: Any) -> bool:
        """Compare fingerprints by their UUID."""
        if type(other) is Fingerprint:
            return self.uuid_str == other.uuid_str
        return False

    def __hash__(self) -> int:
        """Hash of the fingerprint (based on UUID)."""
        return hash(self.uuid_str)

    def to_dict(self) -> dict[str, Any]:
        """Convert the fingerprint to a dictionary representation.

        Returns:
            Dictionary representation of the fingerprint
        """
        return {
            "uuid_str": self.uuid_str,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a Fingerprint from a dictionary representation.

        Args:
            data: Dictionary representation of a fingerprint

        Returns:
            A new Fingerprint instance
        """
        if not data:
            return cls()

        fingerprint = cls(metadata=data.get("metadata", {}))

        # For consistency with existing stored fingerprints, we need to manually set these
        if "uuid_str" in data:
            fingerprint.__dict__["_uuid_str"] = data["uuid_str"]
        if "created_at" in data and isinstance(data["created_at"], str):
            fingerprint.__dict__["_created_at"] = datetime.fromisoformat(
                data["created_at"]
            )

        return fingerprint
