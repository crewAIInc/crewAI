"""
Fingerprint Module

This module provides functionality for generating and validating unique identifiers
for CrewAI agents. These identifiers are used for tracking, auditing, and security.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Fingerprint(BaseModel):
    """
    A class for generating and managing unique identifiers for agents.

    Each agent has dual identifiers:
    - Human-readable ID: For debugging and reference (derived from role if not specified)
    - Fingerprint UUID: Unique runtime identifier for tracking and auditing

    Attributes:
        uuid_str (str): String representation of the UUID for this fingerprint, auto-generated
        created_at (datetime): When this fingerprint was created, auto-generated
        metadata (Dict[str, Any]): Additional metadata associated with this fingerprint
    """

    uuid_str: str = Field(default_factory=lambda: str(uuid.uuid4()), description="String representation of the UUID")
    created_at: datetime = Field(default_factory=datetime.now, description="When this fingerprint was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this fingerprint")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
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
                        raise ValueError(f"Nested metadata keys must be strings, got {type(nested_key)}")
                    if isinstance(nested_value, dict):
                        raise ValueError("Metadata can only be nested one level deep")
        
        # Check for maximum metadata size (prevent DoS)
        if len(str(v)) > 10000:  # Limit metadata size to 10KB
            raise ValueError("Metadata size exceeds maximum allowed (10KB)")
            
        return v

    def __init__(self, **data):
        """Initialize a Fingerprint with auto-generated uuid_str and created_at."""
        # Remove uuid_str and created_at from data to ensure they're auto-generated
        if 'uuid_str' in data:
            data.pop('uuid_str')
        if 'created_at' in data:
            data.pop('created_at')

        # Call the parent constructor with the modified data
        super().__init__(**data)

    @property
    def uuid(self) -> uuid.UUID:
        """Get the UUID object for this fingerprint."""
        return uuid.UUID(self.uuid_str)

    @classmethod
    def _generate_uuid(cls, seed: str) -> str:
        """
        Generate a deterministic UUID based on a seed string.

        Args:
            seed (str): The seed string to use for UUID generation

        Returns:
            str: A string representation of the UUID consistently generated from the seed
        """
        if not isinstance(seed, str):
            raise ValueError("Seed must be a string")
        
        if not seed.strip():
            raise ValueError("Seed cannot be empty or whitespace")
            
        # Create a deterministic UUID using v5 (SHA-1)
        # Custom namespace for CrewAI to enhance security

        # Using a unique namespace specific to CrewAI to reduce collision risks
        CREW_AI_NAMESPACE = uuid.UUID('f47ac10b-58cc-4372-a567-0e02b2c3d479')
        return str(uuid.uuid5(CREW_AI_NAMESPACE, seed))

    @classmethod
    def generate(cls, seed: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> 'Fingerprint':
        """
        Static factory method to create a new Fingerprint.

        Args:
            seed (Optional[str]): A string to use as seed for the UUID generation.
                If None, a random UUID is generated.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store with the fingerprint.

        Returns:
            Fingerprint: A new Fingerprint instance
        """
        fingerprint = cls(metadata=metadata or {})
        if seed:
            # For seed-based generation, we need to manually set the uuid_str after creation
            object.__setattr__(fingerprint, 'uuid_str', cls._generate_uuid(seed))
        return fingerprint

    def __str__(self) -> str:
        """String representation of the fingerprint (the UUID)."""
        return self.uuid_str

    def __eq__(self, other) -> bool:
        """Compare fingerprints by their UUID."""
        if isinstance(other, Fingerprint):
            return self.uuid_str == other.uuid_str
        return False

    def __hash__(self) -> int:
        """Hash of the fingerprint (based on UUID)."""
        return hash(self.uuid_str)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the fingerprint to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the fingerprint
        """
        return {
            "uuid_str": self.uuid_str,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fingerprint':
        """
        Create a Fingerprint from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of a fingerprint

        Returns:
            Fingerprint: A new Fingerprint instance
        """
        if not data:
            return cls()

        fingerprint = cls(metadata=data.get("metadata", {}))

        # For consistency with existing stored fingerprints, we need to manually set these
        if "uuid_str" in data:
            object.__setattr__(fingerprint, 'uuid_str', data["uuid_str"])
        if "created_at" in data and isinstance(data["created_at"], str):
            object.__setattr__(fingerprint, 'created_at', datetime.fromisoformat(data["created_at"]))

        return fingerprint
