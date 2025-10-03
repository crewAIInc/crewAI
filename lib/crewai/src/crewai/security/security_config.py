"""Security Configuration Module

This module provides configuration for CrewAI security features, including:
- Authentication settings
- Scoping rules
- Fingerprinting

The SecurityConfig class is the primary interface for managing security settings
in CrewAI applications.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Self

from crewai.security.fingerprint import Fingerprint


class SecurityConfig(BaseModel):
    """
    Configuration for CrewAI security features.

    This class manages security settings for CrewAI agents, including:
    - Authentication credentials *TODO*
    - Identity information (agent fingerprints)
    - Scoping rules *TODO*
    - Impersonation/delegation tokens *TODO*

    Attributes:
        fingerprint (Fingerprint): The unique fingerprint automatically generated for the component
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
        # Note: Cannot use frozen=True as existing tests modify the fingerprint property
    )

    fingerprint: Fingerprint = Field(
        default_factory=Fingerprint, description="Unique identifier for the component"
    )

    @field_validator("fingerprint", mode="before")
    @classmethod
    def validate_fingerprint(cls, v: Any) -> Fingerprint:
        """Ensure fingerprint is properly initialized."""
        if v is None:
            return Fingerprint()
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Fingerprint seed cannot be empty")
            return Fingerprint.generate(seed=v)
        if isinstance(v, dict):
            return Fingerprint.from_dict(v)
        if isinstance(v, Fingerprint):
            return v

        raise ValueError(f"Invalid fingerprint type: {type(v)}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the security config to a dictionary.

        Returns:
            Dictionary representation of the security config
        """
        return {"fingerprint": self.fingerprint.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Create a SecurityConfig from a dictionary.

        Args:
            data: Dictionary representation of a security config

        Returns:
            A new SecurityConfig instance
        """
        fingerprint_data = data.get("fingerprint")
        fingerprint = (
            Fingerprint.from_dict(fingerprint_data)
            if fingerprint_data
            else Fingerprint()
        )

        return cls(fingerprint=fingerprint)
