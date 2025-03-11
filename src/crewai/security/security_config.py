"""
Security Configuration Module

This module provides configuration for CrewAI security features, including:
- Authentication settings
- Scoping rules
- Fingerprinting

The SecurityConfig class is the primary interface for managing security settings
in CrewAI applications.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

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

    fingerprint: Fingerprint = Field(default_factory=Fingerprint, description="Unique identifier for the component")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, fingerprint: Fingerprint = None):
        """
        Initialize a new SecurityConfig instance.

        Args:
            fingerprint: Fingerprint to use for the component
        """
        self.fingerprint = fingerprint or Fingerprint()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the security config to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the security config
        """
        result = {
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """
        Create a SecurityConfig from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary representation of a security config

        Returns:
            SecurityConfig: A new SecurityConfig instance
        """
        # Make a copy to avoid modifying the original
        data_copy = data.copy()

        fingerprint_data = data_copy.pop("fingerprint", None)
        fingerprint = Fingerprint.from_dict(fingerprint_data) if fingerprint_data else Fingerprint()

        return cls(fingerprint=fingerprint)
