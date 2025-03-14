"""
Security Configuration Module

This module provides configuration for CrewAI security features, including:
- Authentication settings
- Scoping rules
- Fingerprinting

The SecurityConfig class is the primary interface for managing security settings
in CrewAI applications.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        version (str): Version of the security configuration
        fingerprint (Fingerprint): The unique fingerprint automatically generated for the component
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
        # Note: Cannot use frozen=True as existing tests modify the fingerprint property
    )

    version: str = Field(
        default="1.0.0", 
        description="Version of the security configuration"
    )

    fingerprint: Fingerprint = Field(
        default_factory=Fingerprint, 
        description="Unique identifier for the component"
    )
    
    def is_compatible(self, min_version: str) -> bool:
        """
        Check if this security configuration is compatible with the minimum required version.
        
        Args:
            min_version (str): Minimum required version in semver format (e.g., "1.0.0")
            
        Returns:
            bool: True if this configuration is compatible, False otherwise
        """
        # Simple version comparison (can be enhanced with packaging.version if needed)
        current = [int(x) for x in self.version.split(".")]
        minimum = [int(x) for x in min_version.split(".")]
        
        # Compare major, minor, patch versions
        for c, m in zip(current, minimum):
            if c > m:
                return True
            if c < m:
                return False
        return True

    @model_validator(mode='before')
    @classmethod
    def validate_fingerprint(cls, values):
        """Ensure fingerprint is properly initialized."""
        if isinstance(values, dict):
            # Handle case where fingerprint is not provided or is None
            if 'fingerprint' not in values or values['fingerprint'] is None:
                values['fingerprint'] = Fingerprint()
            # Handle case where fingerprint is a string (seed)
            elif isinstance(values['fingerprint'], str):
                if not values['fingerprint'].strip():
                    raise ValueError("Fingerprint seed cannot be empty")
                values['fingerprint'] = Fingerprint.generate(seed=values['fingerprint'])
        return values

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the security config to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the security config
        """
        result = {
            "fingerprint": self.fingerprint.to_dict()
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
