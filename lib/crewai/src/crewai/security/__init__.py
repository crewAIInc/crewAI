"""
CrewAI security module.

This module provides security-related functionality for CrewAI, including:
- Fingerprinting for component identity and tracking
- Security configuration for controlling access and permissions
- Future: authentication, scoping, and delegation mechanisms
"""

from crewai.security.fingerprint import Fingerprint
from crewai.security.security_config import SecurityConfig
from crewai.utilities.exceptions.task_security_exceptions import (
    IterationLimitExceeded,
    PreTaskAttestationError,
    SpendLimitExceeded,
)


__all__ = [
    "Fingerprint",
    "IterationLimitExceeded",
    "PreTaskAttestationError",
    "SecurityConfig",
    "SpendLimitExceeded",
]
