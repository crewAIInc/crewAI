"""
CrewAI security module.

This module provides security-related functionality for CrewAI, including:
- Fingerprinting for component identity and tracking
- Security configuration for controlling access and permissions
- Encrypted agent-to-agent communication
- Future: authentication, scoping, and delegation mechanisms
"""

from crewai.security.fingerprint import Fingerprint
from crewai.security.security_config import SecurityConfig
from crewai.security.encrypted_communication import (
    AgentCommunicationEncryption, 
    EncryptedMessage
)

__all__ = ["Fingerprint", "SecurityConfig", "AgentCommunicationEncryption", "EncryptedMessage"]
