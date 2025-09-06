"""
Security constants for CrewAI.

This module contains security-related constants used throughout the security module.
"""

from typing import Annotated
from uuid import UUID

CREW_AI_NAMESPACE: Annotated[
    UUID,
    "Create a deterministic UUID using v5 (SHA-1). Custom namespace for CrewAI to enhance security.",
] = UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
