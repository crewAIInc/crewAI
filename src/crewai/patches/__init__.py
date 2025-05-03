"""
Patches module for CrewAI.

This module contains patches for dependencies that need to be fixed
without waiting for upstream changes.
"""

from crewai.patches.litellm_patch import apply_patches

# Apply all patches when the module is imported
apply_patches()
