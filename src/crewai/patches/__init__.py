"""
Patches module for CrewAI.

This module contains patches for external dependencies to fix known issues.

Version: 1.0.0
"""

from crewai.patches.litellm_patch import apply_patches, patch_litellm_ollama_pt

__all__ = ["apply_patches", "patch_litellm_ollama_pt"]
