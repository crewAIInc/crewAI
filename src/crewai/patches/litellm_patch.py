"""
Patch for litellm to fix UnicodeDecodeError on Windows systems.

This patch ensures that all file open operations in litellm use UTF-8 encoding,
which prevents UnicodeDecodeError when loading JSON files on Windows systems
where the default encoding is cp1252 or cp1254.
"""

import builtins
import functools
import io
import json
import logging
import os
from importlib import resources
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def apply_patches():
    """Apply all patches to fix litellm encoding issues."""
    logger.info("Applying litellm encoding patches")
    
    original_open = builtins.open
    
    @functools.wraps(original_open)
    def patched_open(
        file, mode='r', buffering=-1, encoding=None, 
        errors=None, newline=None, closefd=True, opener=None
    ):
        if 'r' in mode and encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        
        return original_open(
            file, mode, buffering, encoding, 
            errors, newline, closefd, opener
        )
    
    builtins.open = patched_open
    
    logger.info("Successfully applied litellm encoding patches")


def remove_patches():
    """Remove all patches (for testing purposes)."""
    if hasattr(builtins, '_original_open'):
        builtins.open = builtins._original_open
        logger.info("Removed litellm encoding patches")
