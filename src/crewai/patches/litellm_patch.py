"""
Patch for litellm to fix UnicodeDecodeError on Windows systems.

This patch ensures that all file open operations in litellm use UTF-8 encoding,
which prevents UnicodeDecodeError when loading JSON files on Windows systems
where the default encoding is cp1252 or cp1254.

WARNING: This patch monkey-patches the built-in open() function globally on Windows. 
It forces UTF-8 encoding on all text-mode file opens, which could affect third-party 
libraries expecting default platform encodings. Apply with caution and test comprehensively.
"""

import builtins
import functools
import io
import json
import logging
import os
import sys
from importlib import resources
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def apply_patches():
    """
    Apply patches to fix litellm encoding issues on Windows systems.
    
    This function only applies the patch on Windows platforms where the issue occurs.
    It stores the original open function for proper restoration later.
    """
    # Only apply patch on Windows systems
    if sys.platform != "win32":
        logger.debug("Skipping litellm encoding patches on non-Windows platform")
        return
    
    if hasattr(builtins, '_original_open'):
        logger.debug("Litellm encoding patches already applied")
        return
    
    logger.debug("Applying litellm encoding patches on Windows")
    
    builtins._original_open = builtins.open
    
    @functools.wraps(builtins._original_open)
    def patched_open(
        file, mode='r', buffering=-1, encoding=None, 
        errors=None, newline=None, closefd=True, opener=None
    ):
        if 'r' in mode and encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        
        return builtins._original_open(
            file, mode, buffering, encoding, 
            errors, newline, closefd, opener
        )
    
    builtins.open = patched_open
    
    logger.debug("Successfully applied litellm encoding patches")


def remove_patches():
    """
    Remove all patches (for testing purposes).
    
    This function properly restores the original open function if it was patched.
    """
    if hasattr(builtins, '_original_open'):
        builtins.open = builtins._original_open
        delattr(builtins, '_original_open')
        logger.debug("Removed litellm encoding patches")
