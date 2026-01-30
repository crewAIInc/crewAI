"""File processing hook for CrewAI Platform Tools.

This module provides a hook that processes file markers returned by platform tools
and injects the files into the LLM context for native file handling.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.hooks.tool_hooks import ToolCallHookContext

logger = logging.getLogger(__name__)

_FILE_MARKER_PREFIX = "__CREWAI_FILE__"

_hook_registered = False


def process_file_markers(context: ToolCallHookContext) -> str | None:
    """Process file markers in tool results and inject files into context.

    This hook detects file markers returned by platform tools (e.g., download_file)
    and converts them into FileInput objects that are attached to the hook context.
    The agent executor will then inject these files into the tool message for
    native LLM file handling.

    The marker format is:
        __CREWAI_FILE__:filename:content_type:file_path

    Args:
        context: The tool call hook context containing the tool result.

    Returns:
        A human-readable message if a file was processed, None otherwise.
    """
    result = context.tool_result

    if not result or not result.startswith(_FILE_MARKER_PREFIX):
        return None

    try:
        parts = result.split(":", 3)
        if len(parts) < 4:
            logger.warning(f"Invalid file marker format: {result[:100]}")
            return None

        _, filename, content_type, file_path = parts

        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return f"Error: Downloaded file not found at {file_path}"

        try:
            from crewai_files import File
        except ImportError:
            logger.warning(
                "crewai_files not installed. File will not be attached to LLM context."
            )
            return (
                f"Downloaded file: {filename} ({content_type}). "
                f"File saved at: {file_path}. "
                "Note: Install crewai_files for native LLM file handling."
            )

        file = File(source=file_path, content_type=content_type, filename=filename)

        context.files = {filename: file}

        file_size = os.path.getsize(file_path)
        size_str = _format_file_size(file_size)

        return f"Downloaded file: {filename} ({content_type}, {size_str}). File is attached for LLM analysis."

    except Exception as e:
        logger.exception(f"Error processing file marker: {e}")
        return f"Error processing downloaded file: {e}"


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def register_file_processing_hook() -> bool:
    """Register the file processing hook globally.

    This function should be called once during application initialization
    to enable automatic file injection for platform tools.

    Returns:
        True if the hook was registered, False if it was already registered
        or if registration failed.
    """
    global _hook_registered

    if _hook_registered:
        logger.debug("File processing hook already registered")
        return False

    try:
        from crewai.hooks import register_after_tool_call_hook

        register_after_tool_call_hook(process_file_markers)
        _hook_registered = True
        logger.info("File processing hook registered successfully")
        return True
    except ImportError:
        logger.warning(
            "crewai.hooks not available. File processing hook not registered."
        )
        return False
    except Exception as e:
        logger.exception(f"Failed to register file processing hook: {e}")
        return False
