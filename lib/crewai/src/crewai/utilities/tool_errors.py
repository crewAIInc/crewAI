"""Structured tool error formatting for agent consumption.

When a tool raises an exception, the agent needs structured information
to decide whether to retry, fix its input, or skip the tool entirely.
This module provides a consistent error format across all executors.
"""

import json
import traceback

RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def format_tool_error(exception: Exception, include_traceback: bool = False) -> str:
    """Format a tool execution error as structured JSON for the agent.

    Returns a string with the "Error executing tool:" prefix (for backward
    compatibility with existing parsing) followed by a JSON object containing
    the exception type, message, and retryability hint.
    """
    error_data = {
        "error": True,
        "type": type(exception).__name__,
        "message": str(exception),
        "retryable": isinstance(exception, RETRYABLE_EXCEPTIONS),
    }
    if include_traceback:
        error_data["traceback"] = traceback.format_exc(limit=3)
    return f"Error executing tool: {json.dumps(error_data)}"
