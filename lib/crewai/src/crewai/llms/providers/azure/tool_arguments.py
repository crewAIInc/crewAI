from __future__ import annotations

import json
import logging
from typing import Any


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Coerce a tool-call arguments payload into a dict.

    Azure (and some replay/stream paths) can hand back a JSON string, an
    already-parsed dict, or JSON null. ``json.loads`` only accepts str/bytes,
    so dict and None used to raise TypeError and abort the call.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == "":
        return {}
    if not isinstance(arguments, (str, bytes, bytearray)):
        logging.error(
            "Failed to parse tool arguments: expected str or dict, got %s",
            type(arguments).__name__,
        )
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse tool arguments: {e}")
        return {}
    if isinstance(parsed, dict):
        return parsed
    logging.error(
        "Failed to parse tool arguments: JSON value is %s, not an object",
        type(parsed).__name__,
    )
    return {}
