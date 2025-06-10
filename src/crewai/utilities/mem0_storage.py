import re
from typing import Optional

MIN_AGENT_ID_LENGTH = 3
MAX_AGENT_ID_LENGTH = 255
DEFAULT_AGENT_ID = "default_agent_id"

# Compiled regex patterns for better performance
INVALID_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
IPV4_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def is_ipv4_pattern(name: str) -> bool:
    """
    Check if a string matches an IPv4 address pattern.

    Args:
        name: The string to check

    Returns:
        True if the string matches an IPv4 pattern, False otherwise
    """
    return bool(IPV4_PATTERN.match(name))


def sanitize_agent_id_name(name: Optional[str]) -> str:
    """
    Sanitize a agent_id name to meet Mem0 storage requirements:
    1. 3-255 characters long
    2. Starts and ends with alphanumeric character
    3. Contains only alphanumeric characters, underscores, or hyphens
    4. No consecutive periods
    5. Not a valid IPv4 address

    Args:
        name: The original agent_id name to sanitize

    Returns:
        A sanitized agent id name that meets Mem0 storage requirements
    """
    if not name:
        return DEFAULT_AGENT_ID

    if is_ipv4_pattern(name):
        name = f"ip_{name}"

    sanitized = INVALID_CHARS_PATTERN.sub("_", name)

    if not sanitized[0].isalnum():
        sanitized = "a" + sanitized

    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + "z"

    if len(sanitized) < MIN_AGENT_ID_LENGTH:
        sanitized = sanitized + "x" * (MIN_AGENT_ID_LENGTH - len(sanitized))
    if len(sanitized) > MAX_AGENT_ID_LENGTH:
        sanitized = sanitized[:MAX_AGENT_ID_LENGTH]
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + "z"

    return sanitized
