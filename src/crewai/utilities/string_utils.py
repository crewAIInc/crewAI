import re
from typing import Optional

# Constants for ChromaDB collection name requirements
MIN_LENGTH = 3
MAX_LENGTH = 63
DEFAULT_COLLECTION = "default_collection"

# Compiled regex patterns for better performance
INVALID_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9_-]')
IPV4_PATTERN = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')


def is_ipv4_pattern(name: str) -> bool:
    """
    Check if a string matches an IPv4 address pattern.
    
    Args:
        name: The string to check
        
    Returns:
        True if the string matches an IPv4 pattern, False otherwise
    """
    return bool(IPV4_PATTERN.match(name))


def sanitize_collection_name(name: Optional[str]) -> str:
    """
    Sanitize a collection name to meet ChromaDB requirements:
    1. 3-63 characters long
    2. Starts and ends with alphanumeric character
    3. Contains only alphanumeric characters, underscores, or hyphens
    4. No consecutive periods
    5. Not a valid IPv4 address
    
    Args:
        name: The original collection name to sanitize
        
    Returns:
        A sanitized collection name that meets ChromaDB requirements
    """
    if not name:
        return DEFAULT_COLLECTION
    
    # Handle IPv4 pattern
    if is_ipv4_pattern(name):
        name = f"ip_{name}"
    
    # Replace spaces and invalid characters with underscores
    sanitized = INVALID_CHARS_PATTERN.sub('_', name)
    
    # Ensure it starts with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = 'a' + sanitized
    
    # Ensure it ends with alphanumeric
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + 'z'
    
    # Ensure length is between MIN_LENGTH-MAX_LENGTH characters
    if len(sanitized) < MIN_LENGTH:
        # Add padding with alphanumeric character at the end
        sanitized = sanitized + 'x' * (MIN_LENGTH - len(sanitized))
    if len(sanitized) > MAX_LENGTH:
        sanitized = sanitized[:MAX_LENGTH]
        # Ensure it still ends with alphanumeric after truncation
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + 'z'
    
    return sanitized
