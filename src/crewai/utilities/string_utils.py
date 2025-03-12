import re
from typing import Optional


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
        return "default_collection"
    
    # Replace spaces and invalid characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Ensure it starts with alphanumeric
    if not sanitized[0].isalnum():
        sanitized = 'a' + sanitized
    
    # Ensure it ends with alphanumeric
    if not sanitized[-1].isalnum():
        sanitized = sanitized[:-1] + 'z'
    
    # Ensure length is between 3-63 characters
    if len(sanitized) < 3:
        # Add padding with alphanumeric character at the end
        sanitized = sanitized + 'x' * (3 - len(sanitized))
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        # Ensure it still ends with alphanumeric after truncation
        if not sanitized[-1].isalnum():
            sanitized = sanitized[:-1] + 'z'
    
    return sanitized
