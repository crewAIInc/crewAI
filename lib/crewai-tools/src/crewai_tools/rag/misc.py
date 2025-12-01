import hashlib
from typing import Any


def compute_sha256(content: str) -> str:
    """Compute the SHA-256 hash of the given content.

    Args:
      content: The content to hash.

    Returns:
      The SHA-256 hash of the content as a hexadecimal string.
    """
    return hashlib.sha256(content.encode()).hexdigest()


def sanitize_metadata_for_chromadb(metadata: dict[str, Any]) -> dict[str, Any]:
    """Sanitize metadata to ensure ChromaDB compatibility.

    ChromaDB only accepts str, int, float, or bool values in metadata.
    This function converts other types to strings.

    Args:
        metadata: Dictionary of metadata to sanitize

    Returns:
        Sanitized metadata dictionary with only ChromaDB-compatible types
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to pipe-separated strings
            sanitized[key] = " | ".join(str(v) for v in value)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    return sanitized
