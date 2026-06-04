"""Sideband file reference system for binary data between tool calls.

Binary file data (e.g. base64-encoded content) that exceeds a size threshold
is stored here instead of passing through the LLM context window, preventing
token-by-token corruption.
"""

from __future__ import annotations

import base64
import re
import threading
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ToolFileReference:
    """A reference to binary data stored in the sideband file store."""

    ref_id: str = field(default_factory=lambda: str(uuid4()))
    filename: str = ""
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    data: bytes = b""

    def placeholder(self) -> str:
        return f"[File: {self.filename}, {_human_size(self.size_bytes)}, ref={self.ref_id}]"


class ToolFileStore:
    """Thread-safe in-memory store for binary data between tool calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, ToolFileReference] = {}

    def store(
        self,
        data: bytes,
        filename: str = "file",
        content_type: str = "application/octet-stream",
    ) -> ToolFileReference:
        ref = ToolFileReference(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            data=data,
        )
        with self._lock:
            self._store[ref.ref_id] = ref
        return ref

    def resolve(self, ref_id: str) -> bytes:
        with self._lock:
            ref = self._store.get(ref_id)
        if ref is None:
            raise KeyError(f"File reference not found: {ref_id}")
        return ref.data

    def resolve_reference(self, ref_id: str) -> ToolFileReference:
        with self._lock:
            ref = self._store.get(ref_id)
        if ref is None:
            raise KeyError(f"File reference not found: {ref_id}")
        return ref

    def has(self, ref_id: str) -> bool:
        with self._lock:
            return ref_id in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


tool_file_store = ToolFileStore()

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\n\r]+=*$")
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def is_large_base64(text: str, threshold: int = 4096) -> bool:
    """Check if text looks like base64-encoded data above threshold bytes."""
    stripped = text.strip()
    if len(stripped) < threshold:
        return False
    if not _BASE64_RE.match(stripped):
        return False
    try:
        decoded = base64.b64decode(stripped, validate=True)
        return len(decoded) >= threshold
    except Exception:
        return False


def auto_store_if_binary(result: object, threshold: int = 4096) -> object:
    """If result is a large base64 string, store it and return a ToolFileReference."""
    if isinstance(result, ToolFileReference):
        return result
    if isinstance(result, str) and is_large_base64(result, threshold):
        data = base64.b64decode(result.strip())
        return tool_file_store.store(data, filename="tool_output.bin")
    return result


def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
