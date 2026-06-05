"""Out-of-band binary file passing between tools.

LLMs cannot reproduce opaque strings longer than a few kilobytes byte-perfect.
A base64-encoded binary file (PPTX, PDF, image, ...) returned by one tool and
echoed by the model as the argument to another tool drifts by a few characters,
which invalidates the base64 and corrupts the resulting file.

To avoid routing bytes through the model, a tool returns a :class:`FileArtifact`
instead of a base64 string. The agent executor stores the bytes here and shows
the model a short, stable ``crewai+file://<uuid>`` handle in place of the data.
When the model passes that handle as an argument to a later tool, the executor
expands it back to base64 *just before* the tool runs -- the bytes never enter
the model's context, so they cannot be corrupted.

The handle is namespaced (``crewai+file://``) so resolution only ever fires on
tokens this module minted, never on arbitrary user data. Stored bytes are scoped
to a crew/task execution id and cleared when that execution finishes; a TTL prune
is the safety net for runs that never call :func:`clear_artifact_scope`.

Limitation: handles are ephemeral and scoped to a single run. A handle only
resolves while its run's artifacts are live. If a placeholder's text is persisted
(conversation memory, a checkpoint) and a *later* run echoes that handle, it will
no longer resolve and the literal token is passed through unchanged -- so binary
producer->consumer chains must complete within one run.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Final
from uuid import uuid4


__all__ = [
    "FileArtifact",
    "artifact_scope_id",
    "clear_artifact_scope",
    "resolve_artifact_handles",
    "store_artifact",
    "store_if_artifact",
]

_HANDLE_SCHEME: Final[str] = "crewai+file"
# A minted handle: crewai+file://<uuid4>. Matched case-insensitively because
# uuid hex may arrive upper- or lower-cased after a model round-trip.
_HANDLE_RE: Final[re.Pattern[str]] = re.compile(
    r"crewai\+file://([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

DEFAULT_ARTIFACT_TTL: Final[int] = 3600


@dataclass
class FileArtifact:
    """Binary file produced or consumed by a tool, kept out of the LLM context.

    Return this from a tool's ``_run`` instead of a base64 string when the output
    is binary. The executor stores the bytes and substitutes a short handle in the
    text the model sees, so the model never has to reproduce the data verbatim.

    Attributes:
        data: Raw file bytes.
        filename: Human-readable name, surfaced to the model and useful as a
            default for downstream ``file_name`` arguments.
        mime_type: MIME type of the content.
    """

    data: bytes
    filename: str = "file"
    mime_type: str = "application/octet-stream"

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def as_base64(self) -> str:
        """Return the bytes as an ASCII base64 string (what connectors expect)."""
        return base64.b64encode(self.data).decode("ascii")

    def _placeholder(self, handle: str) -> str:
        """Build the model-facing text that stands in for the bytes."""
        # Neutralize characters that would break the single-line bracketed
        # attribute list (quotes, the closing bracket, newlines).
        filename = _sanitize_attr(self.filename)
        mime_type = _sanitize_attr(self.mime_type)
        return (
            f'[FileArtifact filename="{filename}" '
            f'mime_type="{mime_type}" size={_human_size(self.size_bytes)} '
            f"handle={handle}]\n"
            "The binary content is stored out-of-band to keep it from being "
            "corrupted in transit. To use this file, pass the handle string "
            f"({handle}) as the value of the content/file argument when calling "
            "another tool -- it is expanded to the real data before that tool runs."
        )


@dataclass
class _Entry:
    artifact: FileArtifact
    scope_id: str | None
    expires_at: float | None
    obj_id: int


class _ArtifactStore:
    """Process-local, execution-scoped store keyed by minted handle id.

    Entries are keyed by an opaque uuid (never by user-supplied content), so
    concurrent crews cannot collide. Cleanup is per-scope -- clearing one crew's
    artifacts never touches another's -- with a TTL prune as a backstop.

    Storing the same :class:`FileArtifact` instance again under the same scope
    reuses its handle rather than minting a duplicate. The tool-result cache
    hands back the same object on every cache hit, so this keeps repeated cached
    calls from stacking identical byte copies in memory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        # (id(artifact), scope) -> handle, so re-storing the same instance under
        # the same scope reuses its handle. Keying on the scope too means storing
        # an object under a different scope gets its own handle and its own
        # cleanup entry rather than overwriting the first.
        self._handle_by_obj: dict[tuple[int, str | None], str] = {}

    def store(
        self,
        artifact: FileArtifact,
        scope_id: str | None = None,
        ttl: int = DEFAULT_ARTIFACT_TTL,
    ) -> str:
        norm_scope = str(scope_id) if scope_id is not None else None
        obj_key = (id(artifact), norm_scope)
        expires_at = (time.monotonic() + ttl) if ttl > 0 else None
        with self._lock:
            self._prune_locked()
            existing = self._handle_by_obj.get(obj_key)
            if existing is not None:
                entry = self._entries.get(existing)
                if entry is not None and entry.artifact is artifact:
                    entry.expires_at = expires_at
                    return f"{_HANDLE_SCHEME}://{existing}"
            handle_id = str(uuid4())
            self._entries[handle_id] = _Entry(
                artifact=artifact,
                scope_id=norm_scope,
                expires_at=expires_at,
                obj_id=id(artifact),
            )
            self._handle_by_obj[obj_key] = handle_id
        return f"{_HANDLE_SCHEME}://{handle_id}"

    def resolve(self, handle_id: str) -> FileArtifact | None:
        with self._lock:
            entry = self._entries.get(handle_id)
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= time.monotonic():
                self._delete_locked(handle_id)
                return None
            return entry.artifact

    def clear_scope(self, scope_id: str) -> None:
        scope = str(scope_id)
        with self._lock:
            for handle_id in [
                hid for hid, entry in self._entries.items() if entry.scope_id == scope
            ]:
                self._delete_locked(handle_id)

    def _prune_locked(self) -> None:
        """Drop entries whose per-entry TTL has elapsed. Caller holds the lock."""
        now = time.monotonic()
        for handle_id in [
            hid
            for hid, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]:
            self._delete_locked(handle_id)

    def _delete_locked(self, handle_id: str) -> None:
        """Remove an entry and its object-identity mapping. Caller holds lock."""
        entry = self._entries.pop(handle_id, None)
        if entry is not None:
            self._handle_by_obj.pop((entry.obj_id, entry.scope_id), None)


_store: Final[_ArtifactStore] = _ArtifactStore()


def store_artifact(
    artifact: FileArtifact,
    scope_id: Any | None = None,
    ttl: int = DEFAULT_ARTIFACT_TTL,
) -> str:
    """Store a :class:`FileArtifact` and return its model-facing placeholder text.

    Args:
        artifact: The binary artifact to keep out of the model context.
        scope_id: Execution id (crew or task) used to group the artifact for
            cleanup. ``None`` means it is only reclaimed by the TTL prune.
        ttl: Seconds after which an unreferenced artifact may be pruned.

    Returns:
        The placeholder string to surface to the model in place of the bytes.
    """
    handle = _store.store(artifact, scope_id=scope_id, ttl=ttl)
    return artifact._placeholder(handle)


def resolve_artifact_handles(value: Any) -> Any:
    """Recursively replace stored handles in tool arguments with base64 data.

    Walks strings, dicts, and lists. Any ``crewai+file://<uuid>`` token that
    resolves to a stored artifact is replaced with that artifact's base64 string;
    unknown tokens and all other values are returned unchanged. A new container is
    returned so the caller's original arguments (used for events, caching, and
    logs) keep the short handle.
    """
    if isinstance(value, str):
        if _HANDLE_SCHEME not in value:
            return value

        def _sub(match: re.Match[str]) -> str:
            # Store keys are lowercase uuid4 strings; the regex matches hex
            # case-insensitively, so normalize before lookup in case the model
            # echoed the handle with uppercase hex.
            artifact = _store.resolve(match.group(1).lower())
            return artifact.as_base64() if artifact is not None else match.group(0)

        return _HANDLE_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {key: resolve_artifact_handles(val) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_artifact_handles(item) for item in value]
    return value


def store_if_artifact(result: Any, scope_id: Any | None = None) -> Any:
    """Store ``result`` and return its placeholder if it is a :class:`FileArtifact`.

    Any other value is returned unchanged. This is the single funnel both the
    native and ReAct executor paths route tool output through, so fresh and
    cached results are handled identically.
    """
    if isinstance(result, FileArtifact):
        return store_artifact(result, scope_id=scope_id)
    return result


def clear_artifact_scope(scope_id: Any) -> None:
    """Drop every artifact stored under ``scope_id`` (called when a run ends)."""
    _store.clear_scope(scope_id)


def artifact_scope_id(
    crew: Any | None = None,
    task: Any | None = None,
    agent: Any | None = None,
) -> Any | None:
    """Pick the execution id used to scope a tool's file artifacts for cleanup.

    Prefer the crew id -- it matches the id ``Crew`` passes to
    :func:`clear_artifact_scope` when a run ends -- falling back to the agent's
    crew, then the task id, then ``None`` (TTL-only cleanup). Centralized, and
    given the agent fallback, so every tool-execution path derives the scope the
    same way and can't drift.
    """
    if crew is None:
        crew = getattr(agent, "crew", None)
    crew_id = getattr(crew, "id", None)
    if crew_id is not None:
        return crew_id
    return getattr(task, "id", None)


def _sanitize_attr(text: str) -> str:
    """Strip characters that would break the bracketed placeholder display."""
    return (
        text.replace('"', "'").replace("]", ")").replace("\n", " ").replace("\r", " ")
    )


def _human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1024 or unit == "PB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
