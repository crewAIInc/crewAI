"""
Artifacts: named, versioned files a run produces (reports, charts, PDFs).
Saving with the same identifier in the same execution creates a new version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.artifact_events import ArtifactCreatedEvent
from crewai.execution import get_execution_uuid
from crewai.tools.base_tool import BaseTool


ArtifactKind = Literal[
    "markdown", "html", "svg", "json", "a2ui", "image", "pdf", "text", "csv", "file"
]

# kind -> (extension, content type). None means "take it from the file name".
_KIND_FORMATS: dict[str, tuple[str | None, str | None]] = {
    "markdown": ("md", "text/markdown"),
    "html": ("html", "text/html"),
    "svg": ("svg", "image/svg+xml"),
    "json": ("json", "application/json"),
    "a2ui": ("json", "application/json"),
    "image": (None, None),
    "pdf": ("pdf", "application/pdf"),
    "text": ("txt", "text/plain"),
    "csv": ("csv", "text/csv"),
    "file": (None, "application/octet-stream"),
}

# Kinds whose bytes come from a file, so text output can never fill them.
FILE_KINDS = frozenset({"image", "pdf", "file"})


def guess_kind(content: str) -> ArtifactKind:
    """Pick a text kind from what the output looks like. Defaults to markdown."""
    head = content.lstrip()[:200].lower()
    if head.startswith(("<!doctype html", "<html")):
        return "html"
    if head.startswith("<svg") or (head.startswith("<?xml") and "<svg" in head):
        return "svg"
    if head.startswith(("{", "[")):
        try:
            json.loads(content)
            return "json"
        except ValueError:
            pass
    return "markdown"


_OUTER_FENCE = re.compile(r"\s*```([\w+-]*)[^\n]*\n(.*?)\n?```\s*", re.DOTALL)

_FENCE_KINDS: dict[str, ArtifactKind] = {
    "html": "html",
    "svg": "svg",
    "json": "json",
    "csv": "csv",
    "markdown": "markdown",
    "md": "markdown",
    "text": "text",
    "txt": "text",
}


def unwrap_fence(content: str) -> tuple[str, str | None]:
    """Strip one code fence wrapping the whole reply; return (body, language).

    Content that only contains code blocks, like a markdown doc, is unchanged.
    """
    match = _OUTER_FENCE.fullmatch(content)
    if match is None or "```" in match.group(2):
        return content, None
    return match.group(2), match.group(1).lower() or None


def resolve_artifact(
    artifact: Artifact | bool,
    name: str,
    content: str,
    language: str | None = None,
) -> Artifact:
    """Turn the ``artifact=True`` shorthand into a full :class:`Artifact`.

    ``name`` must be stable across reruns (a task name, a flow method name,
    an agent role), so saving again makes a new version, not a new artifact.
    """
    if isinstance(artifact, Artifact):
        return artifact
    identifier = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.")[:64]
    title = re.sub(r"[_-]+", " ", name).strip()
    return Artifact(
        identifier=identifier.rstrip("-.") or "output",
        title=(title[:1].upper() + title[1:]) or "Output",
        kind=_FENCE_KINDS.get(language or "") or guess_kind(content),
    )


def check_artifact_source(artifact: Artifact) -> Artifact:
    """Reject file kinds with no file_path: text output can never fill them."""
    if artifact.kind in FILE_KINDS and not artifact.file_path:
        raise ValueError(
            f"artifact kind '{artifact.kind}' needs a file: set file_path to "
            "where the agent writes it."
        )
    return artifact


def save_output_artifact(
    artifact: Artifact | bool,
    *,
    content: str,
    name: str,
    from_agent: Any = None,
    from_task: Any = None,
) -> SavedArtifact:
    """Save a run's output as ``artifact``: its ``file_path`` if set, else ``content``."""
    content, language = unwrap_fence(content)
    resolved = resolve_artifact(artifact, name, content, language)
    if resolved.file_path is None:
        return save_artifact(
            resolved, content=content, from_agent=from_agent, from_task=from_task
        )
    return save_artifact(
        resolved,
        file_path=resolved.file_path,
        from_agent=from_agent,
        from_task=from_task,
    )


IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"


class Artifact(BaseModel):
    """What to save: identity and kind of an artifact."""

    identifier: str = Field(
        pattern=IDENTIFIER_PATTERN,
        description="Stable id within the execution. Saving it again creates a new version.",
    )
    title: str = Field(description="Human-readable title.")
    kind: ArtifactKind = Field(description="How the artifact is rendered.")
    file_path: str | None = Field(
        default=None,
        description=(
            "Save the file the agent writes here instead of its text output. "
            "Needed for image, pdf and file."
        ),
    )


class SavedArtifact(Artifact):
    """One stored version of an artifact."""

    id: str = Field(description="UUID of this version, made in the automation.")
    version: int
    content_type: str
    size: int
    url: str
    share_url: str | None = None


class ArtifactNotFoundError(LookupError):
    """No saved artifact matches the identifier (and version) asked for."""


class ArtifactStore(ABC):
    """Where artifact bytes go. Swap with :func:`set_artifact_store`."""

    @abstractmethod
    def save(
        self,
        artifact_id: str,
        execution_id: str,
        artifact: Artifact,
        data: bytes,
        extension: str,
        content_type: str,
    ) -> SavedArtifact:
        """Store ``data`` as the next version of ``artifact``.

        ``artifact_id`` is made once per save, before any upload. A store that
        retries must send the same ``artifact_id`` so the server can drop the
        duplicate instead of creating another version.
        """

    def load(
        self, execution_id: str, identifier: str, version: int | None = None
    ) -> bytes:
        """Return a saved version's bytes: ``version``, or the latest one.

        Lets a run read back what it saved, such as a flow resumed after human
        feedback. Raise :class:`ArtifactNotFoundError` when nothing matches.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot load artifacts")


class LocalArtifactStore(ArtifactStore):
    """Writes artifacts to disk under ``root``."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def save(
        self,
        artifact_id: str,
        execution_id: str,
        artifact: Artifact,
        data: bytes,
        extension: str,
        content_type: str,
    ) -> SavedArtifact:
        directory = self.root / execution_id / artifact.identifier
        directory.mkdir(parents=True, exist_ok=True)
        existing_versions = [
            int(path.stem[1:])
            for path in directory.glob("v*.*")
            if path.is_file() and path.stem[1:].isdigit()
        ]
        version = max(existing_versions, default=0) + 1
        while True:
            while any(directory.glob(f"v{version}.*")):
                version += 1
            path = directory / f"v{version}.{extension}"
            try:
                with path.open("xb") as file:
                    file.write(data)
                break
            except FileExistsError:
                version += 1
        return SavedArtifact(
            **artifact.model_dump(),
            id=artifact_id,
            version=version,
            content_type=content_type,
            size=len(data),
            url=path.resolve().as_uri(),
        )

    def load(
        self, execution_id: str, identifier: str, version: int | None = None
    ) -> bytes:
        directory = self.root / execution_id / identifier
        versions = {
            int(path.stem[1:]): path
            for path in directory.glob("v*.*")
            if path.stem[1:].isdigit()
        }
        number = max(versions, default=None) if version is None else version
        if number not in versions:
            raise ArtifactNotFoundError(
                f"no artifact {identifier!r} version {version or 'latest'} in this run"
            )
        return versions[number].read_bytes()


_store: ArtifactStore = LocalArtifactStore()


def set_artifact_store(store: ArtifactStore) -> None:
    """Replace the process-wide artifact store."""
    global _store
    _store = store


def get_artifact_store() -> ArtifactStore:
    """Return the process-wide artifact store."""
    return _store


def _check_readable(file_path: str) -> Path:
    """Resolve ``file_path`` and refuse anything outside the working directory."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_relative_to(Path.cwd().resolve()):
        raise ValueError(f"file_path must be inside the working directory: {file_path}")
    if not path.is_file():
        raise ValueError(f"file_path is not a file: {file_path}")
    return path


# Leading bytes of the image formats AMP accepts.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
    (b"GIF8", "gif", "image/gif"),
)


def _image_format(data: bytes) -> tuple[str, str]:
    """Return (extension, content type) for image bytes, or raise if unknown."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    for signature, extension, content_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return extension, content_type
    raise ValueError("image bytes must be PNG, JPEG, GIF or WebP")


def save_artifact(
    artifact: Artifact,
    content: str | bytes | None = None,
    file_path: str | None = None,
    *,
    from_agent: Any = None,
    from_task: Any = None,
) -> SavedArtifact:
    """Save ``content`` or the file at ``file_path`` as a new artifact version.

    Emits :class:`ArtifactCreatedEvent` on success. ``from_agent`` and
    ``from_task`` name who saved it, so traces can attribute the artifact.
    """
    if content is not None and file_path is not None:
        raise ValueError("Pass exactly one of content or file_path.")

    extension, content_type = _KIND_FORMATS[artifact.kind]
    if file_path is not None:
        path = _check_readable(file_path)
        data = path.read_bytes()
        extension = extension or path.suffix.lstrip(".") or "bin"
        content_type = (
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
    elif content is not None:
        # Text can never be an image, PDF or file; bytes from code can.
        if artifact.kind in FILE_KINDS and isinstance(content, str):
            raise ValueError(f"kind '{artifact.kind}' needs file_path, not content.")
        data = content.encode("utf-8") if isinstance(content, str) else content
        if artifact.kind == "image":
            extension, content_type = _image_format(data)
        elif artifact.kind == "file":
            extension = "bin"
    else:
        raise ValueError("Pass exactly one of content or file_path.")
    if extension is None:
        raise ValueError(f"kind '{artifact.kind}' needs file_path, not content.")

    saved = get_artifact_store().save(
        str(uuid.uuid4()),
        get_execution_uuid() or "local",
        artifact,
        data,
        extension,
        content_type or "application/octet-stream",
    )
    crewai_event_bus.emit(
        from_task or from_agent,
        ArtifactCreatedEvent(
            from_agent=from_agent,
            from_task=from_task,
            id=saved.id,
            identifier=saved.identifier,
            version=saved.version,
            title=saved.title,
            kind=saved.kind,
            content_type=saved.content_type,
            size=saved.size,
            url=saved.url,
        ),
    )
    return saved


def load_artifact(identifier: str, version: int | None = None) -> bytes:
    """Read back an artifact this run saved: ``version``, or the latest one."""
    if re.match(IDENTIFIER_PATTERN, identifier) is None:
        raise ValueError(f"invalid artifact identifier: {identifier!r}")
    return get_artifact_store().load(
        get_execution_uuid() or "local", identifier, version
    )


class ArtifactReference(BaseModel):
    """Points at an artifact this run saved, so a step can take it as an input file."""

    model_config = ConfigDict(extra="forbid")

    artifact: str = Field(
        pattern=IDENTIFIER_PATTERN, description="Artifact identifier."
    )
    version: int | None = Field(
        default=None, ge=1, description="Version to read; the latest when omitted."
    )

    def load_file(self) -> Any:
        """Load the artifact as a ``crewai_files`` File for an agent's input_files."""
        try:
            from crewai_files import File, FileBytes
        except ImportError as e:
            raise ImportError(
                "Artifact input files need crewai-files: uv add 'crewai[file-processing]'"
            ) from e
        data = load_artifact(self.artifact, self.version)
        return File(source=FileBytes(data=data, filename=self.artifact))


class SaveArtifactSchema(Artifact):
    """Arguments for :class:`SaveArtifactTool`."""

    content: str | None = Field(
        default=None, description="Text content (HTML, markdown, JSON, CSV, ...)."
    )
    file_path: str | None = Field(
        default=None,
        description="Path to a file you created (a chart PNG, a PDF). Use instead of content.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> SaveArtifactSchema:
        if (self.content is None) == (self.file_path is None):
            raise ValueError("Pass exactly one of content or file_path.")
        return self


class SaveArtifactTool(BaseTool):
    """Lets an agent save a file the user can view, download and share."""

    name: str = "save_artifact"
    description: str = (
        "Save a file the user can view, download and share, such as an HTML "
        "report, a markdown doc, a chart image, a PDF or a CSV. Saving the same "
        "identifier again creates a new version. Returns the artifact url, which "
        "you can send onward (for example in a message)."
    )
    args_schema: type[BaseModel] = SaveArtifactSchema
    agent: Any = Field(default=None, exclude=True)
    task: Any = Field(default=None, exclude=True)

    def _run(
        self,
        identifier: str,
        title: str,
        kind: str,
        content: str | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            saved = save_artifact(
                Artifact(identifier=identifier, title=title, kind=kind),
                content=content,
                file_path=file_path,
                from_agent=self.agent,
                from_task=self.task,
            )
        except Exception as e:
            return f"Error saving artifact: {e}"
        return json.dumps(
            saved.model_dump(
                include={"id", "identifier", "version", "url", "share_url"}
            )
        )
