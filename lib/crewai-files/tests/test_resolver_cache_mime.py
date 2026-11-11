"""Uploaded-file cache hits must respect the current file's MIME type."""

import asyncio

from crewai_files.cache.upload_cache import UploadCache
from crewai_files.core.resolved import FileReference
import crewai_files.core.sources as sources
from crewai_files.core.sources import FileBytes
from crewai_files.core.types import File, FileInput
from crewai_files.resolution.resolver import FileResolver, FileResolverConfig
from crewai_files.uploaders.base import FileUploader, UploadResult
import pytest


class RecordingUploader(FileUploader):
    """Return a distinct provider ID for each actual upload."""

    def __init__(self) -> None:
        self.uploaded_types: list[str] = []

    @property
    def provider_name(self) -> str:
        return "openai"

    def upload(self, file: FileInput, purpose: str | None = None) -> UploadResult:
        self.uploaded_types.append(file.content_type)
        return UploadResult(
            file_id=f"upload-{len(self.uploaded_types)}",
            provider=self.provider_name,
            content_type=file.content_type,
        )

    def delete(self, file_id: str) -> bool:
        return True


def _setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FileResolver, RecordingUploader, File, File]:
    # Optional magic libraries can identify these bytes as generic text. Force
    # the two valid filename-based MIME interpretations for a portable test.
    monkeypatch.setattr(
        sources,
        "detect_content_type",
        lambda _data, filename: (
            "text/csv" if filename == "report.csv" else "text/markdown"
        ),
    )
    data = b"col1,col2\n1,2\n"
    csv_file = File(source=FileBytes(data=data, filename="report.csv"))
    md_file = File(source=FileBytes(data=data, filename="report.md"))
    uploader = RecordingUploader()
    resolver = FileResolver(
        config=FileResolverConfig(prefer_upload=True), upload_cache=UploadCache()
    )
    monkeypatch.setattr(resolver, "_get_uploader", lambda _provider: uploader)
    return resolver, uploader, csv_file, md_file


def test_resolve_reuploads_same_bytes_with_different_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, uploader, csv_file, md_file = _setup(monkeypatch)

    csv_ref = resolver.resolve(csv_file, "openai")
    md_ref = resolver.resolve(md_file, "openai")
    md_ref_again = resolver.resolve(md_file, "openai")

    assert isinstance(csv_ref, FileReference)
    assert isinstance(md_ref, FileReference)
    assert isinstance(md_ref_again, FileReference)
    assert csv_ref.file_id == "upload-1"
    assert md_ref.file_id == md_ref_again.file_id == "upload-2"
    assert md_ref.content_type == "text/markdown"
    assert uploader.uploaded_types == ["text/csv", "text/markdown"]


def test_aresolve_reuploads_same_bytes_with_different_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, uploader, csv_file, md_file = _setup(monkeypatch)

    async def scenario() -> None:
        csv_ref = await resolver.aresolve(csv_file, "openai")
        md_ref = await resolver.aresolve(md_file, "openai")
        md_ref_again = await resolver.aresolve(md_file, "openai")

        assert isinstance(csv_ref, FileReference)
        assert isinstance(md_ref, FileReference)
        assert isinstance(md_ref_again, FileReference)
        assert csv_ref.file_id == "upload-1"
        assert md_ref.file_id == md_ref_again.file_id == "upload-2"
        assert md_ref.content_type == "text/markdown"

    asyncio.run(scenario())
    assert uploader.uploaded_types == ["text/csv", "text/markdown"]
