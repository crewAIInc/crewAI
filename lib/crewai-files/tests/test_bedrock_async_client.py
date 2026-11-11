"""Tests for BedrockFileUploader's async session handling."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from crewai_files import FileBytes, ImageFile
from crewai_files.uploaders.bedrock import BedrockFileUploader
import pytest


MINIMAL_PNG = b"\x89PNG\r\n\x1a\n"  # magic header is all content-type detection needs


class _StubS3Client:
    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def upload_fileobj(
        self, fileobj: Any, bucket: str, key: str, **kwargs: Any
    ) -> None:
        self.uploaded.append((bucket, key))

    async def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self.deleted.append((Bucket, Key))


class _StubClientContext:
    def __init__(self, client: _StubS3Client) -> None:
        self._client = client

    async def __aenter__(self) -> _StubS3Client:
        return self._client

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubSession:
    """Stands in for an aioboto3 Session."""

    def __init__(self) -> None:
        self.s3 = _StubS3Client()
        self.requested: list[tuple[str, dict[str, Any]]] = []

    def client(self, service_name: str, **kwargs: Any) -> _StubClientContext:
        self.requested.append((service_name, kwargs))
        return _StubClientContext(self.s3)


def _image_file() -> ImageFile:
    return ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))


@pytest.fixture(autouse=True)
def _no_boto3_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests hermetic: boto3 is not a runtime dependency of the
    crewai-files test environment, so stub out the TransferConfig lookup."""
    monkeypatch.setattr(
        BedrockFileUploader, "_get_transfer_config", staticmethod(lambda: None)
    )


@pytest.mark.asyncio
async def test_aupload_uses_injected_async_session() -> None:
    session = _StubSession()
    uploader = BedrockFileUploader(bucket_name="test-bucket", async_client=session)

    result = await uploader.aupload(_image_file())

    assert result.provider == "bedrock"
    assert result.file_uri == f"s3://test-bucket/{result.file_id}"
    assert session.s3.uploaded == [("test-bucket", result.file_id)]
    assert session.requested[0][0] == "s3"


@pytest.mark.asyncio
async def test_adelete_uses_injected_async_session() -> None:
    session = _StubSession()
    uploader = BedrockFileUploader(bucket_name="test-bucket", async_client=session)

    assert await uploader.adelete("some/key.png") is True
    assert session.s3.deleted == [("test-bucket", "some/key.png")]


@pytest.mark.asyncio
async def test_aupload_creates_session_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_StubSession] = []
    fake_module = ModuleType("aioboto3")
    fake_module.Session = lambda: created.append(_StubSession()) or created[-1]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aioboto3", fake_module)

    uploader = BedrockFileUploader(bucket_name="test-bucket")

    result = await uploader.aupload(_image_file())

    assert len(created) == 1
    assert created[0].s3.uploaded == [("test-bucket", result.file_id)]
