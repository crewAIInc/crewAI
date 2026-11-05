"""cleanup_expired_files must delete expired uploads from the provider when asked."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import crewai_files.cache.cleanup as cleanup_module
from crewai_files.cache.cleanup import acleanup_expired_files, cleanup_expired_files
from crewai_files.cache.upload_cache import UploadCache


class RecordingUploader:
    """Uploader that records which file ids were deleted."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, file_id: str) -> bool:
        self.deleted.append(file_id)
        return True

    async def adelete(self, file_id: str) -> bool:
        self.deleted.append(file_id)
        return True

    def list_files(self) -> list[dict[str, Any]]:
        return []


def _cache_with_one_expired_and_one_live() -> UploadCache:
    cache = UploadCache()
    now = datetime.now(timezone.utc)
    cache.set_by_hash(
        file_hash="expired-hash",
        content_type="image/png",
        provider="gemini",
        file_id="expired-file",
        expires_at=now - timedelta(hours=1),
    )
    cache.set_by_hash(
        file_hash="live-hash",
        content_type="image/png",
        provider="gemini",
        file_id="live-file",
        expires_at=now + timedelta(hours=1),
    )
    return cache


def test_cleanup_expired_files_deletes_expired_from_provider(monkeypatch):
    uploader = RecordingUploader()
    monkeypatch.setattr(cleanup_module, "_uploader_or_none", lambda provider: uploader)
    cache = _cache_with_one_expired_and_one_live()

    removed = cleanup_expired_files(cache, delete_from_provider=True)

    assert removed == 1
    assert uploader.deleted == ["expired-file"]


def test_cleanup_expired_files_keeps_provider_files_by_default(monkeypatch):
    uploader = RecordingUploader()
    monkeypatch.setattr(cleanup_module, "_uploader_or_none", lambda provider: uploader)
    cache = _cache_with_one_expired_and_one_live()

    removed = cleanup_expired_files(cache)

    assert removed == 1
    assert uploader.deleted == []


def test_acleanup_expired_files_deletes_expired_from_provider(monkeypatch):
    uploader = RecordingUploader()
    monkeypatch.setattr(cleanup_module, "_uploader_or_none", lambda provider: uploader)

    async def scenario() -> int:
        cache = UploadCache()
        now = datetime.now(timezone.utc)
        await cache.aset_by_hash(
            file_hash="expired-hash",
            content_type="image/png",
            provider="gemini",
            file_id="expired-file",
            expires_at=now - timedelta(hours=1),
        )
        await cache.aset_by_hash(
            file_hash="live-hash",
            content_type="image/png",
            provider="gemini",
            file_id="live-file",
            expires_at=now + timedelta(hours=1),
        )
        return await acleanup_expired_files(cache, delete_from_provider=True)

    removed = asyncio.run(scenario())

    assert removed == 1
    assert uploader.deleted == ["expired-file"]
