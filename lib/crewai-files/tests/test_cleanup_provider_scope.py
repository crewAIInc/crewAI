"""Cleanup only drops cache entries of the providers it was asked to clean."""

import asyncio

import crewai_files.cache.cleanup as cleanup_module
from crewai_files.cache.cleanup import acleanup_uploaded_files, cleanup_uploaded_files
from crewai_files.cache.upload_cache import UploadCache


class RecordingUploader:
    """Deletes everything and remembers what it was asked to delete."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, file_id: str) -> bool:
        self.deleted.append(file_id)
        return True

    async def adelete(self, file_id: str) -> bool:
        self.deleted.append(file_id)
        return True


def _seed(cache: UploadCache) -> None:
    asyncio.run(cache.aset_by_hash("hash-openai", "text/plain", "openai", "file-a"))
    asyncio.run(
        cache.aset_by_hash("hash-anthropic", "text/plain", "anthropic", "file-b")
    )


def _register(monkeypatch, uploaders: dict[str, RecordingUploader]) -> None:
    monkeypatch.setattr(
        cleanup_module,
        "_uploader_or_none",
        lambda provider: uploaders.get(provider),
    )


class TestCleanupProviderScope:
    """Tests for the `providers` argument of the cleanup helpers."""

    def test_cleanup_keeps_other_providers_cache_entries(self, monkeypatch):
        uploaders = {"openai": RecordingUploader(), "anthropic": RecordingUploader()}
        _register(monkeypatch, uploaders)

        cache = UploadCache()
        _seed(cache)

        cleaned = cleanup_uploaded_files(
            cache, delete_from_provider=True, providers=["openai"]
        )

        assert cleaned == 1
        assert uploaders["openai"].deleted == ["file-a"]
        assert uploaders["anthropic"].deleted == []
        assert cache.get_by_hash("hash-anthropic", "anthropic") is not None
        assert cache.get_by_hash("hash-openai", "openai") is None

    def test_cleanup_without_provider_filter_clears_everything(self, monkeypatch):
        uploaders = {"openai": RecordingUploader(), "anthropic": RecordingUploader()}
        _register(monkeypatch, uploaders)

        cache = UploadCache()
        _seed(cache)

        cleaned = cleanup_uploaded_files(cache, delete_from_provider=True)

        assert cleaned == 2
        assert len(cache) == 0

    def test_cleanup_keeps_cache_entries_when_provider_cannot_be_deleted(
        self, monkeypatch
    ):
        # No uploader for openai: the file is still on the provider, so dropping
        # the cache entry would orphan it.
        _register(monkeypatch, {})

        cache = UploadCache()
        _seed(cache)

        cleaned = cleanup_uploaded_files(cache, providers=["openai"])

        assert cleaned == 0
        assert cache.get_by_hash("hash-openai", "openai") is not None
        assert cache.get_by_hash("hash-anthropic", "anthropic") is not None

    def test_acleanup_keeps_other_providers_cache_entries(self, monkeypatch):
        uploaders = {"openai": RecordingUploader(), "anthropic": RecordingUploader()}
        _register(monkeypatch, uploaders)

        cache = UploadCache()
        _seed(cache)

        async def scenario() -> int:
            return await acleanup_uploaded_files(
                cache, delete_from_provider=True, providers=["openai"]
            )

        cleaned = asyncio.run(scenario())

        assert cleaned == 1
        assert uploaders["anthropic"].deleted == []
        assert cache.get_by_hash("hash-anthropic", "anthropic") is not None
        assert cache.get_by_hash("hash-openai", "openai") is None

    def test_acleanup_keeps_cache_entries_when_provider_cannot_be_deleted(
        self, monkeypatch
    ):
        _register(monkeypatch, {})

        cache = UploadCache()
        _seed(cache)

        async def scenario() -> int:
            return await acleanup_uploaded_files(cache, providers=["openai"])

        cleaned = asyncio.run(scenario())

        assert cleaned == 0
        assert cache.get_by_hash("hash-openai", "openai") is not None
