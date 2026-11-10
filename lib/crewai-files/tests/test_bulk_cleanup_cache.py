"""Bulk provider cleanup must not reuse IDs deleted from the provider."""

import asyncio

import crewai_files.cache.cleanup as cleanup_module
from crewai_files.cache.cleanup import (
    acleanup_provider_files,
    cleanup_provider_files,
)
from crewai_files.cache.upload_cache import UploadCache
from crewai_files.uploaders.factory import ProviderType
import pytest


class FakeUploader:
    """Provider double with success, refusal, and error outcomes."""

    def list_files(self) -> list[dict[str, str]]:
        return [
            {"id": "deleted-id"},
            {"id": "retained-id"},
            {"id": "raised-id"},
        ]

    def delete(self, file_id: str) -> bool:
        if file_id == "raised-id":
            raise RuntimeError("provider deletion failed")
        return file_id == "deleted-id"

    async def adelete(self, file_id: str) -> bool:
        if file_id == "raised-id":
            raise RuntimeError("provider deletion failed")
        return file_id == "deleted-id"


class SuccessfulUploader:
    """Provider double that deletes every listed file."""

    def __init__(self) -> None:
        self.deleted_ids: list[str] = []

    def list_files(self) -> list[dict[str, str]]:
        return [{"id": "first-id"}, {"id": "second-id"}]

    def delete(self, file_id: str) -> bool:
        self.deleted_ids.append(file_id)
        return True

    async def adelete(self, file_id: str) -> bool:
        self.deleted_ids.append(file_id)
        return True


def test_bulk_cleanup_removes_only_successfully_deleted_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = UploadCache()
    cache.set_by_hash("deleted-hash", "text/plain", "openai", "deleted-id")
    cache.set_by_hash("duplicate-hash", "text/plain", "openai", "deleted-id")
    cache.set_by_hash("retained-hash", "text/plain", "openai", "retained-id")
    cache.set_by_hash("raised-hash", "text/plain", "openai", "raised-id")
    monkeypatch.setattr(
        cleanup_module, "_uploader_or_none", lambda _provider: FakeUploader()
    )

    deleted = cleanup_provider_files(
        "openai", cache=cache, delete_all_from_provider=True
    )

    assert deleted == 1
    assert cache.get_by_hash("deleted-hash", "openai") is None
    assert cache.get_by_hash("duplicate-hash", "openai") is None
    retained = cache.get_by_hash("retained-hash", "openai")
    raised = cache.get_by_hash("raised-hash", "openai")
    assert retained is not None
    assert retained.file_id == "retained-id"
    assert raised is not None
    assert raised.file_id == "raised-id"


def test_async_bulk_cleanup_removes_only_successfully_deleted_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = UploadCache()
    monkeypatch.setattr(
        cleanup_module, "_uploader_or_none", lambda _provider: FakeUploader()
    )

    async def scenario() -> None:
        await cache.aset_by_hash("deleted-hash", "text/plain", "openai", "deleted-id")
        await cache.aset_by_hash("duplicate-hash", "text/plain", "openai", "deleted-id")
        await cache.aset_by_hash("retained-hash", "text/plain", "openai", "retained-id")
        await cache.aset_by_hash("raised-hash", "text/plain", "openai", "raised-id")
        deleted = await acleanup_provider_files(
            "openai", cache=cache, delete_all_from_provider=True
        )
        removed_entry = await cache.aget_by_hash("deleted-hash", "openai")
        duplicate_entry = await cache.aget_by_hash("duplicate-hash", "openai")
        retained_entry = await cache.aget_by_hash("retained-hash", "openai")
        raised_entry = await cache.aget_by_hash("raised-hash", "openai")
        assert deleted == 1
        assert removed_entry is None
        assert duplicate_entry is None
        assert retained_entry is not None
        assert retained_entry.file_id == "retained-id"
        assert raised_entry is not None
        assert raised_entry.file_id == "raised-id"

    asyncio.run(scenario())


def test_bulk_cleanup_keeps_deleting_when_cache_removal_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = UploadCache()
    cache.set_by_hash("first-hash", "text/plain", "openai", "first-id")
    cache.set_by_hash("second-hash", "text/plain", "openai", "second-id")
    uploader = SuccessfulUploader()
    monkeypatch.setattr(cleanup_module, "_uploader_or_none", lambda _provider: uploader)
    original_remove = cache.remove_by_file_id

    def remove_with_failure(file_id: str, provider: ProviderType) -> bool:
        if file_id == "first-id":
            raise RuntimeError("cache unavailable")
        return original_remove(file_id, provider)

    monkeypatch.setattr(cache, "remove_by_file_id", remove_with_failure)

    deleted = cleanup_provider_files(
        "openai", cache=cache, delete_all_from_provider=True
    )

    assert deleted == 2
    assert uploader.deleted_ids == ["first-id", "second-id"]
    assert cache.get_by_hash("first-hash", "openai") is not None
    assert cache.get_by_hash("second-hash", "openai") is None


def test_async_bulk_cleanup_counts_all_deletions_when_cache_removal_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = UploadCache()
    uploader = SuccessfulUploader()
    monkeypatch.setattr(cleanup_module, "_uploader_or_none", lambda _provider: uploader)
    original_remove = cache.aremove_by_file_id

    async def remove_with_failure(file_id: str, provider: ProviderType) -> bool:
        if file_id == "first-id":
            raise RuntimeError("cache unavailable")
        return await original_remove(file_id, provider)

    monkeypatch.setattr(cache, "aremove_by_file_id", remove_with_failure)

    async def scenario() -> None:
        await cache.aset_by_hash("first-hash", "text/plain", "openai", "first-id")
        await cache.aset_by_hash("second-hash", "text/plain", "openai", "second-id")

        deleted = await acleanup_provider_files(
            "openai", cache=cache, delete_all_from_provider=True
        )

        assert deleted == 2
        assert uploader.deleted_ids == ["first-id", "second-id"]
        assert await cache.aget_by_hash("first-hash", "openai") is not None
        assert await cache.aget_by_hash("second-hash", "openai") is None

    asyncio.run(scenario())
