"""Cleanup utilities for uploaded files."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from crewai.files.upload_cache import CachedUpload, UploadCache
from crewai.files.uploaders import get_uploader


if TYPE_CHECKING:
    from crewai.files.uploaders.base import FileUploader

logger = logging.getLogger(__name__)


def _safe_delete(
    uploader: FileUploader,
    file_id: str,
    provider: str,
) -> bool:
    """Safely delete a file, logging any errors.

    Args:
        uploader: The file uploader to use.
        file_id: The file ID to delete.
        provider: Provider name for logging.

    Returns:
        True if deleted successfully, False otherwise.
    """
    try:
        if uploader.delete(file_id):
            logger.debug(f"Deleted {file_id} from {provider}")
            return True
        logger.warning(f"Failed to delete {file_id} from {provider}")
        return False
    except Exception as e:
        logger.warning(f"Error deleting {file_id} from {provider}: {e}")
        return False


def cleanup_uploaded_files(
    cache: UploadCache,
    *,
    delete_from_provider: bool = True,
    providers: list[str] | None = None,
) -> int:
    """Clean up uploaded files from the cache and optionally from providers.

    Args:
        cache: The upload cache to clean up.
        delete_from_provider: If True, delete files from the provider as well.
        providers: Optional list of providers to clean up. If None, cleans all.

    Returns:
        Number of files cleaned up.
    """
    cleaned = 0

    provider_uploads: dict[str, list[CachedUpload]] = {}

    for provider in _get_providers_from_cache(cache):
        if providers is not None and provider not in providers:
            continue
        provider_uploads[provider] = cache.get_all_for_provider(provider)

    if delete_from_provider:
        for provider, uploads in provider_uploads.items():
            uploader = get_uploader(provider)
            if uploader is None:
                logger.warning(
                    f"No uploader available for {provider}, skipping cleanup"
                )
                continue

            for upload in uploads:
                if _safe_delete(uploader, upload.file_id, provider):
                    cleaned += 1

    cache.clear()

    logger.info(f"Cleaned up {cleaned} uploaded files")
    return cleaned


def cleanup_expired_files(
    cache: UploadCache,
    *,
    delete_from_provider: bool = False,
) -> int:
    """Clean up expired files from the cache.

    Args:
        cache: The upload cache to clean up.
        delete_from_provider: If True, attempt to delete from provider as well.
            Note: Expired files may already be deleted by the provider.

    Returns:
        Number of expired entries removed from cache.
    """
    expired_entries: list[CachedUpload] = []

    if delete_from_provider:
        for provider in _get_providers_from_cache(cache):
            expired_entries.extend(
                upload
                for upload in cache.get_all_for_provider(provider)
                if upload.is_expired()
            )

    removed = cache.clear_expired()

    if delete_from_provider:
        for upload in expired_entries:
            uploader = get_uploader(upload.provider)
            if uploader is not None:
                try:
                    uploader.delete(upload.file_id)
                except Exception as e:
                    logger.debug(f"Could not delete expired file {upload.file_id}: {e}")

    return removed


def cleanup_provider_files(
    provider: str,
    *,
    cache: UploadCache | None = None,
    delete_all_from_provider: bool = False,
) -> int:
    """Clean up all files for a specific provider.

    Args:
        provider: Provider name to clean up.
        cache: Optional upload cache to clear entries from.
        delete_all_from_provider: If True, delete all files from the provider,
            not just cached ones.

    Returns:
        Number of files deleted.
    """
    deleted = 0
    uploader = get_uploader(provider)

    if uploader is None:
        logger.warning(f"No uploader available for {provider}")
        return 0

    if delete_all_from_provider:
        try:
            files = uploader.list_files()
            for file_info in files:
                file_id = file_info.get("id") or file_info.get("name")
                if file_id and uploader.delete(file_id):
                    deleted += 1
        except Exception as e:
            logger.warning(f"Error listing/deleting files from {provider}: {e}")
    elif cache is not None:
        uploads = cache.get_all_for_provider(provider)
        for upload in uploads:
            if _safe_delete(uploader, upload.file_id, provider):
                deleted += 1
                cache.remove_by_file_id(upload.file_id, provider)

    logger.info(f"Deleted {deleted} files from {provider}")
    return deleted


def _get_providers_from_cache(cache: UploadCache) -> set[str]:
    """Get unique provider names from cache entries.

    Args:
        cache: The upload cache.

    Returns:
        Set of provider names.
    """
    return cache.get_providers()


async def _asafe_delete(
    uploader: FileUploader,
    file_id: str,
    provider: str,
) -> bool:
    """Async safely delete a file, logging any errors.

    Args:
        uploader: The file uploader to use.
        file_id: The file ID to delete.
        provider: Provider name for logging.

    Returns:
        True if deleted successfully, False otherwise.
    """
    try:
        if await uploader.adelete(file_id):
            logger.debug(f"Deleted {file_id} from {provider}")
            return True
        logger.warning(f"Failed to delete {file_id} from {provider}")
        return False
    except Exception as e:
        logger.warning(f"Error deleting {file_id} from {provider}: {e}")
        return False


async def acleanup_uploaded_files(
    cache: UploadCache,
    *,
    delete_from_provider: bool = True,
    providers: list[str] | None = None,
    max_concurrency: int = 10,
) -> int:
    """Async clean up uploaded files from the cache and optionally from providers.

    Args:
        cache: The upload cache to clean up.
        delete_from_provider: If True, delete files from the provider as well.
        providers: Optional list of providers to clean up. If None, cleans all.
        max_concurrency: Maximum number of concurrent delete operations.

    Returns:
        Number of files cleaned up.
    """
    cleaned = 0

    provider_uploads: dict[str, list[CachedUpload]] = {}

    for provider in _get_providers_from_cache(cache):
        if providers is not None and provider not in providers:
            continue
        provider_uploads[provider] = await cache.aget_all_for_provider(provider)

    if delete_from_provider:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def delete_one(uploader: FileUploader, upload: CachedUpload) -> bool:
            async with semaphore:
                return await _asafe_delete(uploader, upload.file_id, upload.provider)

        tasks: list[asyncio.Task[bool]] = []
        for provider, uploads in provider_uploads.items():
            uploader = get_uploader(provider)
            if uploader is None:
                logger.warning(
                    f"No uploader available for {provider}, skipping cleanup"
                )
                continue

            tasks.extend(
                asyncio.create_task(delete_one(uploader, upload)) for upload in uploads
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        cleaned = sum(1 for r in results if r is True)

    await cache.aclear()

    logger.info(f"Cleaned up {cleaned} uploaded files")
    return cleaned


async def acleanup_expired_files(
    cache: UploadCache,
    *,
    delete_from_provider: bool = False,
    max_concurrency: int = 10,
) -> int:
    """Async clean up expired files from the cache.

    Args:
        cache: The upload cache to clean up.
        delete_from_provider: If True, attempt to delete from provider as well.
        max_concurrency: Maximum number of concurrent delete operations.

    Returns:
        Number of expired entries removed from cache.
    """
    expired_entries: list[CachedUpload] = []

    if delete_from_provider:
        for provider in _get_providers_from_cache(cache):
            uploads = await cache.aget_all_for_provider(provider)
            expired_entries.extend(upload for upload in uploads if upload.is_expired())

    removed = await cache.aclear_expired()

    if delete_from_provider and expired_entries:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def delete_expired(upload: CachedUpload) -> None:
            async with semaphore:
                uploader = get_uploader(upload.provider)
                if uploader is not None:
                    try:
                        await uploader.adelete(upload.file_id)
                    except Exception as e:
                        logger.debug(
                            f"Could not delete expired file {upload.file_id}: {e}"
                        )

        await asyncio.gather(
            *[delete_expired(upload) for upload in expired_entries],
            return_exceptions=True,
        )

    return removed


async def acleanup_provider_files(
    provider: str,
    *,
    cache: UploadCache | None = None,
    delete_all_from_provider: bool = False,
    max_concurrency: int = 10,
) -> int:
    """Async clean up all files for a specific provider.

    Args:
        provider: Provider name to clean up.
        cache: Optional upload cache to clear entries from.
        delete_all_from_provider: If True, delete all files from the provider.
        max_concurrency: Maximum number of concurrent delete operations.

    Returns:
        Number of files deleted.
    """
    deleted = 0
    uploader = get_uploader(provider)

    if uploader is None:
        logger.warning(f"No uploader available for {provider}")
        return 0

    semaphore = asyncio.Semaphore(max_concurrency)

    async def delete_file(file_id: str) -> bool:
        async with semaphore:
            return await uploader.adelete(file_id)

    if delete_all_from_provider:
        try:
            files = uploader.list_files()
            tasks = []
            for file_info in files:
                file_id = file_info.get("id") or file_info.get("name")
                if file_id:
                    tasks.append(delete_file(file_id))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            deleted = sum(1 for r in results if r is True)
        except Exception as e:
            logger.warning(f"Error listing/deleting files from {provider}: {e}")
    elif cache is not None:
        uploads = await cache.aget_all_for_provider(provider)
        tasks = []
        for upload in uploads:
            tasks.append(delete_file(upload.file_id))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for upload, result in zip(uploads, results, strict=False):
            if result is True:
                deleted += 1
                await cache.aremove_by_file_id(upload.file_id, provider)

    logger.info(f"Deleted {deleted} files from {provider}")
    return deleted
