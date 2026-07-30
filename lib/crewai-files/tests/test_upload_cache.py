"""Tests for upload cache."""

import asyncio
from datetime import datetime, timedelta, timezone
import time

from crewai_files import FileBytes, ImageFile
from crewai_files.cache.upload_cache import (
    CachedUpload,
    UploadCache,
    _compute_file_hash,
)
import pytest


# Minimal valid PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08"
    b"\x01\x00\x00\x00\x00\xf9Y\xab\xcd\x00\x00\x00\nIDATx\x9cc`\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestCachedUpload:
    """Tests for CachedUpload dataclass."""

    def test_cached_upload_creation(self):
        """Test creating a cached upload."""
        now = datetime.now(timezone.utc)
        cached = CachedUpload(
            file_id="file-123",
            provider="gemini",
            file_uri="files/file-123",
            content_type="image/png",
            uploaded_at=now,
            expires_at=now + timedelta(hours=48),
        )

        assert cached.file_id == "file-123"
        assert cached.provider == "gemini"
        assert cached.file_uri == "files/file-123"
        assert cached.content_type == "image/png"

    def test_is_expired_false(self):
        """Test is_expired returns False for non-expired upload."""
        future = datetime.now(timezone.utc) + timedelta(hours=24)
        cached = CachedUpload(
            file_id="file-123",
            provider="gemini",
            file_uri=None,
            content_type="image/png",
            uploaded_at=datetime.now(timezone.utc),
            expires_at=future,
        )

        assert cached.is_expired() is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired upload."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        cached = CachedUpload(
            file_id="file-123",
            provider="gemini",
            file_uri=None,
            content_type="image/png",
            uploaded_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=past,
        )

        assert cached.is_expired() is True

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when no expiry set."""
        cached = CachedUpload(
            file_id="file-123",
            provider="anthropic",
            file_uri=None,
            content_type="image/png",
            uploaded_at=datetime.now(timezone.utc),
            expires_at=None,
        )

        assert cached.is_expired() is False


class TestUploadCache:
    """Tests for UploadCache class."""

    def test_cache_creation(self):
        """Test creating an empty cache."""
        cache = UploadCache()

        assert len(cache) == 0

    def test_set_and_get(self):
        """Test setting and getting cached uploads."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(
            file=file,
            provider="gemini",
            file_id="file-123",
            file_uri="files/file-123",
        )

        result = cache.get(file, "gemini")

        assert result is not None
        assert result.file_id == "file-123"
        assert result.provider == "gemini"

    def test_get_missing(self):
        """Test getting non-existent entry returns None."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        result = cache.get(file, "gemini")

        assert result is None

    def test_get_different_provider(self):
        """Test getting with different provider returns None."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(file=file, provider="gemini", file_id="file-123")

        result = cache.get(file, "anthropic")  # Different provider

        assert result is None

    def test_remove(self):
        """Test removing cached entry."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(file=file, provider="gemini", file_id="file-123")
        removed = cache.remove(file, "gemini")

        assert removed is True
        assert cache.get(file, "gemini") is None

    def test_remove_missing(self):
        """Test removing non-existent entry returns False."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        removed = cache.remove(file, "gemini")

        assert removed is False

    def test_remove_by_file_id(self):
        """Test removing by file ID."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(file=file, provider="gemini", file_id="file-123")
        removed = cache.remove_by_file_id("file-123", "gemini")

        assert removed is True
        assert len(cache) == 0

    def test_clear_expired(self):
        """Test clearing expired entries."""
        cache = UploadCache()
        file1 = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test1.png"))
        file2 = ImageFile(
            source=FileBytes(data=MINIMAL_PNG + b"x", filename="test2.png")
        )

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        future = datetime.now(timezone.utc) + timedelta(hours=24)

        cache.set(file=file1, provider="gemini", file_id="expired", expires_at=past)
        cache.set(file=file2, provider="gemini", file_id="valid", expires_at=future)

        removed = cache.clear_expired()

        assert removed == 1
        assert len(cache) == 1
        assert cache.get(file2, "gemini") is not None

    def test_clear(self):
        """Test clearing all entries."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(file=file, provider="gemini", file_id="file-123")
        cache.set(file=file, provider="anthropic", file_id="file-456")

        cleared = cache.clear()

        assert cleared == 2
        assert len(cache) == 0

    def test_get_all_for_provider(self):
        """Test getting all cached uploads for a provider."""
        cache = UploadCache()
        file1 = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test1.png"))
        file2 = ImageFile(
            source=FileBytes(data=MINIMAL_PNG + b"x", filename="test2.png")
        )
        file3 = ImageFile(
            source=FileBytes(data=MINIMAL_PNG + b"xx", filename="test3.png")
        )

        cache.set(file=file1, provider="gemini", file_id="file-1")
        cache.set(file=file2, provider="gemini", file_id="file-2")
        cache.set(file=file3, provider="anthropic", file_id="file-3")

        gemini_uploads = cache.get_all_for_provider("gemini")
        anthropic_uploads = cache.get_all_for_provider("anthropic")

        assert len(gemini_uploads) == 2
        assert len(anthropic_uploads) == 1


class TestRunSyncFromInsideAnEventLoop:
    """The sync wrappers must work when a loop is already running.

    ``UploadCache`` exposes ten synchronous wrappers that all funnel through
    ``_run_sync``. Its in-loop branch used to schedule the coroutine on the
    caller's own running loop and then block waiting for it, which the loop
    could never satisfy because ``_run_sync`` occupies its thread. Every such
    call stalled for the full 30s timeout and then raised ``TimeoutError``.
    """

    @staticmethod
    def _make_file(suffix: bytes = b"") -> ImageFile:
        return ImageFile(
            source=FileBytes(data=MINIMAL_PNG + suffix, filename="test.png")
        )

    def test_set_and_get_outside_event_loop(self):
        """Baseline: with no running loop the wrappers already worked."""
        cache = UploadCache()
        file = self._make_file()

        cache.set(file=file, provider="gemini", file_id="file-123")

        cached = cache.get(file=file, provider="gemini")
        assert cached is not None
        assert cached.file_id == "file-123"

    @pytest.mark.asyncio
    async def test_set_and_get_inside_running_event_loop(self):
        """The regression: sync wrappers called from async code.

        A stalled call would fail this by timing out rather than by
        returning a wrong value, so the assertion is preceded by a wall
        clock bound: the whole exchange must finish well inside the 30s
        timeout the old code always spent.
        """
        cache = UploadCache()
        file = self._make_file()

        started = time.monotonic()
        cache.set(file=file, provider="gemini", file_id="file-123")
        cached = cache.get(file=file, provider="gemini")
        elapsed = time.monotonic() - started

        assert cached is not None
        assert cached.file_id == "file-123"
        assert elapsed < 10, (
            f"took {elapsed:.1f}s -- the coroutine is being scheduled on the "
            "caller's own loop, which _run_sync blocks while waiting"
        )

    @pytest.mark.asyncio
    async def test_sync_wrappers_do_not_use_the_callers_loop(self):
        """The coroutine must not be driven by the loop ``_run_sync`` blocks.

        Scheduling it there is the cause of the stall, so the property is
        asserted directly rather than inferred from the absence of a
        timeout.
        """
        caller_loop = asyncio.get_running_loop()
        seen: list[asyncio.AbstractEventLoop] = []

        async def probe() -> str:
            seen.append(asyncio.get_running_loop())
            return "done"

        assert UploadCache._run_sync(probe()) == "done"
        assert seen and seen[0] is not caller_loop

    @pytest.mark.asyncio
    async def test_remaining_sync_wrappers_inside_running_event_loop(self):
        """The other wrappers share ``_run_sync``, so they are covered too."""
        cache = UploadCache()
        file = self._make_file()
        other = self._make_file(suffix=b"x")

        cache.set(file=file, provider="gemini", file_id="file-1")
        cache.set(file=other, provider="gemini", file_id="file-2")

        assert len(cache.get_all_for_provider("gemini")) == 2
        assert cache.get_by_hash(_compute_file_hash(file), "gemini") is not None
        assert cache.remove(file=file, provider="gemini") is True
        assert cache.remove_by_file_id("file-2", "gemini") is True
        assert cache.clear_expired() == 0
        assert cache.clear() == 0

    @pytest.mark.asyncio
    async def test_exceptions_propagate_from_inside_running_event_loop(self):
        """A failure inside the coroutine must reach the sync caller."""

        async def boom() -> None:
            raise RuntimeError("coroutine failed")

        with pytest.raises(RuntimeError, match="coroutine failed"):
            UploadCache._run_sync(boom())
