"""Tests for upload cache."""

from datetime import datetime, timedelta, timezone

from crewai_files import FileBytes, ImageFile
from crewai_files.cache.upload_cache import CachedUpload, UploadCache


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

        # Add one expired and one valid entry
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
