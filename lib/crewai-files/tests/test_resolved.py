"""Tests for resolved file types."""

from datetime import datetime, timezone

from crewai_files.core.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
    UrlReference,
)
import pytest


class TestInlineBase64:
    """Tests for InlineBase64 resolved type."""

    def test_create_inline_base64(self):
        """Test creating InlineBase64 instance."""
        resolved = InlineBase64(
            content_type="image/png",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        )

        assert resolved.content_type == "image/png"
        assert len(resolved.data) > 0

    def test_inline_base64_is_resolved_file(self):
        """Test InlineBase64 is a ResolvedFile."""
        resolved = InlineBase64(content_type="image/png", data="abc123")

        assert isinstance(resolved, ResolvedFile)

    def test_inline_base64_frozen(self):
        """Test InlineBase64 is immutable."""
        resolved = InlineBase64(content_type="image/png", data="abc123")

        with pytest.raises(Exception):
            resolved.data = "xyz789"


class TestInlineBytes:
    """Tests for InlineBytes resolved type."""

    def test_create_inline_bytes(self):
        """Test creating InlineBytes instance."""
        data = b"\x89PNG\r\n\x1a\n"
        resolved = InlineBytes(
            content_type="image/png",
            data=data,
        )

        assert resolved.content_type == "image/png"
        assert resolved.data == data

    def test_inline_bytes_is_resolved_file(self):
        """Test InlineBytes is a ResolvedFile."""
        resolved = InlineBytes(content_type="image/png", data=b"test")

        assert isinstance(resolved, ResolvedFile)


class TestFileReference:
    """Tests for FileReference resolved type."""

    def test_create_file_reference(self):
        """Test creating FileReference instance."""
        resolved = FileReference(
            content_type="image/png",
            file_id="file-abc123",
            provider="gemini",
        )

        assert resolved.content_type == "image/png"
        assert resolved.file_id == "file-abc123"
        assert resolved.provider == "gemini"
        assert resolved.expires_at is None
        assert resolved.file_uri is None

    def test_file_reference_with_expiry(self):
        """Test FileReference with expiry time."""
        expiry = datetime.now(timezone.utc)
        resolved = FileReference(
            content_type="application/pdf",
            file_id="file-xyz789",
            provider="gemini",
            expires_at=expiry,
        )

        assert resolved.expires_at == expiry

    def test_file_reference_with_uri(self):
        """Test FileReference with URI."""
        resolved = FileReference(
            content_type="video/mp4",
            file_id="file-video123",
            provider="gemini",
            file_uri="https://generativelanguage.googleapis.com/v1/files/file-video123",
        )

        assert resolved.file_uri is not None

    def test_file_reference_is_resolved_file(self):
        """Test FileReference is a ResolvedFile."""
        resolved = FileReference(
            content_type="image/png",
            file_id="file-123",
            provider="anthropic",
        )

        assert isinstance(resolved, ResolvedFile)


class TestUrlReference:
    """Tests for UrlReference resolved type."""

    def test_create_url_reference(self):
        """Test creating UrlReference instance."""
        resolved = UrlReference(
            content_type="image/png",
            url="https://storage.googleapis.com/bucket/image.png",
        )

        assert resolved.content_type == "image/png"
        assert resolved.url == "https://storage.googleapis.com/bucket/image.png"

    def test_url_reference_is_resolved_file(self):
        """Test UrlReference is a ResolvedFile."""
        resolved = UrlReference(
            content_type="image/jpeg",
            url="https://example.com/photo.jpg",
        )

        assert isinstance(resolved, ResolvedFile)
