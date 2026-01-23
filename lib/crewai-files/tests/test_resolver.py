"""Tests for FileResolver."""

from crewai_files import FileBytes, ImageFile
from crewai_files.cache.upload_cache import UploadCache
from crewai_files.core.resolved import InlineBase64, InlineBytes
from crewai_files.resolution.resolver import (
    FileResolver,
    FileResolverConfig,
    create_resolver,
)


# Minimal valid PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08"
    b"\x01\x00\x00\x00\x00\xf9Y\xab\xcd\x00\x00\x00\nIDATx\x9cc`\x00\x00"
    b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestFileResolverConfig:
    """Tests for FileResolverConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FileResolverConfig()

        assert config.prefer_upload is False
        assert config.upload_threshold_bytes is None
        assert config.use_bytes_for_bedrock is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FileResolverConfig(
            prefer_upload=True,
            upload_threshold_bytes=1024 * 1024,
            use_bytes_for_bedrock=False,
        )

        assert config.prefer_upload is True
        assert config.upload_threshold_bytes == 1024 * 1024
        assert config.use_bytes_for_bedrock is False


class TestFileResolver:
    """Tests for FileResolver class."""

    def test_resolve_inline_base64(self):
        """Test resolving file as inline base64."""
        resolver = FileResolver()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        resolved = resolver.resolve(file, "openai")

        assert isinstance(resolved, InlineBase64)
        assert resolved.content_type == "image/png"
        assert len(resolved.data) > 0

    def test_resolve_inline_bytes_for_bedrock(self):
        """Test resolving file as inline bytes for Bedrock."""
        config = FileResolverConfig(use_bytes_for_bedrock=True)
        resolver = FileResolver(config=config)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        resolved = resolver.resolve(file, "bedrock")

        assert isinstance(resolved, InlineBytes)
        assert resolved.content_type == "image/png"
        assert resolved.data == MINIMAL_PNG

    def test_resolve_files_multiple(self):
        """Test resolving multiple files."""
        resolver = FileResolver()
        files = {
            "image1": ImageFile(
                source=FileBytes(data=MINIMAL_PNG, filename="test1.png")
            ),
            "image2": ImageFile(
                source=FileBytes(data=MINIMAL_PNG, filename="test2.png")
            ),
        }

        resolved = resolver.resolve_files(files, "openai")

        assert len(resolved) == 2
        assert "image1" in resolved
        assert "image2" in resolved
        assert all(isinstance(r, InlineBase64) for r in resolved.values())

    def test_resolve_with_cache(self):
        """Test resolver uses cache."""
        cache = UploadCache()
        resolver = FileResolver(upload_cache=cache)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        # First resolution
        resolved1 = resolver.resolve(file, "openai")
        # Second resolution (should use same base64 encoding)
        resolved2 = resolver.resolve(file, "openai")

        assert isinstance(resolved1, InlineBase64)
        assert isinstance(resolved2, InlineBase64)
        # Data should be identical
        assert resolved1.data == resolved2.data

    def test_clear_cache(self):
        """Test clearing resolver cache."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        # Add something to cache manually
        cache.set(file=file, provider="gemini", file_id="test")

        resolver = FileResolver(upload_cache=cache)
        resolver.clear_cache()

        assert len(cache) == 0

    def test_get_cached_uploads(self):
        """Test getting cached uploads from resolver."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        cache.set(file=file, provider="gemini", file_id="test-1")
        cache.set(file=file, provider="anthropic", file_id="test-2")

        resolver = FileResolver(upload_cache=cache)

        gemini_uploads = resolver.get_cached_uploads("gemini")
        anthropic_uploads = resolver.get_cached_uploads("anthropic")

        assert len(gemini_uploads) == 1
        assert len(anthropic_uploads) == 1

    def test_get_cached_uploads_empty(self):
        """Test getting cached uploads when no cache."""
        resolver = FileResolver()  # No cache

        uploads = resolver.get_cached_uploads("gemini")

        assert uploads == []


class TestCreateResolver:
    """Tests for create_resolver factory function."""

    def test_create_default_resolver(self):
        """Test creating resolver with default settings."""
        resolver = create_resolver()

        assert resolver.config.prefer_upload is False
        assert resolver.upload_cache is not None

    def test_create_resolver_with_options(self):
        """Test creating resolver with custom options."""
        resolver = create_resolver(
            prefer_upload=True,
            upload_threshold_bytes=5 * 1024 * 1024,
            enable_cache=False,
        )

        assert resolver.config.prefer_upload is True
        assert resolver.config.upload_threshold_bytes == 5 * 1024 * 1024
        assert resolver.upload_cache is None

    def test_create_resolver_cache_enabled(self):
        """Test resolver has cache when enabled."""
        resolver = create_resolver(enable_cache=True)

        assert resolver.upload_cache is not None

    def test_create_resolver_cache_disabled(self):
        """Test resolver has no cache when disabled."""
        resolver = create_resolver(enable_cache=False)

        assert resolver.upload_cache is None
