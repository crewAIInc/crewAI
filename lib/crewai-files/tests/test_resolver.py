"""Tests for FileResolver."""

from crewai_files import FileBytes, ImageFile
from crewai_files.cache.upload_cache import UploadCache
from crewai_files.core.resolved import InlineBase64, InlineBytes
from crewai_files.processing.exceptions import UploaderConfigurationError
from crewai_files.resolution.resolver import (
    FileResolver,
    FileResolverConfig,
    create_resolver,
)
import pytest


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

        resolved1 = resolver.resolve(file, "openai")
        resolved2 = resolver.resolve(file, "openai")

        assert isinstance(resolved1, InlineBase64)
        assert isinstance(resolved2, InlineBase64)
        assert resolved1.data == resolved2.data

    def test_clear_cache(self):
        """Test clearing resolver cache."""
        cache = UploadCache()
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

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


class _NoUploaderResolver(FileResolver):
    """Resolver whose provider has no usable uploader, so every file fails setup."""

    def _get_uploader(self, provider):
        raise UploaderConfigurationError(
            f"no file uploader available for provider {provider!r}"
        )


class _OneBadFileResolver(FileResolver):
    """Resolver that fails one specific file with an ordinary per-file error."""

    async def aresolve(self, file, provider):
        if file.filename == "bad.png":
            raise ValueError("corrupt image stream")
        return await super().aresolve(file, provider)


class TestBatchUploaderErrors:
    """A provider setup failure must surface, an unrelated per-file error must not."""

    def test_get_uploader_wraps_lookup_failure_as_configuration_error(
        self, monkeypatch
    ):
        """Bedrock with no bucket configured raises UploaderConfigurationError, not a raw ValueError."""
        monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
        resolver = FileResolver()

        with pytest.raises(
            UploaderConfigurationError, match="CREWAI_BEDROCK_S3_BUCKET"
        ):
            resolver._get_uploader("bedrock")

    @pytest.mark.asyncio
    async def test_aresolve_files_surfaces_uploader_configuration_error(self):
        """A provider whose uploader cannot be built aborts the whole batch, since it affects every file."""
        resolver = _NoUploaderResolver(config=FileResolverConfig(prefer_upload=True))
        files = {
            "image1": ImageFile(
                source=FileBytes(data=MINIMAL_PNG, filename="test1.png")
            )
        }

        with pytest.raises(UploaderConfigurationError):
            await resolver.aresolve_files(files, "openai")

    @pytest.mark.asyncio
    async def test_aresolve_files_skips_unrelated_per_file_errors(self):
        """One file failing with an ordinary error is logged and skipped; the rest still resolve."""
        resolver = _OneBadFileResolver()
        files = {
            "good": ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="good.png")),
            "bad": ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="bad.png")),
        }

        resolved = await resolver.aresolve_files(files, "openai")

        assert set(resolved) == {"good"}
        assert isinstance(resolved["good"], InlineBase64)
