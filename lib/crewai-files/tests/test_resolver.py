"""Tests for FileResolver."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiofiles
from crewai_files import FileBytes, FilePath, FileStream, ImageFile, TextFile
from crewai_files.cache.upload_cache import UploadCache
from crewai_files.core.resolved import FileReference, InlineBase64, InlineBytes
from crewai_files.core.sources import AsyncFileStream
from crewai_files.processing.exceptions import UploaderConfigurationError
from crewai_files.resolution.resolver import (
    FileResolver,
    FileResolverConfig,
    create_resolver,
)
from crewai_files.uploaders.openai import OpenAIFileUploader
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


class TestAsyncFileResolver:
    """Async resolution reads async sources without losing sync-source support."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("preloaded", [False, True])
    @pytest.mark.parametrize("provider", ["openai", "bedrock"])
    async def test_aresolve_async_stream(
        self, tmp_path: Path, preloaded: bool, provider: str
    ) -> None:
        """Resolve unread and cached async streams for both inline formats."""
        path = tmp_path / "image.png"
        path.write_bytes(MINIMAL_PNG)

        async with aiofiles.open(path, "rb") as stream:
            source = AsyncFileStream(stream=stream, filename="image.png")
            if preloaded:
                await source.aread()
            file = ImageFile(source=source)

            resolved = await FileResolver().aresolve(file, provider)

            assert resolved.content_type == "image/png"
            if provider == "bedrock":
                assert isinstance(resolved, InlineBytes)
                assert resolved.data == MINIMAL_PNG
            else:
                assert isinstance(resolved, InlineBase64)
                assert base64.b64decode(resolved.data) == MINIMAL_PNG
            assert await stream.tell() == len(MINIMAL_PNG)
            assert await file.aread() == MINIMAL_PNG

    @pytest.mark.asyncio
    async def test_aresolve_files_mixed_sources(self, tmp_path: Path) -> None:
        """Resolve mixed sources without rereading a shared sync stream."""
        path = tmp_path / "image.png"
        path.write_bytes(MINIMAL_PNG)

        async with aiofiles.open(path, "rb") as stream:
            sync_file = ImageFile(source=FileStream(stream=BytesIO(MINIMAL_PNG)))
            files = {
                "async_stream": ImageFile(
                    source=AsyncFileStream(stream=stream, filename="image.png")
                ),
                "bytes": ImageFile(source=FileBytes(data=MINIMAL_PNG)),
                "path": ImageFile(source=FilePath(path=path)),
                "sync_stream": sync_file,
                "shared_sync_stream": sync_file,
            }

            resolved = await FileResolver().aresolve_files(files, "openai")

            assert resolved.keys() == files.keys()
            for result in resolved.values():
                assert isinstance(result, InlineBase64)
                assert result.content_type == "image/png"
                assert base64.b64decode(result.data) == MINIMAL_PNG

    @pytest.mark.asyncio
    async def test_aresolve_upload_async_stream_reads_once(self) -> None:
        """Reuse async content and the provider's cached upload reference."""
        stream = SimpleNamespace(read=AsyncMock(return_value=MINIMAL_PNG))
        file = ImageFile(source=AsyncFileStream(stream=stream, filename="image.png"))
        create_file = AsyncMock(return_value=SimpleNamespace(id="file-test"))
        uploader = OpenAIFileUploader(
            async_client=SimpleNamespace(files=SimpleNamespace(create=create_file))
        )
        resolver = FileResolver(
            config=FileResolverConfig(prefer_upload=True),
            upload_cache=UploadCache(),
            _uploaders={"openai": uploader},
        )

        first = await resolver.aresolve(file, "openai")
        cached = await resolver.aresolve(file, "openai")

        assert isinstance(first, FileReference)
        assert isinstance(cached, FileReference)
        assert first.file_id == cached.file_id == "file-test"
        assert first.content_type == "image/png"
        stream.read.assert_awaited_once_with()
        create_file.assert_awaited_once()
        assert create_file.call_args.kwargs["file"].getvalue() == MINIMAL_PNG
        assert create_file.call_args.kwargs["purpose"] == "vision"

    @pytest.mark.asyncio
    async def test_aresolve_empty_async_stream(self) -> None:
        """Treat empty async content as loaded rather than rereading it."""
        stream = SimpleNamespace(read=AsyncMock(return_value=b""))
        file = TextFile(source=AsyncFileStream(stream=stream, filename="empty.txt"))

        resolved = await FileResolver().aresolve(file, "openai")

        assert isinstance(resolved, InlineBase64)
        assert resolved.data == ""
        stream.read.assert_awaited_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", [b"shared async content", b""])
    @pytest.mark.parametrize("separate_files", [False, True])
    async def test_aresolve_files_shared_async_stream(
        self, content: bytes, separate_files: bool
    ) -> None:
        """Concurrent batch entries share one initial read, including empty files."""
        buffer = BytesIO(content)

        async def read() -> bytes:
            """Yield before reading the cursor so batch entries overlap."""
            await asyncio.sleep(0)
            return buffer.read()

        stream = SimpleNamespace(read=AsyncMock(side_effect=read))
        source = AsyncFileStream(stream=stream, filename="shared.txt")
        first = TextFile(source=source)
        second = TextFile(source=source) if separate_files else first

        resolved = await FileResolver().aresolve_files(
            {"first": first, "second": second}, "openai"
        )

        assert set(resolved) == {"first", "second"}
        for result in resolved.values():
            assert isinstance(result, InlineBase64)
            assert base64.b64decode(result.data) == content
        assert await source.aread() == content
        stream.read.assert_awaited_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error", [OSError("read failed"), asyncio.CancelledError()]
    )
    async def test_aresolve_async_stream_after_read_error(
        self, error: BaseException
    ) -> None:
        """A failed or cancelled first read does not block a later resolution."""
        stream = SimpleNamespace(read=AsyncMock(side_effect=[error, MINIMAL_PNG]))
        source = AsyncFileStream(stream=stream, filename="image.png")
        with pytest.raises(type(error)):
            await source.aread()

        resolved = await asyncio.wait_for(
            FileResolver().aresolve(ImageFile(source=source), "openai"), timeout=1
        )

        assert isinstance(resolved, InlineBase64)
        assert base64.b64decode(resolved.data) == MINIMAL_PNG
        assert await source.aread() == MINIMAL_PNG
        assert stream.read.await_count == 2


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
