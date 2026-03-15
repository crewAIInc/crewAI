"""Unit tests for files module."""

import io
import tempfile
from pathlib import Path

import pytest

from crewai_files import (
    AudioFile,
    File,
    FileBytes,
    FilePath,
    FileSource,
    FileStream,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
    normalize_input_files,
    wrap_file_source,
)
from crewai_files.core.sources import detect_content_type


class TestDetectContentType:
    """Tests for MIME type detection."""

    def test_detect_plain_text(self) -> None:
        """Test detection of plain text content."""
        result = detect_content_type(b"Hello, World!")
        assert result == "text/plain"

    def test_detect_json(self) -> None:
        """Test detection of JSON content."""
        result = detect_content_type(b'{"key": "value"}')
        assert result == "application/json"

    def test_detect_png(self) -> None:
        """Test detection of PNG content."""
        # Minimal valid PNG: header + IHDR chunk + IEND chunk
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk length and type
            b"\x00\x00\x00\x01"  # width: 1
            b"\x00\x00\x00\x01"  # height: 1
            b"\x08\x02"  # bit depth: 8, color type: 2 (RGB)
            b"\x00\x00\x00"  # compression, filter, interlace
            b"\x90wS\xde"  # CRC
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
        )
        result = detect_content_type(png_data)
        assert result == "image/png"

    def test_detect_jpeg(self) -> None:
        """Test detection of JPEG header."""
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = detect_content_type(jpeg_header)
        assert result == "image/jpeg"

    def test_detect_pdf(self) -> None:
        """Test detection of PDF header."""
        pdf_header = b"%PDF-1.4"
        result = detect_content_type(pdf_header)
        assert result == "application/pdf"


class TestFilePath:
    """Tests for FilePath class."""

    def test_create_from_existing_file(self, tmp_path: Path) -> None:
        """Test creating FilePath from an existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        fp = FilePath(path=file_path)

        assert fp.filename == "test.txt"
        assert fp.read() == b"test content"

    def test_content_is_cached(self, tmp_path: Path) -> None:
        """Test that file content is cached after first read."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")

        fp = FilePath(path=file_path)
        first_read = fp.read()

        # Modify file after first read
        file_path.write_text("modified")
        second_read = fp.read()

        assert first_read == second_read == b"original"

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """Test that FilePath raises for non-existent files."""
        with pytest.raises(ValueError, match="File not found"):
            FilePath(path=tmp_path / "nonexistent.txt")

    def test_raises_for_directory(self, tmp_path: Path) -> None:
        """Test that FilePath raises for directories."""
        with pytest.raises(ValueError, match="Path is not a file"):
            FilePath(path=tmp_path)

    def test_content_type_detection(self, tmp_path: Path) -> None:
        """Test content type detection from file content."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("plain text content")

        fp = FilePath(path=file_path)

        assert fp.content_type == "text/plain"


class TestFileBytes:
    """Tests for FileBytes class."""

    def test_create_from_bytes(self) -> None:
        """Test creating FileBytes from raw bytes."""
        fb = FileBytes(data=b"test data")

        assert fb.read() == b"test data"
        assert fb.filename is None

    def test_create_with_filename(self) -> None:
        """Test creating FileBytes with optional filename."""
        fb = FileBytes(data=b"test", filename="doc.txt")

        assert fb.filename == "doc.txt"

    def test_content_type_detection(self) -> None:
        """Test content type detection from bytes."""
        fb = FileBytes(data=b"text content")

        assert fb.content_type == "text/plain"


class TestFileStream:
    """Tests for FileStream class."""

    def test_create_from_stream(self) -> None:
        """Test creating FileStream from a file-like object."""
        stream = io.BytesIO(b"stream content")

        fs = FileStream(stream=stream)

        assert fs.read() == b"stream content"

    def test_content_is_cached(self) -> None:
        """Test that stream content is cached."""
        stream = io.BytesIO(b"original")

        fs = FileStream(stream=stream)
        first = fs.read()

        # Even after modifying stream, cached content is returned
        stream.seek(0)
        stream.write(b"modified")
        second = fs.read()

        assert first == second == b"original"

    def test_filename_from_stream(self, tmp_path: Path) -> None:
        """Test filename extraction from stream with name attribute."""
        file_path = tmp_path / "named.txt"
        file_path.write_text("content")

        with open(file_path, "rb") as f:
            fs = FileStream(stream=f)
            assert fs.filename == "named.txt"

    def test_close_stream(self) -> None:
        """Test closing the underlying stream."""
        stream = io.BytesIO(b"data")
        fs = FileStream(stream=stream)

        fs.close()

        assert stream.closed


class TestTypedFileWrappers:
    """Tests for typed file wrapper classes."""

    def test_image_file_from_bytes(self) -> None:
        """Test ImageFile creation from bytes."""
        # Minimal valid PNG structure
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img = ImageFile(source=png_bytes)

        assert img.content_type == "image/png"

    def test_image_file_from_path(self, tmp_path: Path) -> None:
        """Test ImageFile creation from path string."""
        file_path = tmp_path / "test.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        img = ImageFile(source=str(file_path))

        assert img.filename == "test.png"

    def test_text_file_read_text(self) -> None:
        """Test TextFile.read_text method."""
        tf = TextFile(source=b"Hello, World!")

        assert tf.read_text() == "Hello, World!"

    def test_pdf_file_creation(self) -> None:
        """Test PDFFile creation."""
        pdf_bytes = b"%PDF-1.4 content"
        pdf = PDFFile(source=pdf_bytes)

        assert pdf.read() == pdf_bytes

    def test_audio_file_creation(self) -> None:
        """Test AudioFile creation."""
        audio = AudioFile(source=b"audio data")
        assert audio.read() == b"audio data"

    def test_video_file_creation(self) -> None:
        """Test VideoFile creation."""
        video = VideoFile(source=b"video data")
        assert video.read() == b"video data"

    def test_dict_unpacking(self, tmp_path: Path) -> None:
        """Test that files support ** unpacking syntax."""
        file_path = tmp_path / "document.txt"
        file_path.write_text("content")

        tf = TextFile(source=str(file_path))

        # Unpack into dict
        result = {**tf}

        assert "document" in result
        assert result["document"] is tf

    def test_dict_unpacking_no_filename(self) -> None:
        """Test dict unpacking with bytes (no filename)."""
        tf = TextFile(source=b"content")
        result = {**tf}

        assert "file" in result

    def test_keys_method(self, tmp_path: Path) -> None:
        """Test keys() method for dict unpacking."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        tf = TextFile(source=str(file_path))

        assert tf.keys() == ["test"]

    def test_getitem_valid_key(self, tmp_path: Path) -> None:
        """Test __getitem__ with valid key."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("content")

        tf = TextFile(source=str(file_path))

        assert tf["doc"] is tf

    def test_getitem_invalid_key(self, tmp_path: Path) -> None:
        """Test __getitem__ with invalid key raises KeyError."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("content")

        tf = TextFile(source=str(file_path))

        with pytest.raises(KeyError):
            _ = tf["wrong_key"]


class TestWrapFileSource:
    """Tests for wrap_file_source function."""

    def test_wrap_image_source(self) -> None:
        """Test wrapping image source returns ImageFile."""
        # Minimal valid PNG structure
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        source = FileBytes(data=png_bytes)

        result = wrap_file_source(source)

        assert isinstance(result, ImageFile)

    def test_wrap_pdf_source(self) -> None:
        """Test wrapping PDF source returns PDFFile."""
        source = FileBytes(data=b"%PDF-1.4 content")

        result = wrap_file_source(source)

        assert isinstance(result, PDFFile)

    def test_wrap_text_source(self) -> None:
        """Test wrapping text source returns TextFile."""
        source = FileBytes(data=b"plain text")

        result = wrap_file_source(source)

        assert isinstance(result, TextFile)


class TestNormalizeInputFiles:
    """Tests for normalize_input_files function."""

    def test_normalize_path_strings(self, tmp_path: Path) -> None:
        """Test normalizing path strings."""
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        result = normalize_input_files([str(file1), str(file2)])

        assert "doc1.txt" in result
        assert "doc2.txt" in result

    def test_normalize_path_objects(self, tmp_path: Path) -> None:
        """Test normalizing Path objects."""
        file_path = tmp_path / "document.txt"
        file_path.write_text("content")

        result = normalize_input_files([file_path])

        assert "document.txt" in result

    def test_normalize_bytes(self) -> None:
        """Test normalizing raw bytes."""
        result = normalize_input_files([b"content1", b"content2"])

        assert "file_0" in result
        assert "file_1" in result

    def test_normalize_file_source(self) -> None:
        """Test normalizing FileSource objects."""
        source = FileBytes(data=b"content", filename="named.txt")

        result = normalize_input_files([source])

        assert "named.txt" in result

    def test_normalize_mixed_inputs(self, tmp_path: Path) -> None:
        """Test normalizing mixed input types."""
        file_path = tmp_path / "path.txt"
        file_path.write_text("from path")

        inputs = [
            str(file_path),
            b"raw bytes",
            FileBytes(data=b"source", filename="source.txt"),
        ]

        result = normalize_input_files(inputs)

        assert len(result) == 3
        assert "path.txt" in result
        assert "file_1" in result
        assert "source.txt" in result

    def test_empty_input(self) -> None:
        """Test normalizing empty input list."""
        result = normalize_input_files([])
        assert result == {}


class TestGenericFile:
    """Tests for the generic File class with auto-detection."""

    def test_file_from_text_bytes(self) -> None:
        """Test File creation from text bytes auto-detects content type."""
        f = File(source=b"Hello, World!")

        assert f.content_type == "text/plain"
        assert f.read() == b"Hello, World!"

    def test_file_from_png_bytes(self) -> None:
        """Test File creation from PNG bytes auto-detects image type."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        f = File(source=png_bytes)

        assert f.content_type == "image/png"

    def test_file_from_pdf_bytes(self) -> None:
        """Test File creation from PDF bytes auto-detects PDF type."""
        f = File(source=b"%PDF-1.4 content")

        assert f.content_type == "application/pdf"

    def test_file_from_path(self, tmp_path: Path) -> None:
        """Test File creation from path string."""
        file_path = tmp_path / "document.txt"
        file_path.write_text("file content")

        f = File(source=str(file_path))

        assert f.filename == "document.txt"
        assert f.read() == b"file content"
        assert f.content_type == "text/plain"

    def test_file_from_path_object(self, tmp_path: Path) -> None:
        """Test File creation from Path object."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("path object content")

        f = File(source=file_path)

        assert f.filename == "data.txt"
        assert f.read_text() == "path object content"

    def test_file_read_text(self) -> None:
        """Test File.read_text method."""
        f = File(source=b"Text content here")

        assert f.read_text() == "Text content here"

    def test_file_dict_unpacking(self, tmp_path: Path) -> None:
        """Test File supports ** unpacking syntax."""
        file_path = tmp_path / "report.txt"
        file_path.write_text("report content")

        f = File(source=str(file_path))
        result = {**f}

        assert "report" in result
        assert result["report"] is f

    def test_file_dict_unpacking_no_filename(self) -> None:
        """Test File dict unpacking with bytes (no filename)."""
        f = File(source=b"content")
        result = {**f}

        assert "file" in result

    def test_file_keys_method(self, tmp_path: Path) -> None:
        """Test File keys() method."""
        file_path = tmp_path / "chart.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        f = File(source=str(file_path))

        assert f.keys() == ["chart"]

    def test_file_getitem(self, tmp_path: Path) -> None:
        """Test File __getitem__ with valid key."""
        file_path = tmp_path / "image.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        f = File(source=str(file_path))

        assert f["image"] is f

    def test_file_getitem_invalid_key(self, tmp_path: Path) -> None:
        """Test File __getitem__ with invalid key raises KeyError."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("content")

        f = File(source=str(file_path))

        with pytest.raises(KeyError):
            _ = f["wrong"]

    def test_file_with_stream(self) -> None:
        """Test File creation from stream."""
        stream = io.BytesIO(b"stream content")

        f = File(source=stream)

        assert f.read() == b"stream content"
        assert f.content_type == "text/plain"

    def test_file_default_mode(self) -> None:
        """Test File has default mode of 'auto'."""
        f = File(source=b"content")

        assert f.mode == "auto"

    def test_file_custom_mode(self) -> None:
        """Test File with custom mode mode."""
        f = File(source=b"content", mode="strict")

        assert f.mode == "strict"

    def test_file_chunk_mode(self) -> None:
        """Test File with chunk mode mode."""
        f = File(source=b"content", mode="chunk")

        assert f.mode == "chunk"

    def test_image_file_with_mode(self) -> None:
        """Test ImageFile with custom mode."""
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img = ImageFile(source=png_bytes, mode="strict")

        assert img.mode == "strict"
        assert img.content_type == "image/png"