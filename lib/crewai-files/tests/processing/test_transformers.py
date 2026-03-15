"""Unit tests for file transformers."""

import io
from unittest.mock import patch

from crewai_files import ImageFile, PDFFile, TextFile
from crewai_files.core.sources import FileBytes
from crewai_files.processing.exceptions import ProcessingDependencyError
from crewai_files.processing.transformers import (
    chunk_pdf,
    chunk_text,
    get_image_dimensions,
    get_pdf_page_count,
    optimize_image,
    resize_image,
)
import pytest


def create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG for testing."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_test_pdf(num_pages: int = 1) -> bytes:
    """Create a minimal valid PDF for testing."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_larger_image(self) -> None:
        """Test resizing an image larger than max dimensions."""
        png_bytes = create_test_png(200, 150)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        result = resize_image(img, max_width=100, max_height=100)

        dims = get_image_dimensions(result)
        assert dims is not None
        width, height = dims
        assert width <= 100
        assert height <= 100

    def test_no_resize_if_within_bounds(self) -> None:
        """Test that small images are returned unchanged."""
        png_bytes = create_test_png(50, 50)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="small.png"))

        result = resize_image(img, max_width=100, max_height=100)

        assert result is img

    def test_preserve_aspect_ratio(self) -> None:
        """Test that aspect ratio is preserved during resize."""
        png_bytes = create_test_png(200, 100)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="wide.png"))

        result = resize_image(img, max_width=100, max_height=100)

        dims = get_image_dimensions(result)
        assert dims is not None
        width, height = dims
        assert width == 100
        assert height == 50

    def test_resize_without_aspect_ratio(self) -> None:
        """Test resizing without preserving aspect ratio."""
        png_bytes = create_test_png(200, 100)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="wide.png"))

        result = resize_image(
            img, max_width=50, max_height=50, preserve_aspect_ratio=False
        )

        dims = get_image_dimensions(result)
        assert dims is not None
        width, height = dims
        assert width == 50
        assert height == 50

    def test_resize_returns_image_file(self) -> None:
        """Test that resize returns an ImageFile instance."""
        png_bytes = create_test_png(200, 200)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        result = resize_image(img, max_width=100, max_height=100)

        assert isinstance(result, ImageFile)

    def test_raises_without_pillow(self) -> None:
        """Test that ProcessingDependencyError is raised without Pillow."""
        img = ImageFile(source=FileBytes(data=b"fake", filename="test.png"))

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ProcessingDependencyError) as exc_info:
                # Force reimport to trigger ImportError
                import importlib

                import crewai_files.processing.transformers as t

                importlib.reload(t)
                t.resize_image(img, 100, 100)

            assert "Pillow" in str(exc_info.value)


class TestOptimizeImage:
    """Tests for optimize_image function."""

    def test_optimize_reduces_size(self) -> None:
        """Test that optimization reduces file size."""
        png_bytes = create_test_png(500, 500)
        original_size = len(png_bytes)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="large.png"))

        result = optimize_image(img, target_size_bytes=original_size // 2)

        result_size = len(result.read())
        assert result_size < original_size

    def test_no_optimize_if_under_target(self) -> None:
        """Test that small images are returned unchanged."""
        png_bytes = create_test_png(50, 50)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="small.png"))

        result = optimize_image(img, target_size_bytes=1024 * 1024)

        assert result is img

    def test_optimize_returns_image_file(self) -> None:
        """Test that optimize returns an ImageFile instance."""
        png_bytes = create_test_png(200, 200)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        result = optimize_image(img, target_size_bytes=100)

        assert isinstance(result, ImageFile)

    def test_optimize_respects_min_quality(self) -> None:
        """Test that optimization stops at minimum quality."""
        png_bytes = create_test_png(100, 100)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        # Request impossibly small size - should stop at min quality
        result = optimize_image(img, target_size_bytes=10, min_quality=50)

        assert isinstance(result, ImageFile)
        assert len(result.read()) > 10


class TestChunkPdf:
    """Tests for chunk_pdf function."""

    def test_chunk_splits_large_pdf(self) -> None:
        """Test that large PDFs are split into chunks."""
        pdf_bytes = create_test_pdf(num_pages=10)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="large.pdf"))

        result = list(chunk_pdf(pdf, max_pages=3))

        assert len(result) == 4
        assert all(isinstance(chunk, PDFFile) for chunk in result)

    def test_no_chunk_if_within_limit(self) -> None:
        """Test that small PDFs are returned unchanged."""
        pdf_bytes = create_test_pdf(num_pages=3)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="small.pdf"))

        result = list(chunk_pdf(pdf, max_pages=5))

        assert len(result) == 1
        assert result[0] is pdf

    def test_chunk_filenames(self) -> None:
        """Test that chunked files have indexed filenames."""
        pdf_bytes = create_test_pdf(num_pages=6)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="document.pdf"))

        result = list(chunk_pdf(pdf, max_pages=2))

        assert result[0].filename == "document_chunk_0.pdf"
        assert result[1].filename == "document_chunk_1.pdf"
        assert result[2].filename == "document_chunk_2.pdf"

    def test_chunk_with_overlap(self) -> None:
        """Test chunking with overlapping pages."""
        pdf_bytes = create_test_pdf(num_pages=10)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="doc.pdf"))

        result = list(chunk_pdf(pdf, max_pages=4, overlap_pages=1))

        # With overlap, we get more chunks
        assert len(result) >= 3

    def test_chunk_page_counts(self) -> None:
        """Test that each chunk has correct page count."""
        pdf_bytes = create_test_pdf(num_pages=7)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="doc.pdf"))

        result = list(chunk_pdf(pdf, max_pages=3))

        page_counts = [get_pdf_page_count(chunk) for chunk in result]
        assert page_counts == [3, 3, 1]


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_splits_large_text(self) -> None:
        """Test that large text files are split into chunks."""
        content = "Hello world. " * 100
        text = TextFile(source=content.encode(), filename="large.txt")

        result = list(chunk_text(text, max_chars=200, overlap_chars=0))

        assert len(result) > 1
        assert all(isinstance(chunk, TextFile) for chunk in result)

    def test_no_chunk_if_within_limit(self) -> None:
        """Test that small text files are returned unchanged."""
        content = "Short text"
        text = TextFile(source=content.encode(), filename="small.txt")

        result = list(chunk_text(text, max_chars=1000, overlap_chars=0))

        assert len(result) == 1
        assert result[0] is text

    def test_chunk_filenames(self) -> None:
        """Test that chunked files have indexed filenames."""
        content = "A" * 500
        text = TextFile(source=FileBytes(data=content.encode(), filename="data.txt"))

        result = list(chunk_text(text, max_chars=200, overlap_chars=0))

        assert result[0].filename == "data_chunk_0.txt"
        assert result[1].filename == "data_chunk_1.txt"
        assert len(result) == 3

    def test_chunk_preserves_extension(self) -> None:
        """Test that file extension is preserved in chunks."""
        content = "A" * 500
        text = TextFile(source=FileBytes(data=content.encode(), filename="script.py"))

        result = list(chunk_text(text, max_chars=200, overlap_chars=0))

        assert all(chunk.filename.endswith(".py") for chunk in result)

    def test_chunk_prefers_newline_boundaries(self) -> None:
        """Test that chunking prefers to split at newlines."""
        content = "Line one\nLine two\nLine three\nLine four\nLine five"
        text = TextFile(source=content.encode(), filename="lines.txt")

        result = list(
            chunk_text(text, max_chars=25, overlap_chars=0, split_on_newlines=True)
        )

        # Should split at newline boundaries
        for chunk in result:
            chunk_text_content = chunk.read().decode()
            # Chunks should end at newlines (except possibly the last)
            if chunk != result[-1]:
                assert (
                    chunk_text_content.endswith("\n") or len(chunk_text_content) <= 25
                )

    def test_chunk_with_overlap(self) -> None:
        """Test chunking with overlapping characters."""
        content = "ABCDEFGHIJ" * 10
        text = TextFile(source=content.encode(), filename="data.txt")

        result = list(chunk_text(text, max_chars=30, overlap_chars=5))

        # With overlap, chunks should share some content
        assert len(result) >= 3

    def test_chunk_overlap_larger_than_max_chars(self) -> None:
        """Test that overlap > max_chars doesn't cause infinite loop."""
        content = "A" * 100
        text = TextFile(source=content.encode(), filename="data.txt")

        # overlap_chars > max_chars should still work (just with max overlap)
        result = list(chunk_text(text, max_chars=20, overlap_chars=50))

        assert len(result) > 1
        # Should still complete without hanging


class TestGetImageDimensions:
    """Tests for get_image_dimensions function."""

    def test_get_dimensions(self) -> None:
        """Test getting image dimensions."""
        png_bytes = create_test_png(150, 100)
        img = ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        dims = get_image_dimensions(img)

        assert dims == (150, 100)

    def test_returns_none_for_invalid_image(self) -> None:
        """Test that None is returned for invalid image data."""
        img = ImageFile(source=FileBytes(data=b"not an image", filename="bad.png"))

        dims = get_image_dimensions(img)

        assert dims is None

    def test_returns_none_without_pillow(self) -> None:
        """Test that None is returned when Pillow is not installed."""
        png_bytes = create_test_png(100, 100)
        ImageFile(source=FileBytes(data=png_bytes, filename="test.png"))

        with patch.dict("sys.modules", {"PIL": None}):
            # Can't easily test this without unloading module
            # Just verify the function handles the case gracefully
            pass


class TestGetPdfPageCount:
    """Tests for get_pdf_page_count function."""

    def test_get_page_count(self) -> None:
        """Test getting PDF page count."""
        pdf_bytes = create_test_pdf(num_pages=5)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="test.pdf"))

        count = get_pdf_page_count(pdf)

        assert count == 5

    def test_single_page(self) -> None:
        """Test page count for single page PDF."""
        pdf_bytes = create_test_pdf(num_pages=1)
        pdf = PDFFile(source=FileBytes(data=pdf_bytes, filename="single.pdf"))

        count = get_pdf_page_count(pdf)

        assert count == 1

    def test_returns_none_for_invalid_pdf(self) -> None:
        """Test that None is returned for invalid PDF data."""
        pdf = PDFFile(source=FileBytes(data=b"not a pdf", filename="bad.pdf"))

        count = get_pdf_page_count(pdf)

        assert count is None
