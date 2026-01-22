"""Tests for file validators."""

import pytest

from crewai.files import FileBytes, ImageFile, PDFFile, TextFile
from crewai.files.processing.constraints import (
    ANTHROPIC_CONSTRAINTS,
    ImageConstraints,
    PDFConstraints,
    ProviderConstraints,
)
from crewai.files.processing.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from crewai.files.processing.validators import (
    validate_file,
    validate_image,
    validate_pdf,
    validate_text,
)


# Minimal valid PNG: 8x8 pixel RGB image (valid for PIL)
MINIMAL_PNG = bytes([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x4b, 0x6d, 0x29, 0xdc, 0x00, 0x00, 0x00,
    0x12, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xfc, 0xcf, 0x80, 0x1d,
    0x30, 0xe1, 0x10, 0x1f, 0xa4, 0x12, 0x00, 0xcd, 0x41, 0x01, 0x0f, 0xe8,
    0x41, 0xe2, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
    0x42, 0x60, 0x82,
])

# Minimal valid PDF
MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj "
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000052 00000 n \n0000000101 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
)


class TestValidateImage:
    """Tests for validate_image function."""

    def test_validate_valid_image(self):
        """Test validating a valid image within constraints."""
        constraints = ImageConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("image/png",),
        )
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        errors = validate_image(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_image_too_large(self):
        """Test validating an image that exceeds size limit."""
        constraints = ImageConstraints(
            max_size_bytes=10,  # Very small limit
            supported_formats=("image/png",),
        )
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        with pytest.raises(FileTooLargeError) as exc_info:
            validate_image(file, constraints)

        assert "exceeds" in str(exc_info.value)
        assert exc_info.value.file_name == "test.png"

    def test_validate_image_unsupported_format(self):
        """Test validating an image with unsupported format."""
        constraints = ImageConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("image/jpeg",),  # Only JPEG
        )
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            validate_image(file, constraints)

        assert "not supported" in str(exc_info.value)

    def test_validate_image_no_raise(self):
        """Test validating with raise_on_error=False returns errors list."""
        constraints = ImageConstraints(
            max_size_bytes=10,
            supported_formats=("image/jpeg",),
        )
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        errors = validate_image(file, constraints, raise_on_error=False)

        assert len(errors) == 2  # Size error and format error


class TestValidatePDF:
    """Tests for validate_pdf function."""

    def test_validate_valid_pdf(self):
        """Test validating a valid PDF within constraints."""
        constraints = PDFConstraints(
            max_size_bytes=10 * 1024 * 1024,
        )
        file = PDFFile(source=FileBytes(data=MINIMAL_PDF, filename="test.pdf"))

        errors = validate_pdf(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_pdf_too_large(self):
        """Test validating a PDF that exceeds size limit."""
        constraints = PDFConstraints(
            max_size_bytes=10,  # Very small limit
        )
        file = PDFFile(source=FileBytes(data=MINIMAL_PDF, filename="test.pdf"))

        with pytest.raises(FileTooLargeError) as exc_info:
            validate_pdf(file, constraints)

        assert "exceeds" in str(exc_info.value)


class TestValidateText:
    """Tests for validate_text function."""

    def test_validate_valid_text(self):
        """Test validating a valid text file."""
        constraints = ProviderConstraints(
            name="test",
            general_max_size_bytes=10 * 1024 * 1024,
        )
        file = TextFile(source=FileBytes(data=b"Hello, World!", filename="test.txt"))

        errors = validate_text(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_text_too_large(self):
        """Test validating text that exceeds size limit."""
        constraints = ProviderConstraints(
            name="test",
            general_max_size_bytes=5,
        )
        file = TextFile(source=FileBytes(data=b"Hello, World!", filename="test.txt"))

        with pytest.raises(FileTooLargeError):
            validate_text(file, constraints)

    def test_validate_text_no_limit(self):
        """Test validating text with no size limit."""
        constraints = ProviderConstraints(name="test")
        file = TextFile(source=FileBytes(data=b"Hello, World!", filename="test.txt"))

        errors = validate_text(file, constraints, raise_on_error=False)

        assert len(errors) == 0


class TestValidateFile:
    """Tests for validate_file function."""

    def test_validate_file_dispatches_to_image(self):
        """Test validate_file dispatches to image validator."""
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        errors = validate_file(file, ANTHROPIC_CONSTRAINTS, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_file_dispatches_to_pdf(self):
        """Test validate_file dispatches to PDF validator."""
        file = PDFFile(source=FileBytes(data=MINIMAL_PDF, filename="test.pdf"))

        errors = validate_file(file, ANTHROPIC_CONSTRAINTS, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_file_unsupported_type(self):
        """Test validating a file type not supported by provider."""
        constraints = ProviderConstraints(
            name="test",
            image=None,  # No image support
        )
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            validate_file(file, constraints)

        assert "does not support images" in str(exc_info.value)

    def test_validate_file_pdf_not_supported(self):
        """Test validating PDF when provider doesn't support it."""
        constraints = ProviderConstraints(
            name="test",
            pdf=None,  # No PDF support
        )
        file = PDFFile(source=FileBytes(data=MINIMAL_PDF, filename="test.pdf"))

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            validate_file(file, constraints)

        assert "does not support PDFs" in str(exc_info.value)
