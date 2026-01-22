"""Tests for FileProcessor class."""

from crewai_files import FileBytes, ImageFile
from crewai_files.processing.constraints import (
    ANTHROPIC_CONSTRAINTS,
    ImageConstraints,
    ProviderConstraints,
)
from crewai_files.processing.enums import FileHandling
from crewai_files.processing.exceptions import (
    FileTooLargeError,
)
from crewai_files.processing.processor import FileProcessor
import pytest


# Minimal valid PNG: 8x8 pixel RGB image (valid for PIL)
MINIMAL_PNG = bytes(
    [
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,
        0x00,
        0x00,
        0x00,
        0x08,
        0x00,
        0x00,
        0x00,
        0x08,
        0x08,
        0x02,
        0x00,
        0x00,
        0x00,
        0x4B,
        0x6D,
        0x29,
        0xDC,
        0x00,
        0x00,
        0x00,
        0x12,
        0x49,
        0x44,
        0x41,
        0x54,
        0x78,
        0x9C,
        0x63,
        0xFC,
        0xCF,
        0x80,
        0x1D,
        0x30,
        0xE1,
        0x10,
        0x1F,
        0xA4,
        0x12,
        0x00,
        0xCD,
        0x41,
        0x01,
        0x0F,
        0xE8,
        0x41,
        0xE2,
        0x6F,
        0x00,
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,
        0x42,
        0x60,
        0x82,
    ]
)

# Minimal valid PDF
MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj "
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000052 00000 n \n0000000101 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
)


class TestFileProcessorInit:
    """Tests for FileProcessor initialization."""

    def test_init_with_constraints(self):
        """Test initialization with ProviderConstraints."""
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)

        assert processor.constraints == ANTHROPIC_CONSTRAINTS

    def test_init_with_provider_string(self):
        """Test initialization with provider name string."""
        processor = FileProcessor(constraints="anthropic")

        assert processor.constraints == ANTHROPIC_CONSTRAINTS

    def test_init_with_unknown_provider(self):
        """Test initialization with unknown provider sets constraints to None."""
        processor = FileProcessor(constraints="unknown")

        assert processor.constraints is None

    def test_init_with_none_constraints(self):
        """Test initialization with None constraints."""
        processor = FileProcessor(constraints=None)

        assert processor.constraints is None


class TestFileProcessorValidate:
    """Tests for FileProcessor.validate method."""

    def test_validate_valid_file(self):
        """Test validating a valid file returns no errors."""
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        errors = processor.validate(file)

        assert len(errors) == 0

    def test_validate_without_constraints(self):
        """Test validating without constraints returns empty list."""
        processor = FileProcessor(constraints=None)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        errors = processor.validate(file)

        assert len(errors) == 0

    def test_validate_strict_raises_on_error(self):
        """Test STRICT mode raises on validation error."""
        constraints = ProviderConstraints(
            name="test",
            image=ImageConstraints(max_size_bytes=10),
        )
        processor = FileProcessor(constraints=constraints)
        # Set mode to strict on the file
        file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="strict"
        )

        with pytest.raises(FileTooLargeError):
            processor.validate(file)


class TestFileProcessorProcess:
    """Tests for FileProcessor.process method."""

    def test_process_valid_file(self):
        """Test processing a valid file returns it unchanged."""
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        result = processor.process(file)

        assert result == file

    def test_process_without_constraints(self):
        """Test processing without constraints returns file unchanged."""
        processor = FileProcessor(constraints=None)
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))

        result = processor.process(file)

        assert result == file

    def test_process_strict_raises_on_error(self):
        """Test STRICT mode raises on processing error."""
        constraints = ProviderConstraints(
            name="test",
            image=ImageConstraints(max_size_bytes=10),
        )
        processor = FileProcessor(constraints=constraints)
        # Set mode to strict on the file
        file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="strict"
        )

        with pytest.raises(FileTooLargeError):
            processor.process(file)

    def test_process_warn_returns_file(self):
        """Test WARN mode returns file with warning."""
        constraints = ProviderConstraints(
            name="test",
            image=ImageConstraints(max_size_bytes=10),
        )
        processor = FileProcessor(constraints=constraints)
        # Set mode to warn on the file
        file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="warn"
        )

        result = processor.process(file)

        assert result == file


class TestFileProcessorProcessFiles:
    """Tests for FileProcessor.process_files method."""

    def test_process_files_multiple(self):
        """Test processing multiple files."""
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)
        files = {
            "image1": ImageFile(
                source=FileBytes(data=MINIMAL_PNG, filename="test1.png")
            ),
            "image2": ImageFile(
                source=FileBytes(data=MINIMAL_PNG, filename="test2.png")
            ),
        }

        result = processor.process_files(files)

        assert len(result) == 2
        assert "image1" in result
        assert "image2" in result

    def test_process_files_empty(self):
        """Test processing empty files dict."""
        processor = FileProcessor(constraints=ANTHROPIC_CONSTRAINTS)

        result = processor.process_files({})

        assert result == {}


class TestFileHandlingEnum:
    """Tests for FileHandling enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert FileHandling.STRICT.value == "strict"
        assert FileHandling.AUTO.value == "auto"
        assert FileHandling.WARN.value == "warn"
        assert FileHandling.CHUNK.value == "chunk"


class TestFileProcessorPerFileMode:
    """Tests for per-file mode handling."""

    def test_file_default_mode_is_auto(self):
        """Test that files default to auto mode."""
        file = ImageFile(source=FileBytes(data=MINIMAL_PNG, filename="test.png"))
        assert file.mode == "auto"

    def test_file_custom_mode(self):
        """Test setting custom mode on file."""
        file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="strict"
        )
        assert file.mode == "strict"

    def test_processor_respects_file_mode(self):
        """Test processor uses each file's mode setting."""
        constraints = ProviderConstraints(
            name="test",
            image=ImageConstraints(max_size_bytes=10),
        )
        processor = FileProcessor(constraints=constraints)

        # File with strict mode should raise
        strict_file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="strict"
        )
        with pytest.raises(FileTooLargeError):
            processor.process(strict_file)

        # File with warn mode should not raise
        warn_file = ImageFile(
            source=FileBytes(data=MINIMAL_PNG, filename="test.png"), mode="warn"
        )
        result = processor.process(warn_file)
        assert result == warn_file
