"""Tests for file validators."""

from unittest.mock import patch

from crewai_files import AudioFile, FileBytes, ImageFile, PDFFile, TextFile, VideoFile
from crewai_files.processing.constraints import (
    ANTHROPIC_CONSTRAINTS,
    AudioConstraints,
    ImageConstraints,
    PDFConstraints,
    ProviderConstraints,
    VideoConstraints,
)
from crewai_files.processing.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from crewai_files.processing.validators import (
    _get_audio_duration,
    _get_video_duration,
    validate_audio,
    validate_file,
    validate_image,
    validate_pdf,
    validate_text,
    validate_video,
)
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


# Minimal audio bytes for testing (not a valid audio file, used for mocked tests)
MINIMAL_AUDIO = b"\x00" * 100

# Minimal video bytes for testing (not a valid video file, used for mocked tests)
MINIMAL_VIDEO = b"\x00" * 100

# Fallback content type when python-magic cannot detect
FALLBACK_CONTENT_TYPE = "application/octet-stream"


class TestValidateAudio:
    """Tests for validate_audio function and audio duration validation."""

    def test_validate_valid_audio(self):
        """Test validating a valid audio file within constraints."""
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        errors = validate_audio(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_audio_too_large(self):
        """Test validating an audio file that exceeds size limit."""
        constraints = AudioConstraints(
            max_size_bytes=10,  # Very small limit
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        with pytest.raises(FileTooLargeError) as exc_info:
            validate_audio(file, constraints)

        assert "exceeds" in str(exc_info.value)
        assert exc_info.value.file_name == "test.mp3"

    def test_validate_audio_unsupported_format(self):
        """Test validating an audio file with unsupported format."""
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("audio/wav",),  # Only WAV
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            validate_audio(file, constraints)

        assert "not supported" in str(exc_info.value)

    @patch("crewai_files.processing.validators._get_audio_duration")
    def test_validate_audio_duration_passes(self, mock_get_duration):
        """Test validating audio when duration is under limit."""
        mock_get_duration.return_value = 30.0
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        errors = validate_audio(file, constraints, raise_on_error=False)

        assert len(errors) == 0
        mock_get_duration.assert_called_once()

    @patch("crewai_files.processing.validators._get_audio_duration")
    def test_validate_audio_duration_fails(self, mock_get_duration):
        """Test validating audio when duration exceeds limit."""
        mock_get_duration.return_value = 120.5
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        with pytest.raises(FileValidationError) as exc_info:
            validate_audio(file, constraints)

        assert "duration" in str(exc_info.value).lower()
        assert "120.5s" in str(exc_info.value)
        assert "60s" in str(exc_info.value)

    @patch("crewai_files.processing.validators._get_audio_duration")
    def test_validate_audio_duration_no_raise(self, mock_get_duration):
        """Test audio duration validation with raise_on_error=False."""
        mock_get_duration.return_value = 120.5
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        errors = validate_audio(file, constraints, raise_on_error=False)

        assert len(errors) == 1
        assert "duration" in errors[0].lower()

    @patch("crewai_files.processing.validators._get_audio_duration")
    def test_validate_audio_duration_none_skips(self, mock_get_duration):
        """Test that duration validation is skipped when max_duration_seconds is None."""
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=None,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        errors = validate_audio(file, constraints, raise_on_error=False)

        assert len(errors) == 0
        mock_get_duration.assert_not_called()

    @patch("crewai_files.processing.validators._get_audio_duration")
    def test_validate_audio_duration_detection_returns_none(self, mock_get_duration):
        """Test that validation passes when duration detection returns None."""
        mock_get_duration.return_value = None
        constraints = AudioConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("audio/mp3", "audio/mpeg", FALLBACK_CONTENT_TYPE),
        )
        file = AudioFile(source=FileBytes(data=MINIMAL_AUDIO, filename="test.mp3"))

        errors = validate_audio(file, constraints, raise_on_error=False)

        assert len(errors) == 0


class TestValidateVideo:
    """Tests for validate_video function and video duration validation."""

    def test_validate_valid_video(self):
        """Test validating a valid video file within constraints."""
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_video_too_large(self):
        """Test validating a video file that exceeds size limit."""
        constraints = VideoConstraints(
            max_size_bytes=10,  # Very small limit
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        with pytest.raises(FileTooLargeError) as exc_info:
            validate_video(file, constraints)

        assert "exceeds" in str(exc_info.value)
        assert exc_info.value.file_name == "test.mp4"

    def test_validate_video_unsupported_format(self):
        """Test validating a video file with unsupported format."""
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            supported_formats=("video/webm",),  # Only WebM
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            validate_video(file, constraints)

        assert "not supported" in str(exc_info.value)

    @patch("crewai_files.processing.validators._get_video_duration")
    def test_validate_video_duration_passes(self, mock_get_duration):
        """Test validating video when duration is under limit."""
        mock_get_duration.return_value = 30.0
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 0
        mock_get_duration.assert_called_once()

    @patch("crewai_files.processing.validators._get_video_duration")
    def test_validate_video_duration_fails(self, mock_get_duration):
        """Test validating video when duration exceeds limit."""
        mock_get_duration.return_value = 180.0
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        with pytest.raises(FileValidationError) as exc_info:
            validate_video(file, constraints)

        assert "duration" in str(exc_info.value).lower()
        assert "180.0s" in str(exc_info.value)
        assert "60s" in str(exc_info.value)

    @patch("crewai_files.processing.validators._get_video_duration")
    def test_validate_video_duration_no_raise(self, mock_get_duration):
        """Test video duration validation with raise_on_error=False."""
        mock_get_duration.return_value = 180.0
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 1
        assert "duration" in errors[0].lower()

    @patch("crewai_files.processing.validators._get_video_duration")
    def test_validate_video_duration_none_skips(self, mock_get_duration):
        """Test that duration validation is skipped when max_duration_seconds is None."""
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=None,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 0
        mock_get_duration.assert_not_called()

    @patch("crewai_files.processing.validators._get_video_duration")
    def test_validate_video_duration_detection_returns_none(self, mock_get_duration):
        """Test that validation passes when duration detection returns None."""
        mock_get_duration.return_value = None
        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("video/mp4", FALLBACK_CONTENT_TYPE),
        )
        file = VideoFile(source=FileBytes(data=MINIMAL_VIDEO, filename="test.mp4"))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 0


class TestGetAudioDuration:
    """Tests for _get_audio_duration helper function."""

    def test_get_audio_duration_corrupt_file(self):
        """Test handling of corrupt audio data."""
        corrupt_data = b"not valid audio data at all"
        result = _get_audio_duration(corrupt_data)

        assert result is None


class TestGetVideoDuration:
    """Tests for _get_video_duration helper function."""

    def test_get_video_duration_corrupt_file(self):
        """Test handling of corrupt video data."""
        corrupt_data = b"not valid video data at all"
        result = _get_video_duration(corrupt_data)

        assert result is None


class TestRealVideoFile:
    """Tests using real video fixture file."""

    @pytest.fixture
    def sample_video_path(self):
        """Path to sample video fixture."""
        from pathlib import Path

        path = Path(__file__).parent.parent.parent / "fixtures" / "sample_video.mp4"
        if not path.exists():
            pytest.skip("sample_video.mp4 fixture not found")
        return path

    @pytest.fixture
    def sample_video_content(self, sample_video_path):
        """Read sample video content."""
        return sample_video_path.read_bytes()

    def test_get_video_duration_real_file(self, sample_video_content):
        """Test duration detection with real video file."""
        try:
            import av  # noqa: F401
        except ImportError:
            pytest.skip("PyAV not installed")

        duration = _get_video_duration(sample_video_content, "video/mp4")

        assert duration is not None
        assert 4.5 <= duration <= 5.5  # ~5 seconds with tolerance

    def test_get_video_duration_real_file_no_format_hint(self, sample_video_content):
        """Test duration detection without format hint."""
        try:
            import av  # noqa: F401
        except ImportError:
            pytest.skip("PyAV not installed")

        duration = _get_video_duration(sample_video_content)

        assert duration is not None
        assert 4.5 <= duration <= 5.5

    def test_validate_video_real_file_passes(self, sample_video_path):
        """Test validating real video file within constraints."""
        try:
            import av  # noqa: F401
        except ImportError:
            pytest.skip("PyAV not installed")

        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=60,
            supported_formats=("video/mp4",),
        )
        file = VideoFile(source=str(sample_video_path))

        errors = validate_video(file, constraints, raise_on_error=False)

        assert len(errors) == 0

    def test_validate_video_real_file_duration_exceeded(self, sample_video_path):
        """Test validating real video file that exceeds duration limit."""
        try:
            import av  # noqa: F401
        except ImportError:
            pytest.skip("PyAV not installed")

        constraints = VideoConstraints(
            max_size_bytes=10 * 1024 * 1024,
            max_duration_seconds=2,  # Video is ~5 seconds
            supported_formats=("video/mp4",),
        )
        file = VideoFile(source=str(sample_video_path))

        with pytest.raises(FileValidationError) as exc_info:
            validate_video(file, constraints)

        assert "duration" in str(exc_info.value).lower()
        assert "2s" in str(exc_info.value)
