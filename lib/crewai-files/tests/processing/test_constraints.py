"""Tests for provider constraints."""

from crewai_files.processing.constraints import (
    ANTHROPIC_CONSTRAINTS,
    BEDROCK_CONSTRAINTS,
    GEMINI_CONSTRAINTS,
    OPENAI_CONSTRAINTS,
    AudioConstraints,
    ImageConstraints,
    PDFConstraints,
    ProviderConstraints,
    VideoConstraints,
    get_constraints_for_provider,
)
import pytest


class TestImageConstraints:
    """Tests for ImageConstraints dataclass."""

    def test_image_constraints_creation(self):
        """Test creating image constraints with all fields."""
        constraints = ImageConstraints(
            max_size_bytes=5 * 1024 * 1024,
            max_width=8000,
            max_height=8000,
            max_images_per_request=10,
        )

        assert constraints.max_size_bytes == 5 * 1024 * 1024
        assert constraints.max_width == 8000
        assert constraints.max_height == 8000
        assert constraints.max_images_per_request == 10

    def test_image_constraints_defaults(self):
        """Test image constraints with default values."""
        constraints = ImageConstraints(max_size_bytes=1000)

        assert constraints.max_size_bytes == 1000
        assert constraints.max_width is None
        assert constraints.max_height is None
        assert constraints.max_images_per_request is None
        assert "image/png" in constraints.supported_formats

    def test_image_constraints_frozen(self):
        """Test that image constraints are immutable."""
        constraints = ImageConstraints(max_size_bytes=1000)

        with pytest.raises(Exception):
            constraints.max_size_bytes = 2000


class TestPDFConstraints:
    """Tests for PDFConstraints dataclass."""

    def test_pdf_constraints_creation(self):
        """Test creating PDF constraints."""
        constraints = PDFConstraints(
            max_size_bytes=30 * 1024 * 1024,
            max_pages=100,
        )

        assert constraints.max_size_bytes == 30 * 1024 * 1024
        assert constraints.max_pages == 100

    def test_pdf_constraints_defaults(self):
        """Test PDF constraints with default values."""
        constraints = PDFConstraints(max_size_bytes=1000)

        assert constraints.max_size_bytes == 1000
        assert constraints.max_pages is None


class TestAudioConstraints:
    """Tests for AudioConstraints dataclass."""

    def test_audio_constraints_creation(self):
        """Test creating audio constraints."""
        constraints = AudioConstraints(
            max_size_bytes=100 * 1024 * 1024,
            max_duration_seconds=3600,
        )

        assert constraints.max_size_bytes == 100 * 1024 * 1024
        assert constraints.max_duration_seconds == 3600
        assert "audio/mp3" in constraints.supported_formats


class TestVideoConstraints:
    """Tests for VideoConstraints dataclass."""

    def test_video_constraints_creation(self):
        """Test creating video constraints."""
        constraints = VideoConstraints(
            max_size_bytes=2 * 1024 * 1024 * 1024,
            max_duration_seconds=7200,
        )

        assert constraints.max_size_bytes == 2 * 1024 * 1024 * 1024
        assert constraints.max_duration_seconds == 7200
        assert "video/mp4" in constraints.supported_formats


class TestProviderConstraints:
    """Tests for ProviderConstraints dataclass."""

    def test_provider_constraints_creation(self):
        """Test creating full provider constraints."""
        constraints = ProviderConstraints(
            name="test-provider",
            image=ImageConstraints(max_size_bytes=5 * 1024 * 1024),
            pdf=PDFConstraints(max_size_bytes=30 * 1024 * 1024),
            supports_file_upload=True,
            file_upload_threshold_bytes=10 * 1024 * 1024,
        )

        assert constraints.name == "test-provider"
        assert constraints.image is not None
        assert constraints.pdf is not None
        assert constraints.supports_file_upload is True

    def test_provider_constraints_defaults(self):
        """Test provider constraints with default values."""
        constraints = ProviderConstraints(name="test")

        assert constraints.name == "test"
        assert constraints.image is None
        assert constraints.pdf is None
        assert constraints.audio is None
        assert constraints.video is None
        assert constraints.supports_file_upload is False


class TestPredefinedConstraints:
    """Tests for predefined provider constraints."""

    def test_anthropic_constraints(self):
        """Test Anthropic constraints are properly defined."""
        assert ANTHROPIC_CONSTRAINTS.name == "anthropic"
        assert ANTHROPIC_CONSTRAINTS.image is not None
        assert ANTHROPIC_CONSTRAINTS.image.max_size_bytes == 5 * 1024 * 1024
        assert ANTHROPIC_CONSTRAINTS.image.max_width == 8000
        assert ANTHROPIC_CONSTRAINTS.pdf is not None
        assert ANTHROPIC_CONSTRAINTS.pdf.max_pages == 100
        assert ANTHROPIC_CONSTRAINTS.supports_file_upload is True

    def test_openai_constraints(self):
        """Test OpenAI constraints are properly defined."""
        assert OPENAI_CONSTRAINTS.name == "openai"
        assert OPENAI_CONSTRAINTS.image is not None
        assert OPENAI_CONSTRAINTS.image.max_size_bytes == 20 * 1024 * 1024
        assert OPENAI_CONSTRAINTS.pdf is None  # OpenAI doesn't support PDFs

    def test_gemini_constraints(self):
        """Test Gemini constraints are properly defined."""
        assert GEMINI_CONSTRAINTS.name == "gemini"
        assert GEMINI_CONSTRAINTS.image is not None
        assert GEMINI_CONSTRAINTS.pdf is not None
        assert GEMINI_CONSTRAINTS.audio is not None
        assert GEMINI_CONSTRAINTS.video is not None
        assert GEMINI_CONSTRAINTS.supports_file_upload is True

    def test_bedrock_constraints(self):
        """Test Bedrock constraints are properly defined."""
        assert BEDROCK_CONSTRAINTS.name == "bedrock"
        assert BEDROCK_CONSTRAINTS.image is not None
        assert BEDROCK_CONSTRAINTS.image.max_size_bytes == 4_608_000
        assert BEDROCK_CONSTRAINTS.pdf is not None
        assert BEDROCK_CONSTRAINTS.supports_file_upload is False


class TestGetConstraintsForProvider:
    """Tests for get_constraints_for_provider function."""

    def test_get_by_exact_name(self):
        """Test getting constraints by exact provider name."""
        result = get_constraints_for_provider("anthropic")
        assert result == ANTHROPIC_CONSTRAINTS

        result = get_constraints_for_provider("openai")
        assert result == OPENAI_CONSTRAINTS

        result = get_constraints_for_provider("gemini")
        assert result == GEMINI_CONSTRAINTS

    def test_get_by_alias(self):
        """Test getting constraints by alias name."""
        result = get_constraints_for_provider("claude")
        assert result == ANTHROPIC_CONSTRAINTS

        result = get_constraints_for_provider("gpt")
        assert result == OPENAI_CONSTRAINTS

        result = get_constraints_for_provider("google")
        assert result == GEMINI_CONSTRAINTS

    def test_get_case_insensitive(self):
        """Test case-insensitive lookup."""
        result = get_constraints_for_provider("ANTHROPIC")
        assert result == ANTHROPIC_CONSTRAINTS

        result = get_constraints_for_provider("OpenAI")
        assert result == OPENAI_CONSTRAINTS

    def test_get_with_provider_constraints_object(self):
        """Test passing ProviderConstraints object returns it unchanged."""
        custom = ProviderConstraints(name="custom")
        result = get_constraints_for_provider(custom)
        assert result is custom

    def test_get_unknown_provider(self):
        """Test unknown provider returns None."""
        result = get_constraints_for_provider("unknown-provider")
        assert result is None

    def test_get_by_partial_match(self):
        """Test partial match in provider string."""
        result = get_constraints_for_provider("claude-3-sonnet")
        assert result == ANTHROPIC_CONSTRAINTS

        result = get_constraints_for_provider("gpt-4o")
        assert result == OPENAI_CONSTRAINTS

        result = get_constraints_for_provider("gemini-pro")
        assert result == GEMINI_CONSTRAINTS
