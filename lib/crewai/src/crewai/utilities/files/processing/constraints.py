"""Provider-specific file constraints for multimodal content."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageConstraints:
    """Constraints for image files.

    Attributes:
        max_size_bytes: Maximum file size in bytes.
        max_width: Maximum image width in pixels.
        max_height: Maximum image height in pixels.
        max_images_per_request: Maximum number of images per request.
        supported_formats: Supported image MIME types.
    """

    max_size_bytes: int
    max_width: int | None = None
    max_height: int | None = None
    max_images_per_request: int | None = None
    supported_formats: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    )


@dataclass(frozen=True)
class PDFConstraints:
    """Constraints for PDF files.

    Attributes:
        max_size_bytes: Maximum file size in bytes.
        max_pages: Maximum number of pages.
    """

    max_size_bytes: int
    max_pages: int | None = None


@dataclass(frozen=True)
class AudioConstraints:
    """Constraints for audio files.

    Attributes:
        max_size_bytes: Maximum file size in bytes.
        max_duration_seconds: Maximum audio duration in seconds.
        supported_formats: Supported audio MIME types.
    """

    max_size_bytes: int
    max_duration_seconds: int | None = None
    supported_formats: tuple[str, ...] = (
        "audio/mp3",
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
        "audio/m4a",
    )


@dataclass(frozen=True)
class VideoConstraints:
    """Constraints for video files.

    Attributes:
        max_size_bytes: Maximum file size in bytes.
        max_duration_seconds: Maximum video duration in seconds.
        supported_formats: Supported video MIME types.
    """

    max_size_bytes: int
    max_duration_seconds: int | None = None
    supported_formats: tuple[str, ...] = (
        "video/mp4",
        "video/mpeg",
        "video/webm",
        "video/quicktime",
    )


@dataclass(frozen=True)
class ProviderConstraints:
    """Complete set of constraints for a provider.

    Attributes:
        name: Provider name identifier.
        image: Image file constraints.
        pdf: PDF file constraints.
        audio: Audio file constraints.
        video: Video file constraints.
        general_max_size_bytes: Maximum size for any file type.
        supports_file_upload: Whether the provider supports file upload APIs.
        file_upload_threshold_bytes: Size threshold above which to use file upload.
    """

    name: str
    image: ImageConstraints | None = None
    pdf: PDFConstraints | None = None
    audio: AudioConstraints | None = None
    video: VideoConstraints | None = None
    general_max_size_bytes: int | None = None
    supports_file_upload: bool = False
    file_upload_threshold_bytes: int | None = None


# Anthropic constraints (Claude 3+)
# https://docs.anthropic.com/en/docs/build-with-claude/vision
ANTHROPIC_CONSTRAINTS = ProviderConstraints(
    name="anthropic",
    image=ImageConstraints(
        max_size_bytes=5 * 1024 * 1024,  # 5MB
        max_width=8000,
        max_height=8000,
        supported_formats=("image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    pdf=PDFConstraints(
        max_size_bytes=30 * 1024 * 1024,  # 30MB
        max_pages=100,
    ),
    supports_file_upload=True,
    file_upload_threshold_bytes=5 * 1024 * 1024,  # Use upload for files > 5MB
)

# OpenAI constraints (GPT-4o, GPT-4 Vision)
# https://platform.openai.com/docs/guides/vision
OPENAI_CONSTRAINTS = ProviderConstraints(
    name="openai",
    image=ImageConstraints(
        max_size_bytes=20 * 1024 * 1024,  # 20MB
        max_images_per_request=10,
        supported_formats=("image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    # OpenAI does not support PDFs natively
    pdf=None,
    supports_file_upload=True,
    file_upload_threshold_bytes=5 * 1024 * 1024,  # Use upload for files > 5MB
)

# Gemini constraints
# https://ai.google.dev/gemini-api/docs/vision
GEMINI_CONSTRAINTS = ProviderConstraints(
    name="gemini",
    image=ImageConstraints(
        max_size_bytes=100 * 1024 * 1024,  # 100MB inline
        supported_formats=(
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "image/heic",
            "image/heif",
        ),
    ),
    pdf=PDFConstraints(
        max_size_bytes=50 * 1024 * 1024,  # 50MB inline
    ),
    audio=AudioConstraints(
        max_size_bytes=100 * 1024 * 1024,  # 100MB
        supported_formats=(
            "audio/mp3",
            "audio/mpeg",
            "audio/wav",
            "audio/ogg",
            "audio/flac",
            "audio/aac",
            "audio/m4a",
            "audio/opus",
        ),
    ),
    video=VideoConstraints(
        max_size_bytes=2 * 1024 * 1024 * 1024,  # 2GB via File API
        supported_formats=(
            "video/mp4",
            "video/mpeg",
            "video/webm",
            "video/quicktime",
            "video/x-msvideo",
            "video/x-flv",
        ),
    ),
    supports_file_upload=True,
    file_upload_threshold_bytes=20 * 1024 * 1024,  # Use upload for files > 20MB
)

# AWS Bedrock constraints (Claude via Bedrock)
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
BEDROCK_CONSTRAINTS = ProviderConstraints(
    name="bedrock",
    image=ImageConstraints(
        max_size_bytes=4_608_000,  # ~4.5MB (encoded size limit)
        max_width=8000,
        max_height=8000,
        supported_formats=("image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    pdf=PDFConstraints(
        max_size_bytes=3_840_000,  # ~3.75MB
        max_pages=100,
    ),
)

# Azure OpenAI constraints (same as OpenAI)
AZURE_CONSTRAINTS = ProviderConstraints(
    name="azure",
    image=ImageConstraints(
        max_size_bytes=20 * 1024 * 1024,  # 20MB
        max_images_per_request=10,
        supported_formats=("image/png", "image/jpeg", "image/gif", "image/webp"),
    ),
    pdf=None,
)


# Provider name mapping for convenience
_PROVIDER_CONSTRAINTS_MAP: dict[str, ProviderConstraints] = {
    "anthropic": ANTHROPIC_CONSTRAINTS,
    "openai": OPENAI_CONSTRAINTS,
    "gemini": GEMINI_CONSTRAINTS,
    "bedrock": BEDROCK_CONSTRAINTS,
    "azure": AZURE_CONSTRAINTS,
    # Aliases
    "claude": ANTHROPIC_CONSTRAINTS,
    "gpt": OPENAI_CONSTRAINTS,
    "google": GEMINI_CONSTRAINTS,
    "aws": BEDROCK_CONSTRAINTS,
}


def get_constraints_for_provider(
    provider: str | ProviderConstraints,
) -> ProviderConstraints | None:
    """Get constraints for a provider by name or return if already ProviderConstraints.

    Args:
        provider: Provider name string or ProviderConstraints instance.

    Returns:
        ProviderConstraints for the provider, or None if not found.
    """
    if isinstance(provider, ProviderConstraints):
        return provider

    provider_lower = provider.lower()

    # Direct lookup
    if provider_lower in _PROVIDER_CONSTRAINTS_MAP:
        return _PROVIDER_CONSTRAINTS_MAP[provider_lower]

    # Check if provider name contains any known provider
    for key, constraints in _PROVIDER_CONSTRAINTS_MAP.items():
        if key in provider_lower:
            return constraints

    return None
