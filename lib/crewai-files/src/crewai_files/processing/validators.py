"""File validation functions for checking against provider constraints."""

from collections.abc import Sequence
import io
import logging

from crewai_files.core.types import (
    AudioFile,
    FileInput,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai_files.processing.constraints import (
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


logger = logging.getLogger(__name__)


def _get_image_dimensions(content: bytes) -> tuple[int, int] | None:
    """Get image dimensions using Pillow if available.

    Args:
        content: Raw image bytes.

    Returns:
        Tuple of (width, height) or None if Pillow unavailable.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
            return int(width), int(height)
    except ImportError:
        logger.warning(
            "Pillow not installed - cannot validate image dimensions. "
            "Install with: pip install Pillow"
        )
        return None


def _get_pdf_page_count(content: bytes) -> int | None:
    """Get PDF page count using pypdf if available.

    Args:
        content: Raw PDF bytes.

    Returns:
        Page count or None if pypdf unavailable.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return len(reader.pages)
    except ImportError:
        logger.warning(
            "pypdf not installed - cannot validate PDF page count. "
            "Install with: pip install pypdf"
        )
        return None


def _get_audio_duration(content: bytes, filename: str | None = None) -> float | None:
    """Get audio duration in seconds using tinytag if available.

    Args:
        content: Raw audio bytes.
        filename: Optional filename for format detection hint.

    Returns:
        Duration in seconds or None if tinytag unavailable.
    """
    try:
        from tinytag import TinyTag  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "tinytag not installed - cannot validate audio duration. "
            "Install with: pip install tinytag"
        )
        return None

    try:
        tag = TinyTag.get(file_obj=io.BytesIO(content), filename=filename)
        duration: float | None = tag.duration
        return duration
    except Exception as e:
        logger.debug(f"Could not determine audio duration: {e}")
        return None


_VIDEO_FORMAT_MAP: dict[str, str] = {
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-matroska": "matroska",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
    "video/x-flv": "flv",
}


def _get_video_duration(
    content: bytes, content_type: str | None = None
) -> float | None:
    """Get video duration in seconds using av if available.

    Args:
        content: Raw video bytes.
        content_type: Optional MIME type for format detection hint.

    Returns:
        Duration in seconds or None if av unavailable.
    """
    try:
        import av
    except ImportError:
        logger.warning(
            "av (PyAV) not installed - cannot validate video duration. "
            "Install with: pip install av"
        )
        return None

    format_hint = _VIDEO_FORMAT_MAP.get(content_type) if content_type else None

    try:
        with av.open(io.BytesIO(content), format=format_hint) as container:  # type: ignore[attr-defined]
            duration: int | None = container.duration  # type: ignore[union-attr]
            if duration is None:
                return None
            return float(duration) / 1_000_000
    except Exception as e:
        logger.debug(f"Could not determine video duration: {e}")

    return None


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes}B"


def _validate_size(
    file_type: str,
    filename: str | None,
    file_size: int,
    max_size: int,
    errors: list[str],
    raise_on_error: bool,
) -> None:
    """Validate file size against maximum.

    Args:
        file_type: Type label for error messages (e.g., "Image", "PDF").
        filename: Name of the file being validated.
        file_size: Actual file size in bytes.
        max_size: Maximum allowed size in bytes.
        errors: List to append error messages to.
        raise_on_error: If True, raise FileTooLargeError on failure.
    """
    if file_size > max_size:
        msg = (
            f"{file_type} '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(max_size)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=max_size,
            )


def _validate_format(
    file_type: str,
    filename: str | None,
    content_type: str,
    supported_formats: tuple[str, ...],
    errors: list[str],
    raise_on_error: bool,
) -> None:
    """Validate content type against supported formats.

    Args:
        file_type: Type label for error messages (e.g., "Image", "Audio").
        filename: Name of the file being validated.
        content_type: MIME type of the file.
        supported_formats: Tuple of supported MIME types.
        errors: List to append error messages to.
        raise_on_error: If True, raise UnsupportedFileTypeError on failure.
    """
    if content_type not in supported_formats:
        msg = (
            f"{file_type} format '{content_type}' is not supported. "
            f"Supported: {', '.join(supported_formats)}"
        )
        errors.append(msg)
        if raise_on_error:
            raise UnsupportedFileTypeError(
                msg, file_name=filename, content_type=content_type
            )


def validate_image(
    file: ImageFile,
    constraints: ImageConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate an image file against constraints.

    Args:
        file: The image file to validate.
        constraints: Image constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
        FileValidationError: If the file exceeds dimension limits.
        UnsupportedFileTypeError: If the format is not supported.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    _validate_size(
        "Image", filename, file_size, constraints.max_size_bytes, errors, raise_on_error
    )
    _validate_format(
        "Image",
        filename,
        file.content_type,
        constraints.supported_formats,
        errors,
        raise_on_error,
    )

    if constraints.max_width is not None or constraints.max_height is not None:
        dimensions = _get_image_dimensions(content)
        if dimensions is not None:
            width, height = dimensions

            if constraints.max_width and width > constraints.max_width:
                msg = (
                    f"Image '{filename}' width ({width}px) exceeds "
                    f"maximum ({constraints.max_width}px)"
                )
                errors.append(msg)
                if raise_on_error:
                    raise FileValidationError(msg, file_name=filename)

            if constraints.max_height and height > constraints.max_height:
                msg = (
                    f"Image '{filename}' height ({height}px) exceeds "
                    f"maximum ({constraints.max_height}px)"
                )
                errors.append(msg)
                if raise_on_error:
                    raise FileValidationError(msg, file_name=filename)

    return errors


def validate_pdf(
    file: PDFFile,
    constraints: PDFConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate a PDF file against constraints.

    Args:
        file: The PDF file to validate.
        constraints: PDF constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
        FileValidationError: If the file exceeds page limits.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    _validate_size(
        "PDF", filename, file_size, constraints.max_size_bytes, errors, raise_on_error
    )

    if constraints.max_pages is not None:
        page_count = _get_pdf_page_count(content)
        if page_count is not None and page_count > constraints.max_pages:
            msg = (
                f"PDF '{filename}' page count ({page_count}) exceeds "
                f"maximum ({constraints.max_pages})"
            )
            errors.append(msg)
            if raise_on_error:
                raise FileValidationError(msg, file_name=filename)

    return errors


def validate_audio(
    file: AudioFile,
    constraints: AudioConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate an audio file against constraints.

    Args:
        file: The audio file to validate.
        constraints: Audio constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
        FileValidationError: If the file exceeds duration limits.
        UnsupportedFileTypeError: If the format is not supported.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    _validate_size(
        "Audio",
        filename,
        file_size,
        constraints.max_size_bytes,
        errors,
        raise_on_error,
    )
    _validate_format(
        "Audio",
        filename,
        file.content_type,
        constraints.supported_formats,
        errors,
        raise_on_error,
    )

    if constraints.max_duration_seconds is not None:
        duration = _get_audio_duration(content, filename)
        if duration is not None and duration > constraints.max_duration_seconds:
            msg = (
                f"Audio '{filename}' duration ({duration:.1f}s) exceeds "
                f"maximum ({constraints.max_duration_seconds}s)"
            )
            errors.append(msg)
            if raise_on_error:
                raise FileValidationError(msg, file_name=filename)

    return errors


def validate_video(
    file: VideoFile,
    constraints: VideoConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate a video file against constraints.

    Args:
        file: The video file to validate.
        constraints: Video constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
        FileValidationError: If the file exceeds duration limits.
        UnsupportedFileTypeError: If the format is not supported.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    _validate_size(
        "Video",
        filename,
        file_size,
        constraints.max_size_bytes,
        errors,
        raise_on_error,
    )
    _validate_format(
        "Video",
        filename,
        file.content_type,
        constraints.supported_formats,
        errors,
        raise_on_error,
    )

    if constraints.max_duration_seconds is not None:
        duration = _get_video_duration(content)
        if duration is not None and duration > constraints.max_duration_seconds:
            msg = (
                f"Video '{filename}' duration ({duration:.1f}s) exceeds "
                f"maximum ({constraints.max_duration_seconds}s)"
            )
            errors.append(msg)
            if raise_on_error:
                raise FileValidationError(msg, file_name=filename)

    return errors


def validate_text(
    file: TextFile,
    constraints: ProviderConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate a text file against general constraints.

    Args:
        file: The text file to validate.
        constraints: Provider constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
    """
    errors: list[str] = []

    if constraints.general_max_size_bytes is None:
        return errors

    file_size = len(file.read())
    _validate_size(
        "Text file",
        file.filename,
        file_size,
        constraints.general_max_size_bytes,
        errors,
        raise_on_error,
    )

    return errors


def _check_unsupported_type(
    file: FileInput,
    provider_name: str,
    type_name: str,
    raise_on_error: bool,
) -> Sequence[str]:
    """Check if file type is unsupported and handle error.

    Args:
        file: The file being validated.
        provider_name: Name of the provider.
        type_name: Name of the file type (e.g., "images", "PDFs").
        raise_on_error: If True, raise exception instead of returning errors.

    Returns:
        List with error message (only returns when raise_on_error is False).

    Raises:
        UnsupportedFileTypeError: If raise_on_error is True.
    """
    msg = f"Provider '{provider_name}' does not support {type_name}"
    if raise_on_error:
        raise UnsupportedFileTypeError(
            msg, file_name=file.filename, content_type=file.content_type
        )
    return [msg]


def validate_file(
    file: FileInput,
    constraints: ProviderConstraints,
    *,
    raise_on_error: bool = True,
) -> Sequence[str]:
    """Validate a file against provider constraints.

    Dispatches to the appropriate validator based on file type.

    Args:
        file: The file to validate.
        constraints: Provider constraints to validate against.
        raise_on_error: If True, raise exceptions on validation failure.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        FileTooLargeError: If the file exceeds size limits.
        FileValidationError: If the file fails other validation checks.
        UnsupportedFileTypeError: If the file type is not supported.
    """
    if isinstance(file, ImageFile):
        if constraints.image is None:
            return _check_unsupported_type(
                file, constraints.name, "images", raise_on_error
            )
        return validate_image(file, constraints.image, raise_on_error=raise_on_error)

    if isinstance(file, PDFFile):
        if constraints.pdf is None:
            return _check_unsupported_type(
                file, constraints.name, "PDFs", raise_on_error
            )
        return validate_pdf(file, constraints.pdf, raise_on_error=raise_on_error)

    if isinstance(file, AudioFile):
        if constraints.audio is None:
            return _check_unsupported_type(
                file, constraints.name, "audio", raise_on_error
            )
        return validate_audio(file, constraints.audio, raise_on_error=raise_on_error)

    if isinstance(file, VideoFile):
        if constraints.video is None:
            return _check_unsupported_type(
                file, constraints.name, "video", raise_on_error
            )
        return validate_video(file, constraints.video, raise_on_error=raise_on_error)

    if isinstance(file, TextFile):
        return validate_text(file, constraints, raise_on_error=raise_on_error)

    return []
