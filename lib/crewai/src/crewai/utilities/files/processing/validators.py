"""File validation functions for checking against provider constraints."""

from collections.abc import Sequence
import logging

from crewai.utilities.files.content_types import (
    AudioFile,
    File,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai.utilities.files.processing.constraints import (
    AudioConstraints,
    ImageConstraints,
    PDFConstraints,
    ProviderConstraints,
    VideoConstraints,
)
from crewai.utilities.files.processing.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)


logger = logging.getLogger(__name__)

FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes}B"


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

    if file_size > constraints.max_size_bytes:
        msg = (
            f"Image '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(constraints.max_size_bytes)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=constraints.max_size_bytes,
            )

    content_type = file.content_type
    if content_type not in constraints.supported_formats:
        msg = (
            f"Image format '{content_type}' is not supported. "
            f"Supported: {', '.join(constraints.supported_formats)}"
        )
        errors.append(msg)
        if raise_on_error:
            raise UnsupportedFileTypeError(
                msg, file_name=filename, content_type=content_type
            )

    if constraints.max_width is not None or constraints.max_height is not None:
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(content)) as img:
                width, height = img.size

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

        except ImportError:
            logger.warning(
                "Pillow not installed - cannot validate image dimensions. "
                "Install with: pip install Pillow"
            )

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

    if file_size > constraints.max_size_bytes:
        msg = (
            f"PDF '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(constraints.max_size_bytes)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=constraints.max_size_bytes,
            )

    if constraints.max_pages is not None:
        try:
            import io

            from pypdf import PdfReader  # type: ignore[import-not-found]

            reader = PdfReader(io.BytesIO(content))
            page_count = len(reader.pages)

            if page_count > constraints.max_pages:
                msg = (
                    f"PDF '{filename}' page count ({page_count}) exceeds "
                    f"maximum ({constraints.max_pages})"
                )
                errors.append(msg)
                if raise_on_error:
                    raise FileValidationError(msg, file_name=filename)

        except ImportError:
            logger.warning(
                "pypdf not installed - cannot validate PDF page count. "
                "Install with: pip install pypdf"
            )

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
        UnsupportedFileTypeError: If the format is not supported.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    if file_size > constraints.max_size_bytes:
        msg = (
            f"Audio '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(constraints.max_size_bytes)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=constraints.max_size_bytes,
            )

    content_type = file.content_type
    if content_type not in constraints.supported_formats:
        msg = (
            f"Audio format '{content_type}' is not supported. "
            f"Supported: {', '.join(constraints.supported_formats)}"
        )
        errors.append(msg)
        if raise_on_error:
            raise UnsupportedFileTypeError(
                msg, file_name=filename, content_type=content_type
            )

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
        UnsupportedFileTypeError: If the format is not supported.
    """
    errors: list[str] = []
    content = file.read()
    file_size = len(content)
    filename = file.filename

    if file_size > constraints.max_size_bytes:
        msg = (
            f"Video '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(constraints.max_size_bytes)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=constraints.max_size_bytes,
            )

    content_type = file.content_type
    if content_type not in constraints.supported_formats:
        msg = (
            f"Video format '{content_type}' is not supported. "
            f"Supported: {', '.join(constraints.supported_formats)}"
        )
        errors.append(msg)
        if raise_on_error:
            raise UnsupportedFileTypeError(
                msg, file_name=filename, content_type=content_type
            )

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

    content = file.read()
    file_size = len(content)
    filename = file.filename

    if file_size > constraints.general_max_size_bytes:
        msg = (
            f"Text file '{filename}' size ({_format_size(file_size)}) exceeds "
            f"maximum ({_format_size(constraints.general_max_size_bytes)})"
        )
        errors.append(msg)
        if raise_on_error:
            raise FileTooLargeError(
                msg,
                file_name=filename,
                actual_size=file_size,
                max_size=constraints.general_max_size_bytes,
            )

    return errors


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
            msg = f"Provider '{constraints.name}' does not support images"
            if raise_on_error:
                raise UnsupportedFileTypeError(
                    msg, file_name=file.filename, content_type=file.content_type
                )
            return [msg]
        return validate_image(file, constraints.image, raise_on_error=raise_on_error)

    if isinstance(file, PDFFile):
        if constraints.pdf is None:
            msg = f"Provider '{constraints.name}' does not support PDFs"
            if raise_on_error:
                raise UnsupportedFileTypeError(
                    msg, file_name=file.filename, content_type=file.content_type
                )
            return [msg]
        return validate_pdf(file, constraints.pdf, raise_on_error=raise_on_error)

    if isinstance(file, AudioFile):
        if constraints.audio is None:
            msg = f"Provider '{constraints.name}' does not support audio"
            if raise_on_error:
                raise UnsupportedFileTypeError(
                    msg, file_name=file.filename, content_type=file.content_type
                )
            return [msg]
        return validate_audio(file, constraints.audio, raise_on_error=raise_on_error)

    if isinstance(file, VideoFile):
        if constraints.video is None:
            msg = f"Provider '{constraints.name}' does not support video"
            if raise_on_error:
                raise UnsupportedFileTypeError(
                    msg, file_name=file.filename, content_type=file.content_type
                )
            return [msg]
        return validate_video(file, constraints.video, raise_on_error=raise_on_error)

    if isinstance(file, TextFile):
        return validate_text(file, constraints, raise_on_error=raise_on_error)

    return []
