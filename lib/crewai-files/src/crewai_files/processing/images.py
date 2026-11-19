"""Opening images with Pillow, limited to the formats crewai-files handles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import io
from typing import TYPE_CHECKING, Final

from crewai_files.processing.exceptions import UnsupportedImageFormatError


if TYPE_CHECKING:
    from PIL.Image import Image


# Pillow picks a parser from the file's leading bytes, not its declared content
# type, so without this a PSD or JPEG 2000 sent as image/png reaches that
# format's parser. Every Pillow open in crewai-files goes through open_image.
SUPPORTED_IMAGE_FORMATS: Final = ("PNG", "JPEG", "GIF", "WEBP")


@contextmanager
def open_image(
    content: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> Iterator[Image]:
    """Open image bytes with Pillow, refusing any format outside the allowlist.

    Args:
        content: Raw image bytes.
        filename: Name of the file, for the error message.
        content_type: Declared content type, carried on the error.

    Yields:
        The opened Pillow image, closed when the block exits.

    Raises:
        ImportError: If Pillow is not installed.
        UnsupportedImageFormatError: If the bytes are not a PNG, JPEG, GIF or WEBP
            image. No other format's parser is run.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(content), formats=SUPPORTED_IMAGE_FORMATS)
    except UnidentifiedImageError as e:
        name = f" '{filename}'" if filename else ""
        raise UnsupportedImageFormatError(
            f"Cannot process image{name}: it is not a valid PNG, JPEG, GIF "
            "or WEBP image",
            file_name=filename,
            content_type=content_type,
        ) from e

    with img:
        yield img
