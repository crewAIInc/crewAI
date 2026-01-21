"""File transformation functions for resizing, optimizing, and chunking."""

from collections.abc import Sequence
import io
import logging

from crewai.utilities.files.content_types import ImageFile, PDFFile, TextFile
from crewai.utilities.files.file import FileBytes
from crewai.utilities.files.processing.exceptions import ProcessingDependencyError


logger = logging.getLogger(__name__)


def resize_image(
    file: ImageFile,
    max_width: int,
    max_height: int,
    *,
    preserve_aspect_ratio: bool = True,
) -> ImageFile:
    """Resize an image to fit within the specified dimensions.

    Args:
        file: The image file to resize.
        max_width: Maximum width in pixels.
        max_height: Maximum height in pixels.
        preserve_aspect_ratio: If True, maintain aspect ratio while fitting within bounds.

    Returns:
        A new ImageFile with the resized image data.

    Raises:
        ProcessingDependencyError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ProcessingDependencyError(
            "Pillow is required for image resizing",
            dependency="Pillow",
            install_command="pip install Pillow",
        ) from e

    content = file.source.read()

    with Image.open(io.BytesIO(content)) as img:
        original_width, original_height = img.size

        # Check if resize is needed
        if original_width <= max_width and original_height <= max_height:
            return file

        if preserve_aspect_ratio:
            # Calculate scaling factor to fit within bounds
            width_ratio = max_width / original_width
            height_ratio = max_height / original_height
            scale_factor = min(width_ratio, height_ratio)

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
        else:
            new_width = min(original_width, max_width)
            new_height = min(original_height, max_height)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Determine output format
        output_format = img.format or "PNG"
        if output_format.upper() == "JPEG":
            # Handle RGBA images for JPEG
            if resized_img.mode in ("RGBA", "LA", "P"):
                resized_img = resized_img.convert("RGB")

        # Save to bytes
        output_buffer = io.BytesIO()
        resized_img.save(output_buffer, format=output_format)
        output_bytes = output_buffer.getvalue()

        logger.info(
            f"Resized image '{file.filename}' from {original_width}x{original_height} "
            f"to {new_width}x{new_height}"
        )

        return ImageFile(source=FileBytes(data=output_bytes, filename=file.filename))


def optimize_image(
    file: ImageFile,
    target_size_bytes: int,
    *,
    min_quality: int = 20,
    initial_quality: int = 85,
) -> ImageFile:
    """Optimize an image to fit within a target file size.

    Uses iterative quality reduction to achieve target size.

    Args:
        file: The image file to optimize.
        target_size_bytes: Target maximum file size in bytes.
        min_quality: Minimum quality to use (prevents excessive degradation).
        initial_quality: Starting quality for optimization.

    Returns:
        A new ImageFile with the optimized image data.

    Raises:
        ProcessingDependencyError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ProcessingDependencyError(
            "Pillow is required for image optimization",
            dependency="Pillow",
            install_command="pip install Pillow",
        ) from e

    content = file.source.read()
    current_size = len(content)

    # If already within target, return as-is
    if current_size <= target_size_bytes:
        return file

    with Image.open(io.BytesIO(content)) as img:
        # Convert to RGB for JPEG compression if needed
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
            output_format = "JPEG"
        else:
            output_format = img.format or "JPEG"
            if output_format.upper() not in ("JPEG", "JPG"):
                output_format = "JPEG"

        quality = initial_quality
        output_bytes = content

        # Binary search for optimal quality
        while len(output_bytes) > target_size_bytes and quality >= min_quality:
            output_buffer = io.BytesIO()
            img.save(
                output_buffer, format=output_format, quality=quality, optimize=True
            )
            output_bytes = output_buffer.getvalue()

            if len(output_bytes) > target_size_bytes:
                quality -= 5

        logger.info(
            f"Optimized image '{file.filename}' from {current_size} bytes to "
            f"{len(output_bytes)} bytes (quality={quality})"
        )

        filename = file.filename
        if (
            filename
            and output_format.upper() == "JPEG"
            and not filename.lower().endswith((".jpg", ".jpeg"))
        ):
            filename = filename.rsplit(".", 1)[0] + ".jpg"

        return ImageFile(source=FileBytes(data=output_bytes, filename=filename))


def chunk_pdf(
    file: PDFFile,
    max_pages: int,
    *,
    overlap_pages: int = 0,
) -> Sequence[PDFFile]:
    """Split a PDF into chunks of maximum page count.

    Args:
        file: The PDF file to chunk.
        max_pages: Maximum pages per chunk.
        overlap_pages: Number of overlapping pages between chunks (for context).

    Returns:
        List of PDFFile objects, one per chunk.

    Raises:
        ProcessingDependencyError: If pypdf is not installed.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore[import-not-found]
    except ImportError as e:
        raise ProcessingDependencyError(
            "pypdf is required for PDF chunking",
            dependency="pypdf",
            install_command="pip install pypdf",
        ) from e

    content = file.source.read()
    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)

    # If within limit, return as-is
    if total_pages <= max_pages:
        return [file]

    chunks: list[PDFFile] = []
    filename = file.filename or "document.pdf"
    base_filename = filename.rsplit(".", 1)[0]
    step = max_pages - overlap_pages

    chunk_num = 0
    start_page = 0

    while start_page < total_pages:
        end_page = min(start_page + max_pages, total_pages)

        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        output_bytes = output_buffer.getvalue()

        chunk_filename = f"{base_filename}_chunk_{chunk_num}.pdf"
        chunks.append(
            PDFFile(source=FileBytes(data=output_bytes, filename=chunk_filename))
        )

        logger.info(
            f"Created PDF chunk '{chunk_filename}' with pages {start_page + 1}-{end_page}"
        )

        start_page += step
        chunk_num += 1

    return chunks


def chunk_text(
    file: TextFile,
    max_chars: int,
    *,
    overlap_chars: int = 200,
    split_on_newlines: bool = True,
) -> Sequence[TextFile]:
    """Split a text file into chunks of maximum character count.

    Args:
        file: The text file to chunk.
        max_chars: Maximum characters per chunk.
        overlap_chars: Number of overlapping characters between chunks.
        split_on_newlines: If True, prefer splitting at newline boundaries.

    Returns:
        List of TextFile objects, one per chunk.
    """
    content = file.source.read()
    text = content.decode("utf-8", errors="replace")
    total_chars = len(text)

    # If within limit, return as-is
    if total_chars <= max_chars:
        return [file]

    chunks: list[TextFile] = []
    filename = file.filename or "text.txt"
    base_filename = filename.rsplit(".", 1)[0]
    extension = filename.rsplit(".", 1)[-1] if "." in filename else "txt"

    chunk_num = 0
    start_pos = 0

    while start_pos < total_chars:
        end_pos = min(start_pos + max_chars, total_chars)

        # If not at end, try to find a better split point
        if end_pos < total_chars and split_on_newlines:
            # Look for last newline within the chunk
            last_newline = text.rfind("\n", start_pos, end_pos)
            if last_newline > start_pos + max_chars // 2:  # Don't split too early
                end_pos = last_newline + 1

        chunk_text = text[start_pos:end_pos]
        chunk_bytes = chunk_text.encode("utf-8")

        chunk_filename = f"{base_filename}_chunk_{chunk_num}.{extension}"
        chunks.append(
            TextFile(source=FileBytes(data=chunk_bytes, filename=chunk_filename))
        )

        logger.info(
            f"Created text chunk '{chunk_filename}' with {len(chunk_text)} characters"
        )

        # Move start position with overlap
        start_pos = end_pos - overlap_chars if end_pos < total_chars else total_chars
        chunk_num += 1

    return chunks


def get_image_dimensions(file: ImageFile) -> tuple[int, int] | None:
    """Get the dimensions of an image file.

    Args:
        file: The image file to measure.

    Returns:
        Tuple of (width, height) in pixels, or None if dimensions cannot be determined.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed - cannot get image dimensions")
        return None

    content = file.source.read()

    try:
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
            return (width, height)
    except Exception as e:
        logger.warning(f"Failed to get image dimensions: {e}")
        return None


def get_pdf_page_count(file: PDFFile) -> int | None:
    """Get the page count of a PDF file.

    Args:
        file: The PDF file to measure.

    Returns:
        Number of pages, or None if page count cannot be determined.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed - cannot get PDF page count")
        return None

    content = file.source.read()

    try:
        reader = PdfReader(io.BytesIO(content))
        return len(reader.pages)
    except Exception as e:
        logger.warning(f"Failed to get PDF page count: {e}")
        return None
