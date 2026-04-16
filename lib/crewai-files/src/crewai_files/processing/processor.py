"""FileProcessor for validating and transforming files based on provider constraints."""

import asyncio
from collections.abc import Sequence
import logging

from crewai_files.core.types import (
    AudioFile,
    File,
    FileInput,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
)
from crewai_files.processing.constraints import (
    ProviderConstraints,
    get_constraints_for_provider,
)
from crewai_files.processing.enums import FileHandling
from crewai_files.processing.exceptions import (
    FileProcessingError,
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)
from crewai_files.processing.transformers import (
    chunk_pdf,
    chunk_text,
    get_image_dimensions,
    get_pdf_page_count,
    optimize_image,
    resize_image,
)
from crewai_files.processing.validators import validate_file


logger = logging.getLogger(__name__)


class FileProcessor:
    """Processes files according to provider constraints and per-file mode mode.

    Validates files against provider-specific limits and optionally transforms
    them (resize, compress, chunk) to meet those limits. Each file specifies
    its own mode mode via `file.mode`.

    Attributes:
        constraints: Provider constraints for validation.
    """

    def __init__(
        self,
        constraints: ProviderConstraints | str | None = None,
    ) -> None:
        """Initialize the FileProcessor.

        Args:
            constraints: Provider constraints or provider name string.
                If None, validation is skipped.
        """
        if isinstance(constraints, str):
            resolved = get_constraints_for_provider(constraints)
            if resolved is None:
                logger.warning(
                    f"Unknown provider '{constraints}' - validation disabled"
                )
            self.constraints = resolved
        else:
            self.constraints = constraints

    def validate(self, file: FileInput) -> Sequence[str]:
        """Validate a file against provider constraints.

        Args:
            file: The file to validate.

        Returns:
            List of validation error messages (empty if valid).

        Raises:
            FileValidationError: If file.mode is STRICT and validation fails.
        """
        if self.constraints is None:
            return []

        mode = self._get_mode(file)
        raise_on_error = mode == FileHandling.STRICT
        return validate_file(file, self.constraints, raise_on_error=raise_on_error)

    @staticmethod
    def _get_mode(file: FileInput) -> FileHandling:
        """Get the mode mode for a file.

        Args:
            file: The file to get mode for.

        Returns:
            The file's mode mode, defaulting to AUTO.
        """
        mode = getattr(file, "mode", None)
        if mode is None:
            return FileHandling.AUTO
        if isinstance(mode, str):
            return FileHandling(mode)
        if isinstance(mode, FileHandling):
            return mode
        return FileHandling.AUTO

    def process(self, file: FileInput) -> FileInput | Sequence[FileInput]:
        """Process a single file according to constraints and its mode mode.

        Args:
            file: The file to process.

        Returns:
            The processed file (possibly transformed) or a sequence of files
            if the file was chunked.

        Raises:
            FileProcessingError: If file.mode is STRICT and processing fails.
        """
        if self.constraints is None:
            return file

        mode = self._get_mode(file)

        try:
            errors = self.validate(file)

            if not errors:
                return file

            if mode == FileHandling.STRICT:
                raise FileValidationError("; ".join(errors), file_name=file.filename)

            if mode == FileHandling.WARN:
                for error in errors:
                    logger.warning(error)
                return file

            if mode == FileHandling.AUTO:
                return self._auto_process(file)

            if mode == FileHandling.CHUNK:
                return self._chunk_process(file)

            return file

        except (FileValidationError, FileTooLargeError, UnsupportedFileTypeError):
            raise
        except Exception as e:
            logger.error(f"Error processing file '{file.filename}': {e}")
            if mode == FileHandling.STRICT:
                raise FileProcessingError(str(e), file_name=file.filename) from e
            return file

    def process_files(
        self,
        files: dict[str, FileInput],
    ) -> dict[str, FileInput]:
        """Process multiple files according to constraints.

        Args:
            files: Dictionary mapping names to file inputs.

        Returns:
            Dictionary mapping names to processed files. If a file is chunked,
            multiple entries are created with indexed names.
        """
        result: dict[str, FileInput] = {}

        for name, file in files.items():
            processed = self.process(file)

            if isinstance(processed, Sequence) and not isinstance(
                processed, (str, bytes)
            ):
                for i, chunk in enumerate(processed):
                    chunk_name = f"{name}_chunk_{i}"
                    result[chunk_name] = chunk
            else:
                result[name] = processed

        return result

    async def aprocess_files(
        self,
        files: dict[str, FileInput],
        max_concurrency: int = 10,
    ) -> dict[str, FileInput]:
        """Async process multiple files in parallel.

        Args:
            files: Dictionary mapping names to file inputs.
            max_concurrency: Maximum number of concurrent processing tasks.

        Returns:
            Dictionary mapping names to processed files. If a file is chunked,
            multiple entries are created with indexed names.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single(
            key: str, input_file: FileInput
        ) -> tuple[str, FileInput | Sequence[FileInput]]:
            """Process a single file with semaphore limiting."""
            async with semaphore:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self.process, input_file)
                return key, result

        tasks = [process_single(n, f) for n, f in files.items()]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, FileInput] = {}
        for item in gather_results:
            if isinstance(item, BaseException):
                logger.error(f"Processing failed: {item}")
                continue
            entry_name, processed = item
            if isinstance(processed, Sequence) and not isinstance(
                processed, (str, bytes)
            ):
                for i, chunk in enumerate(processed):
                    output[f"{entry_name}_chunk_{i}"] = chunk
            elif isinstance(
                processed, (AudioFile, File, ImageFile, PDFFile, TextFile, VideoFile)
            ):
                output[entry_name] = processed

        return output

    def _auto_process(self, file: FileInput) -> FileInput:
        """Automatically resize/compress file to meet constraints.

        Args:
            file: The file to process.

        Returns:
            The processed file.
        """
        if self.constraints is None:
            return file

        if isinstance(file, ImageFile) and self.constraints.image is not None:
            return self._auto_process_image(file)

        if isinstance(file, PDFFile) and self.constraints.pdf is not None:
            logger.warning(
                f"Cannot auto-compress PDF '{file.filename}'. "
                "Consider using CHUNK mode for large PDFs."
            )
            return file

        if isinstance(file, (AudioFile, VideoFile)):
            logger.warning(
                f"Auto-processing not supported for {type(file).__name__}. "
                "File will be used as-is."
            )
            return file

        return file

    def _auto_process_image(self, file: ImageFile) -> ImageFile:
        """Auto-process an image file.

        Args:
            file: The image file to process.

        Returns:
            The processed image file.
        """
        if self.constraints is None or self.constraints.image is None:
            return file

        image_constraints = self.constraints.image
        processed = file
        content = file.read()
        current_size = len(content)

        if image_constraints.max_width or image_constraints.max_height:
            dimensions = get_image_dimensions(file)
            if dimensions:
                width, height = dimensions
                max_w = image_constraints.max_width or width
                max_h = image_constraints.max_height or height

                if width > max_w or height > max_h:
                    try:
                        processed = resize_image(file, max_w, max_h)
                        content = processed.read()
                        current_size = len(content)
                    except Exception as e:
                        logger.warning(f"Failed to resize image: {e}")

        if current_size > image_constraints.max_size_bytes:
            try:
                processed = optimize_image(processed, image_constraints.max_size_bytes)
            except Exception as e:
                logger.warning(f"Failed to optimize image: {e}")

        return processed

    def _chunk_process(self, file: FileInput) -> FileInput | Sequence[FileInput]:
        """Split file into chunks to meet constraints.

        Args:
            file: The file to chunk.

        Returns:
            Original file if chunking not needed, or sequence of chunked files.
        """
        if self.constraints is None:
            return file

        if isinstance(file, PDFFile) and self.constraints.pdf is not None:
            max_pages = self.constraints.pdf.max_pages
            if max_pages is not None:
                page_count = get_pdf_page_count(file)
                if page_count is not None and page_count > max_pages:
                    try:
                        return list(chunk_pdf(file, max_pages))
                    except Exception as e:
                        logger.warning(f"Failed to chunk PDF: {e}")
                        return file

        if isinstance(file, TextFile):
            # Use general max size as character limit approximation
            max_size = self.constraints.general_max_size_bytes
            if max_size is not None:
                content = file.read()
                if len(content) > max_size:
                    try:
                        return list(chunk_text(file, max_size))
                    except Exception as e:
                        logger.warning(f"Failed to chunk text file: {e}")
                        return file

        if isinstance(file, (ImageFile, AudioFile, VideoFile)):
            logger.warning(
                f"Chunking not supported for {type(file).__name__}. "
                "Consider using AUTO mode for images."
            )

        return file
