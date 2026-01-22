"""File handling utilities for crewAI tasks."""

from crewai.files.cleanup import (
    cleanup_expired_files,
    cleanup_provider_files,
    cleanup_uploaded_files,
)
from crewai.files.content_types import (
    AudioContentType,
    AudioExtension,
    AudioFile,
    BaseFile,
    File,
    FileMode,
    ImageContentType,
    ImageExtension,
    ImageFile,
    PDFContentType,
    PDFExtension,
    PDFFile,
    TextContentType,
    TextExtension,
    TextFile,
    VideoContentType,
    VideoExtension,
    VideoFile,
)
from crewai.files.file import (
    FileBytes,
    FilePath,
    FileSource,
    FileSourceInput,
    FileStream,
    RawFileInput,
)
from crewai.files.processing import (
    ANTHROPIC_CONSTRAINTS,
    BEDROCK_CONSTRAINTS,
    GEMINI_CONSTRAINTS,
    OPENAI_CONSTRAINTS,
    AudioConstraints,
    FileHandling,
    FileProcessingError,
    FileProcessor,
    FileTooLargeError,
    FileValidationError,
    ImageConstraints,
    PDFConstraints,
    ProcessingDependencyError,
    ProviderConstraints,
    UnsupportedFileTypeError,
    VideoConstraints,
    get_constraints_for_provider,
)
from crewai.files.resolved import (
    FileReference,
    InlineBase64,
    InlineBytes,
    ResolvedFile,
    ResolvedFileType,
    UrlReference,
)
from crewai.files.resolver import (
    FileResolver,
    FileResolverConfig,
    create_resolver,
)
from crewai.files.upload_cache import (
    CachedUpload,
    UploadCache,
    get_upload_cache,
    reset_upload_cache,
)
from crewai.files.uploaders import FileUploader, UploadResult, get_uploader


FileInput = AudioFile | File | ImageFile | PDFFile | TextFile | VideoFile


def wrap_file_source(source: FileSource) -> FileInput:
    """Wrap a FileSource in the appropriate typed FileInput wrapper.

    Args:
        source: The file source to wrap.

    Returns:
        Typed FileInput wrapper based on content type.
    """
    content_type = source.content_type

    if content_type.startswith("image/"):
        return ImageFile(source=source)
    if content_type.startswith("audio/"):
        return AudioFile(source=source)
    if content_type.startswith("video/"):
        return VideoFile(source=source)
    if content_type == "application/pdf":
        return PDFFile(source=source)
    return TextFile(source=source)


def normalize_input_files(
    input_files: list[FileSourceInput | FileInput],
) -> dict[str, FileInput]:
    """Convert a list of file sources to a named dictionary of FileInputs.

    Args:
        input_files: List of file source inputs or File objects.

    Returns:
        Dictionary mapping names to FileInput wrappers.
    """
    from pathlib import Path

    result: dict[str, FileInput] = {}

    for i, item in enumerate(input_files):
        if isinstance(item, BaseFile):
            name = item.filename or f"file_{i}"
            if "." in name:
                name = name.rsplit(".", 1)[0]
            result[name] = item
            continue

        file_source: FilePath | FileBytes | FileStream
        if isinstance(item, (FilePath, FileBytes, FileStream)):
            file_source = item
        elif isinstance(item, Path):
            file_source = FilePath(path=item)
        elif isinstance(item, str):
            file_source = FilePath(path=Path(item))
        elif isinstance(item, (bytes, memoryview)):
            file_source = FileBytes(data=bytes(item))
        else:
            continue

        name = file_source.filename or f"file_{i}"
        result[name] = wrap_file_source(file_source)

    return result


__all__ = [
    "ANTHROPIC_CONSTRAINTS",
    "BEDROCK_CONSTRAINTS",
    "GEMINI_CONSTRAINTS",
    "OPENAI_CONSTRAINTS",
    "AudioConstraints",
    "AudioContentType",
    "AudioExtension",
    "AudioFile",
    "BaseFile",
    "CachedUpload",
    "File",
    "FileBytes",
    "FileHandling",
    "FileInput",
    "FileMode",
    "FilePath",
    "FileProcessingError",
    "FileProcessor",
    "FileReference",
    "FileResolver",
    "FileResolverConfig",
    "FileSource",
    "FileSourceInput",
    "FileStream",
    "FileTooLargeError",
    "FileUploader",
    "FileValidationError",
    "ImageConstraints",
    "ImageContentType",
    "ImageExtension",
    "ImageFile",
    "InlineBase64",
    "InlineBytes",
    "PDFConstraints",
    "PDFContentType",
    "PDFExtension",
    "PDFFile",
    "ProcessingDependencyError",
    "ProviderConstraints",
    "RawFileInput",
    "ResolvedFile",
    "ResolvedFileType",
    "TextContentType",
    "TextExtension",
    "TextFile",
    "UnsupportedFileTypeError",
    "UploadCache",
    "UploadResult",
    "UrlReference",
    "VideoConstraints",
    "VideoContentType",
    "VideoExtension",
    "VideoFile",
    "cleanup_expired_files",
    "cleanup_provider_files",
    "cleanup_uploaded_files",
    "create_resolver",
    "get_constraints_for_provider",
    "get_upload_cache",
    "get_uploader",
    "normalize_input_files",
    "reset_upload_cache",
    "wrap_file_source",
]
