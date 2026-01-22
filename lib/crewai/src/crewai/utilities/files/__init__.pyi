"""Type stubs for backwards compatibility re-exports from crewai.files.

.. deprecated::
    Import from crewai.files instead.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from typing_extensions import deprecated

import crewai.files as _files

FileMode = Literal["strict", "auto", "warn", "chunk"]
ImageExtension = _files.ImageExtension
ImageContentType = _files.ImageContentType
PDFExtension = _files.PDFExtension
PDFContentType = _files.PDFContentType
TextExtension = _files.TextExtension
TextContentType = _files.TextContentType
AudioExtension = _files.AudioExtension
AudioContentType = _files.AudioContentType
VideoExtension = _files.VideoExtension
VideoContentType = _files.VideoContentType
FileInput = _files.FileInput
FileSource = _files.FileSource
FileSourceInput = _files.FileSourceInput
RawFileInput = _files.RawFileInput
ResolvedFileType = _files.ResolvedFileType
FileHandling = _files.FileHandling

# Deprecated classes
@deprecated("Import from crewai.files instead")
class BaseFile(_files.BaseFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class ImageFile(_files.ImageFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class PDFFile(_files.PDFFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class TextFile(_files.TextFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class AudioFile(_files.AudioFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class VideoFile(_files.VideoFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class File(_files.File):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FilePath(_files.FilePath):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileBytes(_files.FileBytes):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileStream(_files.FileStream):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileResolver(_files.FileResolver):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileResolverConfig(_files.FileResolverConfig):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileProcessor(_files.FileProcessor):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileUploader(_files.FileUploader):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class UploadCache(_files.UploadCache):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class CachedUpload(_files.CachedUpload):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class UploadResult(_files.UploadResult):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class ResolvedFile(_files.ResolvedFile):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileReference(_files.FileReference):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class UrlReference(_files.UrlReference):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class InlineBase64(_files.InlineBase64):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class InlineBytes(_files.InlineBytes):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class ProviderConstraints(_files.ProviderConstraints):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class ImageConstraints(_files.ImageConstraints):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class AudioConstraints(_files.AudioConstraints):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class VideoConstraints(_files.VideoConstraints):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class PDFConstraints(_files.PDFConstraints):
    """.. deprecated:: Import from crewai.files instead."""
    ...

# Exceptions
@deprecated("Import from crewai.files instead")
class FileProcessingError(_files.FileProcessingError):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileValidationError(_files.FileValidationError):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class FileTooLargeError(_files.FileTooLargeError):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class UnsupportedFileTypeError(_files.UnsupportedFileTypeError):
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
class ProcessingDependencyError(_files.ProcessingDependencyError):
    """.. deprecated:: Import from crewai.files instead."""
    ...

# Constants
OPENAI_CONSTRAINTS: _files.ProviderConstraints
ANTHROPIC_CONSTRAINTS: _files.ProviderConstraints
GEMINI_CONSTRAINTS: _files.ProviderConstraints
BEDROCK_CONSTRAINTS: _files.ProviderConstraints

# Deprecated functions
@deprecated("Import from crewai.files instead")
def create_resolver(
    provider: str,
    config: FileResolverConfig | None = None,
) -> FileResolver:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def get_uploader(provider: str, **kwargs: Any) -> FileUploader | None:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def get_upload_cache() -> UploadCache:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def reset_upload_cache() -> None:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def get_constraints_for_provider(provider: str) -> ProviderConstraints:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def cleanup_uploaded_files(provider: str | None = None) -> int:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def cleanup_expired_files() -> int:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def cleanup_provider_files(provider: str) -> int:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def normalize_input_files(
    input_files: list[FileSourceInput | FileInput],
) -> dict[str, FileInput]:
    """.. deprecated:: Import from crewai.files instead."""
    ...

@deprecated("Import from crewai.files instead")
def wrap_file_source(source: FileSource) -> FileInput:
    """.. deprecated:: Import from crewai.files instead."""
    ...

__all__: list[str]