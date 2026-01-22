"""Exceptions for file processing operations."""


class FileProcessingError(Exception):
    """Base exception for file processing errors."""

    def __init__(self, message: str, file_name: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the issue.
            file_name: Optional name of the file that caused the error.
        """
        self.file_name = file_name
        super().__init__(message)


class FileValidationError(FileProcessingError):
    """Raised when file validation fails."""


class FileTooLargeError(FileValidationError):
    """Raised when a file exceeds the maximum allowed size."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        actual_size: int | None = None,
        max_size: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the issue.
            file_name: Optional name of the file that caused the error.
            actual_size: The actual size of the file in bytes.
            max_size: The maximum allowed size in bytes.
        """
        self.actual_size = actual_size
        self.max_size = max_size
        super().__init__(message, file_name)


class UnsupportedFileTypeError(FileValidationError):
    """Raised when a file type is not supported by the provider."""

    def __init__(
        self,
        message: str,
        file_name: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the issue.
            file_name: Optional name of the file that caused the error.
            content_type: The content type that is not supported.
        """
        self.content_type = content_type
        super().__init__(message, file_name)


class ProcessingDependencyError(FileProcessingError):
    """Raised when a required processing dependency is not installed."""

    def __init__(
        self,
        message: str,
        dependency: str,
        install_command: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the issue.
            dependency: Name of the missing dependency.
            install_command: Optional command to install the dependency.
        """
        self.dependency = dependency
        self.install_command = install_command
        super().__init__(message)


class TransientFileError(FileProcessingError):
    """Transient error that may succeed on retry (network, timeout)."""


class PermanentFileError(FileProcessingError):
    """Permanent error that will not succeed on retry (auth, format)."""


class UploadError(FileProcessingError):
    """Base exception for upload errors."""


class TransientUploadError(UploadError, TransientFileError):
    """Upload failed but may succeed on retry (network issues, rate limits)."""


class PermanentUploadError(UploadError, PermanentFileError):
    """Upload failed permanently (auth failure, invalid file, unsupported type)."""


def classify_upload_error(e: Exception, filename: str | None = None) -> Exception:
    """Classify an exception as transient or permanent upload error.

    Analyzes the exception type name and status code to determine if
    the error is likely transient (retryable) or permanent.

    Args:
        e: The exception to classify.
        filename: Optional filename for error context.

    Returns:
        A TransientUploadError or PermanentUploadError wrapping the original.
    """
    error_type = type(e).__name__

    if "RateLimit" in error_type or "APIConnection" in error_type:
        return TransientUploadError(f"Transient upload error: {e}", file_name=filename)
    if "Authentication" in error_type or "Permission" in error_type:
        return PermanentUploadError(
            f"Authentication/permission error: {e}", file_name=filename
        )
    if "BadRequest" in error_type or "InvalidRequest" in error_type:
        return PermanentUploadError(f"Invalid request: {e}", file_name=filename)

    status_code = getattr(e, "status_code", None)
    if status_code is not None:
        if status_code >= 500 or status_code == 429:
            return TransientUploadError(
                f"Server error ({status_code}): {e}", file_name=filename
            )
        if status_code in (401, 403):
            return PermanentUploadError(
                f"Auth error ({status_code}): {e}", file_name=filename
            )
        if status_code == 400:
            return PermanentUploadError(
                f"Bad request ({status_code}): {e}", file_name=filename
            )

    return TransientUploadError(f"Upload failed: {e}", file_name=filename)
