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
