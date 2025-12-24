"""Error message definitions for CrewAI database operations.

This module provides standardized error classes and message templates
for database operations and agent repository handling.
"""

from typing import Final


class DatabaseOperationError(Exception):
    """Base exception class for database operation errors.

    Attributes:
        original_error: The original exception that caused this error, if any.
    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        """Initialize the database operation error.

        Args:
            message: The error message to display
            original_error: The original exception that caused this error, if any
        """
        super().__init__(message)
        self.original_error = original_error


class DatabaseError:
    """Standardized error message templates for database operations.

    Provides consistent error message formatting for various database
    operation failures.
    """

    INIT_ERROR: Final[str] = "Database initialisatie fout: {}"
    SAVE_ERROR: Final[str] = "Fout bij opslaan van taak outputs: {}"
    UPDATE_ERROR: Final[str] = "Fout bij bijwerken van taak outputs: {}"
    LOAD_ERROR: Final[str] = "Fout bij laden van taak outputs: {}"
    DELETE_ERROR: Final[str] = "Fout bij verwijderen van taak outputs: {}"

    @classmethod
    def format_error(cls, template: str, error: Exception) -> str:
        """Format an error message with the given template and error.

        Args:
            template: The error message template to use
            error: The exception to format into the template

        Returns:
            The formatted error message
        """
        return template.format(str(error))


class AgentRepositoryError(Exception):
    """Exception raised when an agent repository is not found."""
