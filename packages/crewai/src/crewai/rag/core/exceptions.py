"""Core exceptions for RAG module."""


class ClientMethodMismatchError(TypeError):
    """Raised when a method is called with the wrong client type.

    Typically used when a sync method is called with an async client,
    or vice versa.
    """

    def __init__(
        self, method_name: str, expected_client: str, alt_method: str, alt_client: str
    ) -> None:
        """Create a ClientMethodMismatchError.

        Args:
            method_name: Method that was called incorrectly.
            expected_client: Required client type.
            alt_method: Suggested alternative method.
            alt_client: Client type for the alternative method.
        """
        message = (
            f"Method {method_name}() requires a {expected_client}. "
            f"Use {alt_method}() for {alt_client}."
        )
        super().__init__(message)
