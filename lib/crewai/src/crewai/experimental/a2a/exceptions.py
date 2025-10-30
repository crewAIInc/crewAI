"""Custom exceptions for A2A Agent Adapter."""


class A2AError(Exception):
    """Base exception for A2A adapter errors."""


class A2ATaskFailedError(A2AError):
    """Raised when A2A agent task fails or is rejected.

    This exception is raised when the A2A agent reports a task
    in the 'failed' or 'rejected' state.
    """


class A2AInputRequiredError(A2AError):
    """Raised when A2A agent requires additional input.

    This exception is raised when the A2A agent reports a task
    in the 'input_required' state, indicating that it needs more
    information to complete the task.
    """


class A2AConfigurationError(A2AError):
    """Raised when A2A adapter configuration is invalid.

    This exception is raised during initialization or setup when
    the adapter configuration is invalid or incompatible.
    """


class A2AConnectionError(A2AError):
    """Raised when connection to A2A agent fails.

    This exception is raised when the adapter cannot establish
    a connection to the A2A agent or when network errors occur.
    """


class A2AAuthenticationError(A2AError):
    """Raised when A2A agent requires authentication.

    This exception is raised when the A2A agent reports a task
    in the 'auth_required' state, indicating that authentication
    is needed before the task can continue.
    """


class A2ATaskCanceledError(A2AError):
    """Raised when A2A task is canceled.

    This exception is raised when the A2A agent reports a task
    in the 'canceled' state, indicating the task was canceled
    either by the user or the system.
    """
