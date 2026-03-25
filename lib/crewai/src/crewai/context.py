import contextvars


_platform_integration_token: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("platform_integration_token", default=None)
)


def get_platform_integration_token() -> str | None:
    """Get the platform integration token from the current context."""
    return _platform_integration_token.get()


_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None
)


def set_current_task_id(task_id: str | None) -> contextvars.Token[str | None]:
    """Set the current task ID in the context. Returns a token for reset."""
    return _current_task_id.set(task_id)


def reset_current_task_id(token: contextvars.Token[str | None]) -> None:
    """Reset the current task ID to its previous value."""
    _current_task_id.reset(token)


def get_current_task_id() -> str | None:
    """Get the current task ID from the context."""
    return _current_task_id.get()
