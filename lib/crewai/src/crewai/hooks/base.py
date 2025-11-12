from __future__ import annotations

from typing import TypeVar


# Type variable for hook context types
HookContextT = TypeVar("HookContextT")


def validate_hook_callable(hook: object, hook_type: str) -> None:
    """Validate that a hook is callable.

    Args:
        hook: The hook object to validate
        hook_type: Description of the hook type for error messages

    Raises:
        TypeError: If the hook is not callable
    """
    if not callable(hook):
        raise TypeError(f"{hook_type} must be callable, got {type(hook).__name__}")
