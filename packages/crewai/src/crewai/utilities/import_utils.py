"""Import utilities for optional dependencies."""

import importlib
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Exception raised when an optional dependency is not installed."""

    pass


def require(name: str, *, purpose: str) -> ModuleType:
    """Import a module, raising a helpful error if it's not installed.

    Args:
        name: The module name to import.
        purpose: Description of what requires this dependency.

    Returns:
        The imported module.

    Raises:
        OptionalDependencyError: If the module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"{purpose} requires the optional dependency '{name}'.\n"
            f"Install it with: uv add {name}"
        ) from exc
