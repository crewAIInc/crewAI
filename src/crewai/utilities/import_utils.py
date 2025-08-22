"""Import utilities for optional dependencies."""

import importlib
from typing import Any


def require(name: str, instantiate_class: str | None = None) -> Any:
    """Import a module or class, raising a helpful error if it's not installed.

    Args:
        name: The module name to import.
        instantiate_class: Optional class name to instantiate from the module.

    Returns:
        The imported module or instantiated class.

    Raises:
        RuntimeError: If the module is not installed.
    """
    try:
        module = importlib.import_module(name)
        if instantiate_class:
            cls = getattr(module, instantiate_class)
            return cls()
        return module
    except ModuleNotFoundError as e:
        raise RuntimeError(f"{name} is required. `pip install {name}`") from e
