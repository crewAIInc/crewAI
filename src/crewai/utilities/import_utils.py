"""Import utilities for optional dependencies."""

import importlib
from types import ModuleType
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, TypeAdapter
from typing_extensions import deprecated


@deprecated(
    "Not needed when using `crewai.utilities.import_utils.import_and_validate_definition`"
)
class OptionalDependencyError(ImportError):
    """Exception raised when an optional dependency is not installed."""


@deprecated(
    "Use `crewai.utilities.import_utils.import_and_validate_definition` instead."
)
def require(name: str, *, purpose: str, attr: str | None = None) -> ModuleType | Any:
    """Import a module, optionally returning a specific attribute.

    Args:
        name: The module name to import.
        purpose: Description of what requires this dependency.
        attr: Optional attribute name to get from the module.

    Returns:
        The imported module or the specified attribute.

    Raises:
        OptionalDependencyError: If the module is not installed.
        AttributeError: If the specified attribute doesn't exist.
    """
    try:
        module = importlib.import_module(name)
        if attr is not None:
            return getattr(module, attr)
        return module
    except ImportError as exc:
        package_name = name.split(".")[0]
        raise OptionalDependencyError(
            f"{purpose} requires the optional dependency '{name}'.\n"
            f"Install it with: uv add {package_name}"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(f"Module '{name}' has no attribute '{attr}'") from exc


def validate_import_path(v: str) -> Any:
    """Import and return the class/function from the import path.

    Args:
        v: Import path string in the format 'module.path.ClassName'.

    Returns:
        The imported class or function.

    Raises:
        ValueError: If the import path is malformed or the module cannot be imported.
    """
    module_path, _, attr = v.rpartition(".")
    if not module_path or not attr:
        raise ValueError(f"import_path '{v}' must be of the form 'module.ClassName'")

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        parts = module_path.split(".")
        if not parts:
            raise ValueError(f"Malformed import path: '{v}'") from exc
        package = parts[0]
        raise ValueError(
            f"Package '{package}' could not be imported. Install it with: uv add {package}"
        ) from exc

    if not hasattr(mod, attr):
        raise ValueError(f"Attribute '{attr}' not found in module '{module_path}'")
    return getattr(mod, attr)


ImportedDefinition: TypeAlias = Annotated[Any, AfterValidator(validate_import_path)]
adapter = TypeAdapter(ImportedDefinition)


def import_and_validate_definition(v: str) -> Any:
    """Pydantic-compatible function to import a class/function from a string path.

    Args:
        v: Import path string in the format 'module.path.ClassName'.
    Returns:
        The imported class or function
    """
    return adapter.validate_python(v)
