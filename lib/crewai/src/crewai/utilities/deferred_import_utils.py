"""Lazy loader for Python packages.

Makes it easy to load subpackages and functions on demand.

Pulled from https://github.com/scientific-python/lazy-loader/blob/main/src/lazy_loader/__init__.py,
modernized a little.
"""

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import importlib
import importlib.metadata
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import threading
import types
from typing import Any, NoReturn
import warnings

import packaging.requirements


_threadlock = threading.Lock()


@dataclass(frozen=True, slots=True)
class _FrameData:
    """Captured stack frame information for delayed error reporting."""

    filename: str
    lineno: int
    function: str
    code_context: Sequence[str] | None


def attach(
    package_name: str,
    submodules: set[str] | None = None,
    submod_attrs: dict[str, list[str]] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """Attach lazily loaded submodules, functions, or other attributes.

    Replaces a package's `__getattr__`, `__dir__`, and `__all__` such that
    imports work normally but occur upon first use.

    Example:
        __getattr__, __dir__, __all__ = lazy.attach(
            __name__, ["mysubmodule"], {"foo": ["someattr"]}
        )

    Args:
        package_name: The package name, typically ``__name__``.
        submodules: Set of submodule names to attach.
        submod_attrs: Mapping of submodule names to lists of attributes.
            These attributes are imported as they are used.

    Returns:
        A tuple of (__getattr__, __dir__, __all__) to assign in the package.
    """
    submod_attrs = submod_attrs or {}
    submodules = set(submodules) if submodules else set()
    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }
    __all__ = sorted(submodules | attr_to_modules.keys())

    def __getattr__(name: str) -> Any:  # noqa: N807
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        if name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            submod = importlib.import_module(submod_path)
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__() -> list[str]:  # noqa: N807
        return __all__.copy()

    if os.environ.get("EAGER_IMPORT"):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, __all__.copy()


class DelayedImportErrorModule(types.ModuleType):
    """Module type that delays raising ModuleNotFoundError until attribute access.

    Captures stack frame data to provide helpful error messages showing where
    the original import was attempted.
    """

    def __init__(
        self,
        frame_data: _FrameData,
        *args: Any,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the delayed error module.

        Args:
            frame_data: Captured frame information for error reporting.
            *args: Positional arguments passed to ModuleType.
            message: The error message to display when accessed.
            **kwargs: Keyword arguments passed to ModuleType.
        """
        self._frame_data = frame_data
        self._message = message
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> NoReturn:
        """Raise ModuleNotFoundError with detailed context on any attribute access."""
        frame = self._frame_data
        code = "".join(frame.code_context) if frame.code_context else ""
        raise ModuleNotFoundError(
            f"{self._message}\n\n"
            "This error is lazily reported, having originally occurred in\n"
            f"  File {frame.filename}, line {frame.lineno}, in {frame.function}\n\n"
            f"----> {code.strip()}"
        )


def load(
    fullname: str,
    *,
    require: str | None = None,
    error_on_import: bool = False,
    suppress_warning: bool = False,
) -> types.ModuleType:
    """Return a lazily imported proxy for a module.

    The proxy module delays actual import until first attribute access.

    Example:
        np = lazy.load("numpy")

        def myfunc():
            np.norm(...)

    Warning:
        Lazily loading subpackages causes the parent package to be eagerly
        loaded. Use `lazy_loader.attach` instead for subpackages.

    Args:
        fullname: The full name of the module to import (e.g., "scipy").
        require: A PEP-508 dependency requirement (e.g., "numpy >=1.24").
            If specified, raises an error if the installed version doesn't match.
        error_on_import: If True, raise import errors immediately.
            If False (default), delay errors until module is accessed.
        suppress_warning: If True, suppress the warning when loading subpackages.

    Returns:
        A proxy module that loads on first attribute access.
    """
    with _threadlock:
        module = sys.modules.get(fullname)

        # Most common, short-circuit
        if module is not None and require is None:
            return module

        have_module = module is not None

        if not suppress_warning and "." in fullname:
            msg = (
                "subpackages can technically be lazily loaded, but it causes the "
                "package to be eagerly loaded even if it is already lazily loaded. "
                "So, you probably shouldn't use subpackages with this lazy feature."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        spec = None

        if not have_module:
            spec = importlib.util.find_spec(fullname)
            have_module = spec is not None

        if not have_module:
            not_found_message = f"No module named '{fullname}'"
        elif require is not None:
            try:
                have_module = _check_requirement(require)
            except ModuleNotFoundError as e:
                raise ValueError(
                    f"Found module '{fullname}' but cannot test "
                    "requirement '{require}'. "
                    "Requirements must match distribution name, not module name."
                ) from e

            not_found_message = f"No distribution can be found matching '{require}'"

        if not have_module:
            if error_on_import:
                raise ModuleNotFoundError(not_found_message)

            parent = inspect.stack()[1]
            frame_data = _FrameData(
                filename=parent.filename,
                lineno=parent.lineno,
                function=parent.function,
                code_context=parent.code_context,
            )
            del parent
            return DelayedImportErrorModule(
                frame_data,
                "DelayedImportErrorModule",
                message=not_found_message,
            )

        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module

            if spec.loader is not None:
                loader = importlib.util.LazyLoader(spec.loader)
                loader.exec_module(module)

        if module is None:
            raise ModuleNotFoundError(f"No module named '{fullname}'")

    return module


def _check_requirement(require: str) -> bool:
    """Verify that a package requirement is satisfied.

    Args:
        require: A dependency requirement as defined in PEP-508.

    Returns:
        True if the installed version matches the requirement, False otherwise.

    Raises:
        ModuleNotFoundError: If the package is not installed.
    """
    req = packaging.requirements.Requirement(require)
    return req.specifier.contains(
        importlib.metadata.version(req.name),
        prereleases=True,
    )


@dataclass
class _StubVisitor(ast.NodeVisitor):
    """AST visitor to parse a stub file for submodules and submod_attrs."""

    _submodules: set[str] = field(default_factory=set)
    _submod_attrs: dict[str, list[str]] = field(default_factory=dict)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit an ImportFrom node and extract submodule/attribute information.

        Args:
            node: The AST ImportFrom node to visit.

        Raises:
            ValueError: If the import is not a relative import or uses star import.
        """
        if node.level != 1:
            raise ValueError(
                "Only within-module imports are supported (`from .* import`)"
            )
        names = [alias.name for alias in node.names]
        if node.module:
            if "*" in names:
                raise ValueError(
                    f"lazy stub loader does not support star import "
                    f"`from {node.module} import *`"
                )
            self._submod_attrs.setdefault(node.module, []).extend(names)
        else:
            self._submodules.update(names)


def attach_stub(
    package_name: str,
    filename: str,
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """Attach lazily loaded submodules and functions from a type stub.

    Parses a `.pyi` stub file to infer submodules and attributes. This allows
    static type checkers to find imports while providing lazy loading at runtime.

    Args:
        package_name: The package name, typically ``__name__``.
        filename: Path to `.py` file with an adjacent `.pyi` file.
            Typically use ``__file__``.

    Returns:
        A tuple of (__getattr__, __dir__, __all__) to assign in the package.

    Raises:
        ValueError: If stub file is not found or contains invalid imports.
    """
    path = Path(filename)
    stubfile = path if path.suffix == ".pyi" else path.with_suffix(".pyi")

    if not stubfile.exists():
        raise ValueError(f"Cannot load imports from non-existent stub {stubfile!r}")

    visitor = _StubVisitor()
    visitor.visit(ast.parse(stubfile.read_text()))
    return attach(package_name, visitor._submodules, visitor._submod_attrs)


def lazy_exports_stub(package_name: str, filename: str) -> None:
    """Install lazy loading on a module based on its .pyi stub file.

    Parses the adjacent `.pyi` stub file to determine what to export lazily.
    Type checkers see the stub, runtime gets lazy loading.

    Example:
        # __init__.py
        from crewai.utilities.lazy import lazy_exports_stub
        lazy_exports_stub(__name__, __file__)

        # __init__.pyi
        from .config import ChromaDBConfig, ChromaDBSettings
        from .types import EmbeddingType

    Args:
        package_name: The package name, typically ``__name__``.
        filename: Path to the module file, typically ``__file__``.
    """
    __getattr__, __dir__, __all__ = attach_stub(package_name, filename)
    module = sys.modules[package_name]
    module.__getattr__ = __getattr__  # type: ignore[method-assign]
    module.__dir__ = __dir__  # type: ignore[method-assign]
    module.__dict__["__all__"] = __all__


def lazy_exports(
    package_name: str,
    submod_attrs: dict[str, list[str]],
    submodules: set[str] | None = None,
) -> None:
    """Install lazy loading on a module.

    Example:
        from crewai.utilities.lazy import lazy_exports

        lazy_exports(__name__, {
            'config': ['ChromaDBConfig', 'ChromaDBSettings'],
            'types': ['EmbeddingType'],
        })

    Args:
        package_name: The package name, typically ``__name__``.
        submod_attrs: Mapping of submodule names to lists of attributes.
        submodules: Optional set of submodule names to expose directly.
    """
    __getattr__, __dir__, __all__ = attach(package_name, submodules, submod_attrs)
    module = sys.modules[package_name]
    module.__getattr__ = __getattr__  # type: ignore[method-assign]
    module.__dir__ = __dir__  # type: ignore[method-assign]
    module.__dict__["__all__"] = __all__
