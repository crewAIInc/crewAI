"""Import-time placeholders for native packages missing under WebAssembly.

crewai imports ``chromadb`` types at module level for its knowledge and RAG
integrations. chromadb is compiled code with no Pyodide build, so this finder
serves placeholder modules in its place: imports succeed, and any attempt to
actually construct one of its objects raises an error naming the package.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from typing import Any


class UnavailableDependencyError(ImportError):
    """Raised when code uses a package that has no WebAssembly build."""


class _Placeholder:
    """Base for placeholder classes: usable in annotations, never instantiable."""

    _package = ""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise UnavailableDependencyError(
            f"{type(self).__name__} requires '{self._package}', which is not "
            "available in Pyodide/WebAssembly."
        )

    def __class_getitem__(cls, _item: Any) -> type[_Placeholder]:
        return cls

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.any_schema()


class _PlaceholderModule(ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        package = self.__name__.split(".")[0]
        placeholder = type(name, (_Placeholder,), {"_package": package})
        setattr(self, name, placeholder)
        return placeholder


class _PlaceholderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, packages: frozenset[str]) -> None:
        self._packages = packages

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".")[0] not in self._packages:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = _PlaceholderModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module: ModuleType) -> None:
        pass


def install_placeholders(packages: Sequence[str]) -> None:
    """Serve placeholders for each of ``packages`` that is not installed."""
    missing = frozenset(p for p in packages if importlib.util.find_spec(p) is None)
    if missing:
        sys.meta_path.append(_PlaceholderFinder(missing))
