"""Compatibility shim: ``crewai.a2a`` re-exports :mod:`crewai_a2a`.

The package lives in the ``crewai-a2a`` distribution (install via the
``crewai[a2a]`` extra). This module aliases the old import path so existing
code using ``crewai.a2a.*`` keeps working.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType


try:
    import crewai_a2a as _crewai_a2a
except ImportError as exc:
    raise ImportError(
        "crewai.a2a requires the 'crewai-a2a' package. "
        "Install it with: pip install 'crewai[a2a]'"
    ) from exc


class _A2AAliasFinder(MetaPathFinder, Loader):
    _SRC = "crewai.a2a"
    _DST = "crewai_a2a"

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.startswith(self._SRC + "."):
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        target = self._DST + spec.name[len(self._SRC) :]
        module = importlib.import_module(target)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        return None


if not any(isinstance(f, _A2AAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _A2AAliasFinder())

for _attr in getattr(_crewai_a2a, "__all__", []):
    globals()[_attr] = getattr(_crewai_a2a, _attr)

__all__ = list(getattr(_crewai_a2a, "__all__", []))
