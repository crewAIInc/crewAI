"""Backwards compatibility re-exports from crewai.files.

Deprecated: Import from crewai.files instead.
"""

import sys
from typing import Any

from typing_extensions import deprecated

import crewai.files as _files


@deprecated("crewai.utilities.files is deprecated. Import from crewai.files instead.")
class _DeprecatedModule:
    """Deprecated module wrapper."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_files, name)

    def __dir__(self) -> list[str]:
        return list(_files.__all__)


sys.modules[__name__] = _DeprecatedModule()  # type: ignore[assignment]
