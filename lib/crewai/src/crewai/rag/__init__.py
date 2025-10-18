"""RAG (Retrieval-Augmented Generation) infrastructure for CrewAI."""

import importlib
import sys
from types import ModuleType
from typing import Any

from crewai.rag.config.types import RagConfigType
from crewai.rag.config.utils import set_rag_config

_module_path = __path__
_module_file = __file__


class _RagModule(ModuleType):
    """Module wrapper to intercept attribute setting for config."""

    __path__ = _module_path
    __file__ = _module_file

    def __init__(self, module_name: str):
        """Initialize the module wrapper.

        Args:
            module_name: Name of the module.
        """
        super().__init__(module_name)

    def __setattr__(self, name: str, value: RagConfigType) -> None:
        """Set module attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name == "config":
            return set_rag_config(value)
        raise AttributeError(f"Setting attribute '{name}' is not allowed.")

    def __getattr__(self, name: str) -> Any:
        """Get module attributes.

        Args:
            name: Attribute name.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If attribute doesn't exist.
        """
        try:
            return importlib.import_module(f"{self.__name__}.{name}")
        except ImportError as e:
            raise AttributeError(
                f"module '{self.__name__}' has no attribute '{name}'"
            ) from e


sys.modules[__name__] = _RagModule(__name__)
