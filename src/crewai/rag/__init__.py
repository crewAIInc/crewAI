"""RAG (Retrieval-Augmented Generation) infrastructure for CrewAI."""

import sys
import importlib
from types import ModuleType
from typing import Any

from crewai.rag.config.utils import set_rag_config


class _RagModule(ModuleType):
    """Module wrapper to intercept attribute setting for config."""

    def __init__(self, module_name: str):
        """Initialize the module wrapper.

        Args:
            module_name: Name of the module.
        """
        super().__init__(module_name)
        self.__path__ = __path__
        self.__file__ = __file__

    def __setattr__(self, name: str, value: Any) -> None:
        """Set module attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name == "config":
            set_rag_config(value)
        else:
            super().__setattr__(name, value)

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
        except ImportError:
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")


sys.modules[__name__] = _RagModule(__name__)
