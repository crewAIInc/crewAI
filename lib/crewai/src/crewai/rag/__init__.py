"""RAG (Retrieval-Augmented Generation) infrastructure for CrewAI."""

import importlib
import sys
from types import ModuleType
from typing import Any


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

    def __setattr__(self, name: str, value: Any) -> None:
        """Set module attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name == "config":
            if isinstance(value, ModuleType) and value is sys.modules.get(
                f"{self.__name__}.config"
            ):
                # importlib registers the crewai.rag.config submodule on the
                # package when it is first imported; store it as a plain
                # module attribute instead of treating it as a RAG config.
                return super().__setattr__(name, value)
            # Imported lazily so that `import crewai.rag` does not pull in the
            # provider config chain (chromadb, qdrant) at import time.
            from crewai.rag.config.utils import set_rag_config

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
