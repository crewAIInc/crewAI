"""Shared mocks for the Azure CosmosDB NoSQL tool tests.

The tests do not require live ``azure-cosmos`` / ``azure-identity`` /
``openai`` installations. We register lightweight ``MagicMock`` stand-ins for
the modules in :mod:`sys.modules` so importing the tools succeeds and we can
assert against the recorded calls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _install_stub(module_path: str) -> MagicMock:
    """Register a MagicMock at ``module_path`` if no real module is present."""
    if module_path in sys.modules and not isinstance(
        sys.modules[module_path], MagicMock
    ):
        return sys.modules[module_path]  # real module present, leave it
    stub = sys.modules.get(module_path)
    if stub is None:
        stub = MagicMock(name=module_path)
        sys.modules[module_path] = stub
    return stub


# azure.cosmos / azure.core.credentials — only stub if azure-cosmos is missing.
try:  # pragma: no cover - presence depends on extras
    import azure.cosmos  # noqa: F401
except ImportError:  # pragma: no cover
    azure_pkg = _install_stub("azure")
    azure_cosmos = _install_stub("azure.cosmos")
    azure_pkg.cosmos = azure_cosmos
    azure_core = _install_stub("azure.core")
    azure_core_creds = _install_stub("azure.core.credentials")
    azure_pkg.core = azure_core
    azure_core.credentials = azure_core_creds

try:  # pragma: no cover
    import openai  # noqa: F401
except ImportError:  # pragma: no cover
    _install_stub("openai")
