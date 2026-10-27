"""Helpers for lazily exposing optional embedding provider implementations."""

from importlib import import_module
from typing import Any


def load_provider(name: str, exports: dict[str, str]) -> Any:
    """Load a provider implementation only when its package export is accessed."""
    module_path = exports.get(name)
    if module_path is None:
        raise AttributeError(name)
    return getattr(import_module(module_path), name)
