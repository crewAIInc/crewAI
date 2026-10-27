"""Shared test configuration for CrewAI's core package."""

import builtins
import sys

import pytest


@pytest.fixture(autouse=True)
def skip_litellm_paths_on_python_310(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip LiteLLM paths unsupported by patched LiteLLM releases on Python 3.10."""
    if sys.version_info >= (3, 11):
        return

    original_import = builtins.__import__

    def skip_litellm() -> bool:
        pytest.skip("Patched LiteLLM releases support this path only on Python 3.11+.")

    def skip_litellm_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "litellm" or name.startswith("litellm."):
            skip_litellm()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("crewai.llm._ensure_litellm", skip_litellm)
    monkeypatch.setattr(builtins, "__import__", skip_litellm_import)
