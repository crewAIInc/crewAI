"""Shared test configuration for CrewAI's core package."""

import sys

import pytest


@pytest.fixture(autouse=True)
def skip_litellm_paths_on_python_310(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip LiteLLM paths unsupported by patched LiteLLM releases on Python 3.10."""
    if sys.version_info >= (3, 11):
        return

    def skip_litellm() -> bool:
        pytest.skip("Patched LiteLLM releases support this path only on Python 3.11+.")

    monkeypatch.setattr("crewai.llm._ensure_litellm", skip_litellm)
