"""Tests for the CREWAI_EXPERIMENTAL gate on Skills Repository."""

from __future__ import annotations

import pytest

from crewai.experimental.skills._flag import (
    ExperimentalFeatureDisabledError,
    require_experimental_skills,
)
from crewai.experimental.skills.registry import resolve_registry_ref


def test_require_raises_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREWAI_EXPERIMENTAL", raising=False)
    with pytest.raises(ExperimentalFeatureDisabledError):
        require_experimental_skills()


def test_resolve_registry_ref_raises_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CREWAI_EXPERIMENTAL", raising=False)
    with pytest.raises(ExperimentalFeatureDisabledError):
        resolve_registry_ref("@acme/my-skill")


def test_require_passes_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_EXPERIMENTAL", "1")
    require_experimental_skills()