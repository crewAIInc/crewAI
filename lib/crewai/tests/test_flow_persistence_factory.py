"""Tests for the pluggable flow persistence factory seam.

We verify our own logic: that ``default_flow_persistence`` returns the
registered factory's result, and that it falls back to the built-in SQLite
persistence when no factory is registered.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

import crewai.flow.persistence.factory as factory
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset the factory around each test without clobbering preexisting state."""
    original = factory._factory
    factory.set_flow_persistence_factory(None)
    yield
    factory.set_flow_persistence_factory(original)


def test_default_uses_registered_factory():
    sentinel = SQLiteFlowPersistence()
    factory.set_flow_persistence_factory(lambda: sentinel)

    assert factory.default_flow_persistence() is sentinel


def test_default_falls_back_to_sqlite():
    assert isinstance(factory.default_flow_persistence(), SQLiteFlowPersistence)


def test_persist_decorator_honors_falsy_persistence():
    # @persist with an explicit but falsy FlowPersistence must keep it, not
    # replace it with the default via a truthiness check.
    class _FalsyPersistence(FlowPersistence):
        def __bool__(self) -> bool:
            return False

        def init_db(self) -> None:
            pass

        def save_state(
            self,
            flow_uuid: str,
            method_name: str,
            state_data: dict[str, Any] | BaseModel,
        ) -> None:
            pass

        def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
            return None

    falsy = _FalsyPersistence()

    @persist(persistence=falsy)
    class _DummyFlow:
        pass

    assert _DummyFlow.__flow_persistence_config__.persistence is falsy
