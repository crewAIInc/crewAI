"""A writable MemorySlice must reject writes outside its scopes."""

from unittest.mock import Mock

from crewai.memory.memory_scope import MemorySlice


def test_sibling_scope_is_rejected():
    memory = Mock()
    view = MemorySlice(memory=memory, scopes=["/team/alpha"], read_only=False)

    assert view.remember("fact", scope="/team/alphabet") is None
    assert not memory.remember.called


def test_scope_within_slice_is_allowed():
    memory = Mock()
    view = MemorySlice(memory=memory, scopes=["/team/alpha"], read_only=False)

    view.remember("fact", scope="/team/alpha/sub")

    assert memory.remember.called
