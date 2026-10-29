"""MemorySlice metadata queries treat paths as relative to each slice root."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai.memory.memory_scope import MemorySlice
from crewai.memory.unified_memory import Memory


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> MemorySlice:
    """A slice with both in-scope and similarly named sibling records."""
    embedder = MagicMock(side_effect=lambda texts: [[0.1] * 4 for _ in texts])
    memory = Memory(
        storage=str(tmp_path_factory.mktemp("slice-memory")),
        llm=MagicMock(),
        embedder=embedder,
    )
    for scope, category in [
        ("/team/project/phase", "team-project"),
        ("/company/project/phase", "company-project"),
        ("/teamproject/phase", "outside-team"),
        ("/companyproject/phase", "outside-company"),
    ]:
        memory.remember(
            category,
            scope=scope,
            categories=[category],
            importance=0.5,
        )
    return MemorySlice(memory=memory, scopes=["/team", "/company"])


@pytest.mark.parametrize("path", ["project", "/project"])
def test_list_scopes_stays_under_each_root(view: MemorySlice, path: str) -> None:
    """A relative path cannot reveal a similarly named sibling's child scopes."""
    assert set(view.list_scopes(path)) == {
        "/team/project/phase",
        "/company/project/phase",
    }


@pytest.mark.parametrize("path", ["project", "/project"])
def test_info_stays_under_each_root(view: MemorySlice, path: str) -> None:
    """Scope info counts and categories come from the requested subtrees."""
    info = view.info(path)
    assert info.record_count == 2
    assert set(info.categories) == {"team-project", "company-project"}


@pytest.mark.parametrize("path", ["project", "/project"])
def test_categories_stay_under_each_root(view: MemorySlice, path: str) -> None:
    """Relative category queries cannot read sibling-scope categories."""
    assert view.list_categories(path) == {"team-project": 1, "company-project": 1}


def test_unanchored_slice_root_keeps_existing_records(tmp_path: Path) -> None:
    """Joining a child path must retain a root's unanchored spelling."""
    embedder = MagicMock(side_effect=lambda texts: [[0.1] * 4 for _ in texts])
    memory = Memory(
        storage=str(tmp_path / "memory"), llm=MagicMock(), embedder=embedder
    )
    memory.remember(
        "Unanchored record",
        scope="team/project/phase",
        categories=["unanchored"],
        importance=0.5,
    )
    view = MemorySlice(memory=memory, scopes=["team"])

    assert view.list_scopes("project") == ["team/project/phase"]
    assert view.list_categories("project") == {"unanchored": 1}
    assert view.list_categories() == {"unanchored": 1}
