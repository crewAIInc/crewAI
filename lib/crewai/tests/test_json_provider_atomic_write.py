"""Atomic-write regression tests for the JSON checkpoint provider.

``JsonProvider`` used to write checkpoints directly to the final path, so a
crash mid-write left a truncated file at exactly the location ``prune`` keeps
(the newest) and a resume would read. Writes now go to a ``.tmp`` file that is
``os.replace``d into place, so the destination only ever holds complete data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from crewai.state.provider.json_provider import JsonProvider


def test_successful_write_publishes_complete_file(tmp_path: Path) -> None:
    provider = JsonProvider()
    location = provider.checkpoint('{"step": 1}', str(tmp_path))

    out = Path(location)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == {"step": 1}
    # The temp file is consumed by the replace, not left behind.
    assert list(tmp_path.rglob("*.tmp")) == []


@pytest.mark.asyncio
async def test_async_successful_write_publishes_complete_file(
    tmp_path: Path,
) -> None:
    provider = JsonProvider()
    location = await provider.acheckpoint('{"step": 1}', str(tmp_path))

    out = Path(location)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == {"step": 1}
    assert list(tmp_path.rglob("*.tmp")) == []


def test_failed_publish_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = JsonProvider()
    existing = Path(provider.checkpoint('{"step": 0}', str(tmp_path)))

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "crewai.state.provider.json_provider.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="disk full"):
        provider.checkpoint('{"step": 1}', str(tmp_path))

    # The failed write's destination was never created: the only checkpoint
    # file is the pre-existing one, with its original content intact.
    json_files = sorted(tmp_path.rglob("*.json"))
    assert [p.name for p in json_files] == [existing.name]
    assert existing.read_text(encoding="utf-8") == '{"step": 0}'


@pytest.mark.asyncio
async def test_async_failed_publish_leaves_no_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = JsonProvider()
    existing = Path(await provider.acheckpoint('{"step": 0}', str(tmp_path)))

    async def fail_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("aiofiles.os.replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        await provider.acheckpoint('{"step": 1}', str(tmp_path))

    json_files = sorted(tmp_path.rglob("*.json"))
    assert [p.name for p in json_files] == [existing.name]
    assert existing.read_text(encoding="utf-8") == '{"step": 0}'


def test_write_over_existing_file_stays_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rewrite of an existing checkpoint keeps the old data readable."""
    provider = JsonProvider()
    location = provider.checkpoint('{"step": 0}', str(tmp_path))

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "crewai.state.provider.json_provider.os.replace", fail_replace
    )

    # Force the same destination path by stubbing the unique-filename builder.
    monkeypatch.setattr(
        "crewai.state.provider.json_provider._build_path",
        lambda directory, branch="main", parent_id=None: Path(location),
    )
    with pytest.raises(OSError, match="disk full"):
        provider.checkpoint('{"step": 1 "truncated', str(tmp_path))

    # The restore path still sees the complete previous checkpoint.
    assert provider.from_checkpoint(location) == '{"step": 0}'
