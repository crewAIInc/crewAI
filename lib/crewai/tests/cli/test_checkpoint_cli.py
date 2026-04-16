from __future__ import annotations

from pathlib import Path

from crewai.cli.checkpoint_cli import _info_json_latest, _list_json
from crewai.state.provider.json_provider import JsonProvider


def test_list_json_discovers_branch_subdirectories(tmp_path: Path) -> None:
    provider = JsonProvider()
    provider.checkpoint('{"entities": [], "event_record": {}}', str(tmp_path))
    provider.checkpoint(
        '{"entities": [], "event_record": {}}',
        str(tmp_path),
        branch="fork/exp1",
    )

    entries = _list_json(str(tmp_path))

    assert len(entries) == 2
    assert {Path(entry["path"]).parent.name for entry in entries} == {"main", "exp1"}


def test_info_json_latest_reads_latest_checkpoint_from_branch_subdirectory(
    tmp_path: Path,
) -> None:
    provider = JsonProvider()
    provider.checkpoint('{"entities": [], "event_record": {}}', str(tmp_path))
    latest_path = provider.checkpoint(
        '{"entities": [], "event_record": {}}',
        str(tmp_path),
        branch="fork/exp1",
    )

    latest = _info_json_latest(str(tmp_path))

    assert latest is not None
    assert latest["path"] == latest_path
