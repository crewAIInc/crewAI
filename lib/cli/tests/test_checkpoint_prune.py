"""Retention count validation for the checkpoint prune command."""

from contextlib import closing
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch

from click.testing import CliRunner
from crewai_cli.cli import crewai
import pytest


_CHECKPOINT_IDS = ("first", "second", "third")


@pytest.fixture(params=["json", "sqlite"])
def checkpoint_location(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    if request.param == "json":
        location = tmp_path / "checkpoints"
        location.mkdir()
        for timestamp, checkpoint_id in enumerate(_CHECKPOINT_IDS, start=1):
            checkpoint = location / f"{checkpoint_id}.json"
            checkpoint.write_text("{}", encoding="utf-8")
            os.utime(checkpoint, (timestamp, timestamp))
    else:
        location = tmp_path / "checkpoints.db"
        with closing(sqlite3.connect(location)) as db, db:
            db.execute("CREATE TABLE checkpoints (id TEXT)")
            db.executemany(
                "INSERT INTO checkpoints VALUES (?)",
                [(checkpoint_id,) for checkpoint_id in _CHECKPOINT_IDS],
            )
    return location


def _checkpoint_ids(location: Path) -> set[str]:
    if location.is_dir():
        return {path.stem for path in location.glob("*.json")}
    with closing(sqlite3.connect(location)) as db:
        return {
            checkpoint_id
            for (checkpoint_id,) in db.execute("SELECT id FROM checkpoints")
        }


@pytest.mark.parametrize("dry_run", [False, True])
def test_prune_rejects_negative_keep(checkpoint_location: Path, dry_run: bool) -> None:
    args = [
        "checkpoint",
        "--location",
        str(checkpoint_location),
        "prune",
        "--keep",
        "-1",
    ]
    if dry_run:
        args.append("--dry-run")

    with patch("crewai_cli.checkpoint_cli._record_checkpoint_usage") as record_usage:
        result = CliRunner().invoke(crewai, args)

    assert result.exit_code == 2, (result.output, _checkpoint_ids(checkpoint_location))
    assert "--keep" in result.output
    assert _checkpoint_ids(checkpoint_location) == set(_CHECKPOINT_IDS)
    record_usage.assert_not_called()


@pytest.mark.parametrize("keep", [0, 2])
def test_prune_keeps_most_recent_checkpoints(
    checkpoint_location: Path, keep: int
) -> None:
    with patch("crewai_cli.checkpoint_cli._record_checkpoint_usage"):
        result = CliRunner().invoke(
            crewai,
            [
                "checkpoint",
                "--location",
                str(checkpoint_location),
                "prune",
                "--keep",
                str(keep),
            ],
        )

    assert result.exit_code == 0, result.output
    assert _checkpoint_ids(checkpoint_location) == set(
        _CHECKPOINT_IDS[len(_CHECKPOINT_IDS) - keep :]
    )
