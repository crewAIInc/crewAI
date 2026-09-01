from pathlib import Path

import tomli
from pytest import MonkeyPatch

from crewai_cli.update_crew import migrate_pyproject


def test_migrate_poetry_project_preserves_non_poetry_tool_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "legacy.toml"
    output_file = tmp_path / "migrated.toml"
    input_file.write_text(
        """
[tool.poetry]
name = "example-crew"
version = "0.1.0"
description = "An example crew"
authors = ["Ada Lovelace <ada@example.com>"]

[tool.poetry.dependencies]
python = ">=3.10,<3.14"
crewai = "^1.0.0"

[tool.crewai]
type = "crew"

[tool.ruff]
line-length = 88
""".strip(),
        encoding="utf-8",
    )

    migrate_pyproject(str(input_file), str(output_file))

    migrated = tomli.loads(output_file.read_text(encoding="utf-8"))
    assert migrated["project"]["name"] == "example-crew"
    assert migrated["tool"]["crewai"] == {"type": "crew"}
    assert migrated["tool"]["ruff"] == {"line-length": 88}
    assert "poetry" not in migrated["tool"]
