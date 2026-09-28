from pathlib import Path

from crewai_tools.tools.directory_read_tool.directory_read_tool import (
    DirectoryReadTool,
)


def test_lists_actual_path_when_descendant_repeats_base_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "workspace"
    repeated = base / base.relative_to(base.anchor)
    repeated.mkdir(parents=True)
    target = repeated / "sentinel.txt"
    target.write_text("content", encoding="utf-8")

    result = DirectoryReadTool()._run(directory=str(base))

    assert result == f"File paths: \n-{target}"
    assert target.exists()
