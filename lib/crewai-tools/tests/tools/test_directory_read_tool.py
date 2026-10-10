from pathlib import Path
from unittest.mock import patch

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


def test_lists_files_from_filesystem_root(monkeypatch) -> None:
    root = Path.cwd().anchor
    monkeypatch.chdir(root)

    # Keep real path validation but do not traverse the host filesystem.
    with patch(
        "crewai_tools.tools.directory_read_tool.directory_read_tool.os.walk",
        return_value=[(root, [], ["sentinel.txt"])],
    ) as walk:
        result = DirectoryReadTool()._run(directory=root)

    walk.assert_called_once_with(root)
    assert result == f"File paths: \n-{Path(root) / 'sentinel.txt'}"
