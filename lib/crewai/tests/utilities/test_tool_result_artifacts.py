from pathlib import Path

from crewai.utilities.tool_result_artifacts import ToolResultArtifactStore


def test_store_returns_opaque_identifier_and_preserves_content(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)

    artifact = store.store("first line\nsecond line")

    assert artifact.artifact_id.startswith("tool-result-")
    assert artifact.file_path.name == "result.md"
    assert artifact.file_path.read_text(encoding="utf-8") == "first line\nsecond line"

    store.cleanup()


def test_each_result_has_a_separate_private_file(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    first = store.store("first result")
    second = store.store("second result")

    assert first.artifact_id != second.artifact_id
    assert first.file_path.parent != second.file_path.parent
    assert first.file_path.read_text(encoding="utf-8") == "first result"
    assert second.file_path.read_text(encoding="utf-8") == "second result"

    store.cleanup()


def test_cleanup_removes_execution_directory_and_is_idempotent(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    store.store("content")

    store.cleanup()
    store.cleanup()

    assert list(tmp_path.iterdir()) == []
