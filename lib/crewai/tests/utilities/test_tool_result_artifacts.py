from pathlib import Path

from crewai.utilities.tool_result_artifacts import ToolResultArtifactStore


def test_store_returns_opaque_identifier_and_preserves_content(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)

    artifact = store.store("first line\nsecond line", estimated_token_count=9)

    assert artifact.artifact_id.startswith("tool-result-")
    assert artifact.directory == store.directory
    assert artifact.relative_path == Path(artifact.artifact_id) / "result.md"
    assert artifact.file_path.name == "result.md"
    assert artifact.file_path.read_text(encoding="utf-8") == "first line\nsecond line"
    assert artifact.line_count == 2
    assert artifact.estimated_token_count == 9

    store.cleanup()


def test_each_result_has_a_separate_private_file(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    first = store.store("first result", estimated_token_count=3)
    second = store.store("second result", estimated_token_count=4)

    assert first.artifact_id != second.artifact_id
    assert first.file_path.parent != second.file_path.parent
    assert first.file_path.read_text(encoding="utf-8") == "first result"
    assert second.file_path.read_text(encoding="utf-8") == "second result"

    store.cleanup()


def test_store_keeps_a_bounded_preview_of_the_first_ten_lines(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    content = "".join(f"line {index}\n" for index in range(12))

    artifact = store.store(content, estimated_token_count=20)

    assert artifact.preview == "".join(f"line {index}\n" for index in range(10))
    assert artifact.preview_truncated is True

    store.cleanup()


def test_store_caps_a_single_long_preview_line(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)

    artifact = store.store("x" * 2_001, estimated_token_count=501)

    assert len(artifact.preview) == 2_000
    assert artifact.preview_truncated is True

    store.cleanup()


def test_cleanup_removes_execution_directory_and_is_idempotent(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    store.store("content", estimated_token_count=2)

    store.cleanup()
    store.cleanup()

    assert list(tmp_path.iterdir()) == []


def test_store_line_count_preserves_unicode_separators(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    # U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR should not be split as universal newlines
    content = '\u2028'.join(['part'] * 100) + '\ntail\n'
    artifact = store.store(content, estimated_token_count=50)

    assert artifact.line_count == 2
    assert artifact.preview_truncated is False
    assert '\u2028' in artifact.preview

    store.cleanup()


def test_store_line_count_splits_carriage_return_only_lines(tmp_path: Path) -> None:
    store = ToolResultArtifactStore(base_dir=tmp_path)
    content = "\r".join(f"line {index}" for index in range(12))
    artifact = store.store(content, estimated_token_count=20)

    assert artifact.line_count == 12
    assert artifact.preview_truncated is True

    store.cleanup()
