"""Tests for the framework-owned deferred-result reader."""

from crewai.tools.agent_tools.deferred_tool_result_reader import (
    DeferredToolResultReader,
)


def test_reads_only_an_allowed_artifact(tmp_path):
    artifact = tmp_path / "result.md"
    artifact.write_text("first\nsecond\nthird\n")
    other_file = tmp_path / "other.md"
    other_file.write_text("must not be readable")
    reader = DeferredToolResultReader()
    reader.allow_file(str(artifact), "artifact/result.md")

    assert reader.run(
        file_path="artifact/result.md", start_line=2, line_count=1
    ) == "second\n"
    assert "not available" in reader.run(file_path=str(other_file))


def test_revokes_an_artifact_alias(tmp_path):
    artifact = tmp_path / "result.md"
    artifact.write_text("content")
    reader = DeferredToolResultReader()
    reader.allow_file(str(artifact), "artifact/result.md")

    reader.disallow_file("artifact/result.md")

    assert "not available" in reader.run(file_path="artifact/result.md")


def test_wrong_path_lists_the_readable_artifact_paths(tmp_path):
    artifact = tmp_path / "result.md"
    artifact.write_text("artifact content")
    reader = DeferredToolResultReader()
    reader.allow_file(str(artifact), "tool-result-abc/result.md")

    message = reader.run(file_path="./tool-result-abc/result.md")

    assert message == (
        "Error: This temporary tool-result artifact is not available. "
        "Pass one of these paths exactly as given: `tool-result-abc/result.md`. "
        "Artifacts from earlier runs are deleted."
    )


def test_path_with_nothing_readable_says_artifacts_end_with_their_run(tmp_path):
    artifact = tmp_path / "result.md"
    artifact.write_text("artifact content")
    reader = DeferredToolResultReader()
    reader.allow_file(str(artifact), "tool-result-abc/result.md")
    reader.disallow_file("tool-result-abc/result.md")

    message = reader.run(file_path="tool-result-abc/result.md")

    assert message == (
        "Error: This temporary tool-result artifact is not available. "
        "Artifacts are deleted when the run that created them ends."
    )
