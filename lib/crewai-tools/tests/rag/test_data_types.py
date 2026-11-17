"""Tests for DataTypes content classification."""

from pathlib import Path

import pytest

from crewai_tools.rag.chunkers.structured_chunker import JsonChunker
from crewai_tools.rag.data_types import DataType, DataTypes
from crewai_tools.rag.loaders.jsonl_loader import JSONLLoader


class TestDataTypesFromContentGitHub:
    """GitHub URL detection must use hostname matching, not substrings."""

    def test_github_com_url(self) -> None:
        assert (
            DataTypes.from_content("https://github.com/crewai/crewai")
            == DataType.GITHUB
        )

    def test_github_subdomain_url(self) -> None:
        assert (
            DataTypes.from_content("https://gist.github.com/user/abc")
            == DataType.GITHUB
        )

    def test_spoofed_github_hostname_is_website(self) -> None:
        # Substring checks like `"github.com" in netloc` would misclassify this.
        assert (
            DataTypes.from_content("https://github.com.evil.example/crewai")
            == DataType.WEBSITE
        )

    def test_github_in_path_is_not_github(self) -> None:
        assert (
            DataTypes.from_content("https://example.com/github.com/repo")
            == DataType.WEBSITE
        )


class TestDataTypesJSONL:
    """JSONL uses the existing registry for explicit selection and discovery."""

    def test_explicit_type(self) -> None:
        assert DataType("jsonl") is DataType.JSONL
        assert isinstance(DataType.JSONL.get_loader(), JSONLLoader)
        assert isinstance(DataType.JSONL.get_chunker(), JsonChunker)

    @pytest.mark.parametrize(
        ("extension", "expected"),
        [(".jsonl", DataType.JSONL), (".ndjson", DataType.JSONL), (".json", DataType.JSON)],
    )
    @pytest.mark.parametrize("as_path", [True, False])
    def test_file_extension(
        self, tmp_path: Path, extension: str, expected: DataType, as_path: bool
    ) -> None:
        path = tmp_path / f"records{extension}"
        path.write_text('{"name": "first"}\n', encoding="utf-8")
        assert DataTypes.from_content(path if as_path else str(path)) is expected

    @pytest.mark.parametrize("extension", [".jsonl", ".ndjson"])
    @pytest.mark.parametrize("suffix", ["", "?download=1#records"])
    def test_url_extension(self, extension: str, suffix: str) -> None:
        assert (
            DataTypes.from_content(f"https://example.com/records{extension}{suffix}")
            is DataType.JSONL
        )

    def test_inline_content_remains_text_without_explicit_type(self) -> None:
        assert DataTypes.from_content('{"a": 1}\n{"b": 2}') is DataType.TEXT
