"""Tests for DataTypes content classification."""

from pathlib import Path

import pytest

from crewai_tools.rag.chunkers.base_chunker import RecursiveCharacterTextSplitter
from crewai_tools.rag.chunkers.structured_chunker import JsonChunker
from crewai_tools.rag.data_types import DataType, DataTypes
from crewai_tools.rag.loaders.jsonl_loader import JSONLLoader


class TestDataTypeChunkers:
    @pytest.mark.parametrize(
        ("data_type", "text"),
        [
            (DataType.TEXT, "a\nb"),
            (DataType.TEXT, "Para one.\n\nPara two."),
            (DataType.JSONL, '{"a": 1}\n{"b": 2}'),
        ],
    )
    def test_kept_separators_are_not_duplicated(
        self, data_type: DataType, text: str
    ) -> None:
        assert data_type.get_chunker().chunk(text) == [text]

    def test_kept_separators_at_chunk_boundary(self) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)

        assert splitter.split_text("a\nb\nc\nd") == ["a\nb\nc", "\nc\nd"]

    @pytest.mark.parametrize("separator", ["\n", "\n\n", ", ", " "])
    def test_removed_separators_are_reinserted(self, separator: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            separators=[separator], keep_separator=False
        )
        text = separator.join(["a", "b", "c"])

        assert splitter.split_text(text) == [text]


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

    def test_repository_named_docs_is_github(self) -> None:
        # A "docs" path segment on github.com is a repository name, not a DocsSite.
        assert (
            DataTypes.from_content("https://github.com/crewAIInc/docs")
            == DataType.GITHUB
        )

    def test_github_docs_host_is_still_a_docs_site(self) -> None:
        assert (
            DataTypes.from_content("https://docs.github.com/en/rest")
            == DataType.DOCS_SITE
        )


class TestDataTypesDocsSite:
    """Docs detection keeps working for non-GitHub hosts."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/docs/intro",
            "https://docs.example.com/guide",
        ],
    )
    def test_docs_urls_are_docs_sites(self, url: str) -> None:
        assert DataTypes.from_content(url) == DataType.DOCS_SITE


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
