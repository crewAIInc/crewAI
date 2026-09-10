"""Tests for DataTypes content classification."""

from crewai_tools.rag.data_types import DataType, DataTypes


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
