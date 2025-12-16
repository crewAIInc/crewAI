from unittest.mock import MagicMock, patch

import pytest
from github import GithubException

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.github_loader import GithubLoader
from crewai_tools.rag.source_content import SourceContent


class TestGithubLoader:
    def setup_mock_repo(
        self,
        full_name: str = "owner/repo",
        description: str = "Test repo",
        language: str = "Python",
        stars: int = 100,
        forks: int = 10,
    ) -> MagicMock:
        repo = MagicMock()
        repo.full_name = full_name
        repo.description = description
        repo.language = language
        repo.stargazers_count = stars
        repo.forks_count = forks

        readme = MagicMock()
        readme.decoded_content = b"# Test README\n\nThis is a test."
        repo.get_readme.return_value = readme

        content_file = MagicMock()
        content_file.path = "README.md"
        content_file.type = "file"
        repo.get_contents.return_value = [content_file]

        repo.get_pulls.return_value = []
        repo.get_issues.return_value = []

        return repo

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_load_public_repo_without_token(self, mock_github_class: MagicMock) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo()

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/repo"),
            metadata={"content_types": ["repo", "code"]},
        )

        assert isinstance(result, LoaderResult)
        assert "owner/repo" in result.content
        mock_github_class.assert_called_once()
        call_args = mock_github_class.call_args
        assert call_args == ((None,),) or call_args == ((),)

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_load_with_token_passes_token_to_github(
        self, mock_github_class: MagicMock
    ) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo()

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/private-repo"),
            metadata={"gh_token": "ghp_test_token_123", "content_types": ["repo"]},
        )

        assert isinstance(result, LoaderResult)
        mock_github_class.assert_called_once_with("ghp_test_token_123")

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_private_repo_access_fails_without_token(
        self, mock_github_class: MagicMock
    ) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.side_effect = GithubException(
            404, {"message": "Not Found"}, None
        )

        loader = GithubLoader()
        with pytest.raises(ValueError, match="Unable to access repository"):
            loader.load(
                SourceContent("https://github.com/owner/private-repo"),
                metadata={"content_types": ["repo"]},
            )

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_private_repo_access_succeeds_with_token(
        self, mock_github_class: MagicMock
    ) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo(
            full_name="owner/private-repo"
        )

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/private-repo"),
            metadata={"gh_token": "ghp_valid_token", "content_types": ["repo"]},
        )

        assert isinstance(result, LoaderResult)
        assert "owner/private-repo" in result.content
        mock_github_class.assert_called_once_with("ghp_valid_token")

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_load_with_all_content_types(
        self, mock_github_class: MagicMock
    ) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github

        repo = self.setup_mock_repo()

        pr = MagicMock()
        pr.number = 1
        pr.title = "Test PR"
        pr.body = "PR description"
        repo.get_pulls.return_value = [pr]

        issue = MagicMock()
        issue.number = 1
        issue.title = "Test Issue"
        issue.body = "Issue description"
        issue.pull_request = None
        repo.get_issues.return_value = [issue]

        mock_github.get_repo.return_value = repo

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/repo"),
            metadata={"content_types": ["repo", "code", "pr", "issue"]},
        )

        assert "Repository: owner/repo" in result.content
        assert "README" in result.content
        assert "Test PR" in result.content
        assert "Test Issue" in result.content

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_invalid_github_url(self, mock_github_class: MagicMock) -> None:
        loader = GithubLoader()
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            loader.load(SourceContent("https://gitlab.com/owner/repo"))

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_invalid_repo_url_format(self, mock_github_class: MagicMock) -> None:
        loader = GithubLoader()
        with pytest.raises(ValueError, match="Invalid GitHub repository URL"):
            loader.load(SourceContent("https://github.com/owner"))

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_default_content_types(self, mock_github_class: MagicMock) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo()

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/repo"),
            metadata={},
        )

        assert "Repository: owner/repo" in result.content
        assert "README" in result.content

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_metadata_in_result(self, mock_github_class: MagicMock) -> None:
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo()

        loader = GithubLoader()
        result = loader.load(
            SourceContent("https://github.com/owner/repo"),
            metadata={"content_types": ["repo"]},
        )

        assert result.metadata["source"] == "https://github.com/owner/repo"
        assert result.metadata["repo"] == "owner/repo"
        assert result.metadata["content_types"] == ["repo"]
