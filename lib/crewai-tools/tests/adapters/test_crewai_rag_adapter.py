from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.data_types import DataType


class TestCrewAIRagAdapterKwargsPassthrough:
    """Tests to verify that kwargs (including metadata with gh_token) are passed to loaders."""

    @patch("crewai_tools.adapters.crewai_rag_adapter.get_rag_client")
    def test_add_passes_kwargs_to_loader(
        self, mock_get_rag_client: MagicMock
    ) -> None:
        """Test that kwargs are passed through to the loader.load() method.

        This is a regression test for GitHub issue #4088 where GithubSearchTool
        was unable to access private repositories because the gh_token in metadata
        was not being passed to the GithubLoader.
        """
        from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

        mock_client = MagicMock()
        mock_get_rag_client.return_value = mock_client
        mock_client.search.return_value = []

        adapter = CrewAIRagAdapter(collection_name="test_collection")

        mock_loader = MagicMock()
        mock_loader.load.return_value = LoaderResult(
            content="Test content",
            metadata={"source": "https://github.com/owner/repo"},
            doc_id="test_doc_id",
        )

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = ["Test content"]

        test_metadata = {"gh_token": "ghp_test_token", "content_types": ["repo"]}

        with patch.object(DataType.GITHUB, "get_loader", return_value=mock_loader):
            with patch.object(DataType.GITHUB, "get_chunker", return_value=mock_chunker):
                adapter.add(
                    "https://github.com/owner/repo",
                    data_type=DataType.GITHUB,
                    metadata=test_metadata,
                )

        mock_loader.load.assert_called_once()
        call_kwargs = mock_loader.load.call_args[1]
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["gh_token"] == "ghp_test_token"
        assert call_kwargs["metadata"]["content_types"] == ["repo"]

    @patch("crewai_tools.adapters.crewai_rag_adapter.get_rag_client")
    def test_add_passes_all_kwargs_to_loader(
        self, mock_get_rag_client: MagicMock
    ) -> None:
        """Test that all kwargs are passed through to the loader."""
        from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

        mock_client = MagicMock()
        mock_get_rag_client.return_value = mock_client
        mock_client.search.return_value = []

        adapter = CrewAIRagAdapter(collection_name="test_collection")

        mock_loader = MagicMock()
        mock_loader.load.return_value = LoaderResult(
            content="Test content",
            metadata={"source": "test"},
            doc_id="test_doc_id",
        )

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = ["Test content"]

        with patch.object(DataType.TEXT, "get_loader", return_value=mock_loader):
            with patch.object(DataType.TEXT, "get_chunker", return_value=mock_chunker):
                adapter.add(
                    "Some text content",
                    data_type=DataType.TEXT,
                    metadata={"custom_key": "custom_value"},
                )

        mock_loader.load.assert_called_once()
        call_kwargs = mock_loader.load.call_args[1]
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["custom_key"] == "custom_value"


class TestGithubSearchToolPrivateRepoAccess:
    """Integration tests for GithubSearchTool private repository access.

    These tests verify the fix for GitHub issue #4088.
    """

    def setup_mock_repo(
        self,
        full_name: str = "owner/repo",
        description: str = "Test repo",
    ) -> MagicMock:
        mock_repo = MagicMock()
        mock_repo.full_name = full_name
        mock_repo.description = description
        mock_repo.language = "Python"
        mock_repo.stargazers_count = 10
        mock_repo.forks_count = 2

        readme = MagicMock()
        readme.decoded_content = b"# README"
        mock_repo.get_readme.return_value = readme
        mock_repo.get_contents.return_value = []
        mock_repo.get_pulls.return_value = []
        mock_repo.get_issues.return_value = []

        return mock_repo

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_github_search_tool_passes_token_to_loader(
        self, mock_github_class: MagicMock
    ) -> None:
        """Test that GithubSearchTool passes gh_token through to GithubLoader.

        This is the main regression test for issue #4088.
        This test directly tests the GithubLoader to verify the token is passed.
        """
        from crewai_tools.rag.loaders.github_loader import GithubLoader
        from crewai_tools.rag.source_content import SourceContent

        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo(
            full_name="owner/private-repo"
        )

        loader = GithubLoader()
        loader.load(
            SourceContent("https://github.com/owner/private-repo"),
            metadata={"gh_token": "ghp_test_private_token", "content_types": ["repo"]},
        )

        mock_github_class.assert_called_with("ghp_test_private_token")

    @patch("crewai_tools.rag.loaders.github_loader.Github")
    def test_github_search_tool_without_token_uses_public_access(
        self, mock_github_class: MagicMock
    ) -> None:
        """Test that GithubSearchTool without token uses public GitHub access."""
        from crewai_tools.rag.loaders.github_loader import GithubLoader
        from crewai_tools.rag.source_content import SourceContent

        mock_github = MagicMock()
        mock_github_class.return_value = mock_github
        mock_github.get_repo.return_value = self.setup_mock_repo(
            full_name="owner/public-repo"
        )

        loader = GithubLoader()
        loader.load(
            SourceContent("https://github.com/owner/public-repo"),
            metadata={"content_types": ["repo"]},
        )

        mock_github_class.assert_called_once()
        call_args = mock_github_class.call_args
        assert call_args == ((None,),) or call_args == ((),)
