"""Tests for path and URL validation in RagTool.add() — both positional and keyword args."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.rag.rag_tool import RagTool


@pytest.fixture()
def mock_rag_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[])
    return mock_client


@pytest.fixture()
def tool(mock_rag_client: MagicMock) -> RagTool:
    with (
        patch("crewai_tools.adapters.crewai_rag_adapter.get_rag_client", return_value=mock_rag_client),
        patch("crewai_tools.adapters.crewai_rag_adapter.create_client", return_value=mock_rag_client),
    ):
        return RagTool()


# ---------------------------------------------------------------------------
# Positional arg validation (existing behaviour, regression guard)
# ---------------------------------------------------------------------------

class TestPositionalArgValidation:
    def test_blocks_traversal_in_positional_arg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe"):
            tool.add("../../etc/passwd")

    def test_blocks_file_url_in_positional_arg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe"):
            tool.add("file:///etc/passwd")


# ---------------------------------------------------------------------------
# Keyword arg validation (the newly fixed gap)
# ---------------------------------------------------------------------------

class TestKwargPathValidation:
    def test_blocks_traversal_via_path_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe path"):
            tool.add(path="../../etc/passwd")

    def test_blocks_traversal_via_file_path_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe file_path"):
            tool.add(file_path="/etc/passwd")

    def test_blocks_traversal_via_directory_path_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe directory_path"):
            tool.add(directory_path="../../sensitive_dir")

    def test_blocks_file_url_via_url_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe url"):
            tool.add(url="file:///etc/passwd")

    def test_blocks_private_ip_via_url_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe url"):
            tool.add(url="http://169.254.169.254/latest/meta-data/")

    def test_blocks_private_ip_via_website_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe website"):
            tool.add(website="http://192.168.1.1/")

    def test_blocks_file_url_via_github_url_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe github_url"):
            tool.add(github_url="file:///etc/passwd")

    def test_blocks_file_url_via_youtube_url_kwarg(self, tool):
        with pytest.raises(ValueError, match="Blocked unsafe youtube_url"):
            tool.add(youtube_url="file:///etc/passwd")

