from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from crewai_tools.tools.github_search_tool.github_search_tool import GithubSearchTool
from crewai_tools.tools.rag.rag_tool import RagTool


@contextmanager
def _without_vector_store() -> Iterator[None]:
    """Skip RagTool's adapter construction: these tests only cover field plumbing."""
    with patch("crewai_tools.adapters.crewai_rag_adapter.CrewAIRagAdapter"):
        yield


def test_constructor_content_types_are_retained() -> None:
    """content_types passed to the constructor must reach the model field."""
    with _without_vector_store():
        tool = GithubSearchTool(gh_token="token", content_types=["repo"])

    assert tool.content_types == ["repo"]


def test_content_types_are_used_by_later_additions() -> None:
    """A later add() must reuse the constructor's content_types, not the default."""
    with _without_vector_store():
        tool = GithubSearchTool(gh_token="token", content_types=["repo"])

    with patch.object(RagTool, "add") as mock_add:
        tool.add(repo="other/project")

    assert mock_add.call_args is not None
    assert mock_add.call_args.kwargs["metadata"]["content_types"] == ["repo"]
