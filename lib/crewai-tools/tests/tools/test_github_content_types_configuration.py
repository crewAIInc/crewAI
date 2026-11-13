from types import SimpleNamespace
from unittest.mock import Mock, patch

from crewai_tools import GithubSearchTool
from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
import pytest


@pytest.mark.parametrize("override", [None, ["code"]])
def test_add_respects_configured_content_types(
    monkeypatch: pytest.MonkeyPatch, override: list[str] | None
) -> None:
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("140.82.112.3", 443))],
    )
    repository = SimpleNamespace(
        full_name="example/project", description="An offline project",
        language="Python", stargazers_count=1, forks_count=0,
        get_readme=Mock(return_value=SimpleNamespace(decoded_content=b"Project README")),
        get_contents=Mock(return_value=[]), get_pulls=Mock(return_value=[]),
        get_issues=Mock(return_value=[]),
    )
    client = Mock()
    with (
        patch("crewai_tools.adapters.crewai_rag_adapter.get_rag_client", return_value=client),
        patch("crewai_tools.rag.loaders.github_loader.Github") as github,
    ):
        github.return_value.get_repo.return_value = repository
        tool = GithubSearchTool(
            content_types=["repo"], gh_token="synthetic-token",
            adapter=CrewAIRagAdapter(collection_name="test_content_types"),
        )
        tool.add(repo="example/project", content_types=override)

    records = client.add_documents.call_args.kwargs["documents"]
    content = "\n".join(record["content"] for record in records)
    repository.get_pulls.assert_not_called()
    repository.get_issues.assert_not_called()
    if override is None:
        assert "Repository: example/project" in content
        assert "Project README" not in content
        repository.get_readme.assert_not_called()
    else:
        assert "Project README" in content
        assert "Repository: example/project" not in content
        repository.get_readme.assert_called_once_with()
    assert tool.content_types == ["repo"]
