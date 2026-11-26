from types import SimpleNamespace
from unittest.mock import patch

from crewai_tools.rag.loaders.github_loader import GithubLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.mark.parametrize("content_types", [["repo"], ["code"], ["repo", "code"]])
def test_github_loader_preserves_repository_source(content_types: list[str]) -> None:
    url = "https://github.com/example/project"
    repository = SimpleNamespace(
        full_name="example/project",
        description="An offline repository",
        language="Python",
        stargazers_count=1,
        forks_count=0,
        get_readme=lambda: SimpleNamespace(decoded_content=b"Repository README"),
        get_contents=lambda path: [],
    )
    with patch("crewai_tools.rag.loaders.github_loader.Github") as github:
        github.return_value.get_repo.return_value = repository
        result = GithubLoader().load(
            SourceContent(url), metadata={"content_types": content_types}
        )

    assert result.content
    assert result.source == url
    assert result.source == result.metadata["source"]
    assert result.metadata["repo"] == "example/project"
    assert result.doc_id == GithubLoader.generate_doc_id(url, result.content)


@pytest.mark.parametrize("content_types", [["issue"], ["repo", "issue"]])
def test_github_loader_collects_issues_after_pull_requests(
    content_types: list[str],
) -> None:
    url = "https://github.com/example/project"
    pull_requests = [
        SimpleNamespace(pull_request={"url": f"https://example.test/pr/{number}"})
        for number in range(10)
    ]
    issues = [
        SimpleNamespace(
            pull_request=None,
            number=number,
            title=f"Issue {number}",
            body=f"Details for {number}",
        )
        for number in range(100, 106)
    ]
    repository = SimpleNamespace(
        full_name="example/project",
        description="An offline repository",
        language="Python",
        stargazers_count=1,
        forks_count=0,
        get_issues=lambda **_: pull_requests + issues,
    )

    with patch("crewai_tools.rag.loaders.github_loader.Github") as github:
        github.return_value.get_repo.return_value = repository
        result = GithubLoader().load(
            SourceContent(url), metadata={"content_types": content_types}
        )

    for number in range(100, 105):
        assert f"- Issue #{number}: Issue {number}" in result.content
    assert "- Issue #105:" not in result.content
    assert result.content.count("- Issue #") == 5
