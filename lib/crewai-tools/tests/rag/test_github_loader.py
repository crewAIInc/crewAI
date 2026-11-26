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


def test_github_loader_skips_pull_requests_when_collecting_issues() -> None:
    """Open issues after a PR-filled first page must still be loaded.

    GitHub returns pull requests from get_issues(state="open"). Truncating that
    mixed list before filtering PRs can drop every real issue.
    """
    url = "https://github.com/example/project"
    pull_request = SimpleNamespace(pull_request={"url": "https://example.test/pr"})
    issue = SimpleNamespace(
        pull_request=None, number=42, title="Existing issue", body="Still open"
    )
    repository = SimpleNamespace(
        get_issues=lambda **_: [pull_request] * 10 + [issue]
    )

    with patch("crewai_tools.rag.loaders.github_loader.Github") as github:
        github.return_value.get_repo.return_value = repository
        result = GithubLoader().load(
            SourceContent(url),
            metadata={"content_types": ["issue"]},
        )

    assert "- Issue #42: Existing issue" in result.content
    assert "Still open" in result.content
    assert result.source == url


def test_github_loader_collects_up_to_five_real_issues_past_prs() -> None:
    """Keep scanning past PR entries until five issues are found."""
    url = "https://github.com/example/project"
    pull_request = SimpleNamespace(pull_request={"url": "https://example.test/pr"})
    issues = [
        SimpleNamespace(
            pull_request=None,
            number=i,
            title=f"Issue {i}",
            body=f"Body {i}",
        )
        for i in range(1, 7)
    ]
    # Three PRs then six issues — loader should return issues 1..5 only.
    repository = SimpleNamespace(
        get_issues=lambda **_: [pull_request] * 3 + issues
    )

    with patch("crewai_tools.rag.loaders.github_loader.Github") as github:
        github.return_value.get_repo.return_value = repository
        result = GithubLoader().load(
            SourceContent(url),
            metadata={"content_types": ["issue"]},
        )

    for i in range(1, 6):
        assert f"- Issue #{i}: Issue {i}" in result.content
    assert "- Issue #6: Issue 6" not in result.content

