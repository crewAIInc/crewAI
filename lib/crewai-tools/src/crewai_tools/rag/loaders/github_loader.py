"""GitHub repository content loader."""

from github import Github, GithubException

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class GithubLoader(BaseLoader):
    """Loader for GitHub repository content."""

    def load(self, source: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load content from a GitHub repository.

        Args:
            source: GitHub repository URL
            **kwargs: Additional arguments including gh_token and content_types

        Returns:
            LoaderResult with repository content
        """
        metadata = kwargs.get("metadata", {})
        gh_token = metadata.get("gh_token")
        content_types = metadata.get("content_types", ["code", "repo"])

        repo_url = source.source
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"Invalid GitHub URL: {repo_url}")

        parts = repo_url.replace("https://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        repo_name = f"{parts[0]}/{parts[1]}"

        g = Github(gh_token) if gh_token else Github()

        try:
            repo = g.get_repo(repo_name)
        except GithubException as e:
            raise ValueError(f"Unable to access repository {repo_name}: {e}") from e

        all_content = []

        if "repo" in content_types:
            all_content.append(f"Repository: {repo.full_name}")
            all_content.append(f"Description: {repo.description or 'No description'}")
            all_content.append(f"Language: {repo.language or 'Not specified'}")
            all_content.append(f"Stars: {repo.stargazers_count}")
            all_content.append(f"Forks: {repo.forks_count}")
            all_content.append("")

        if "code" in content_types:
            try:
                readme = repo.get_readme()
                all_content.append("README:")
                all_content.append(readme.decoded_content.decode(errors="ignore"))
                all_content.append("")
            except GithubException:
                pass

            try:
                contents = repo.get_contents("")
                if isinstance(contents, list):
                    all_content.append("Repository structure:")
                    for content_file in contents[:20]:
                        all_content.append(  # noqa: PERF401
                            f"- {content_file.path} ({content_file.type})"
                        )
                    all_content.append("")
            except GithubException:
                pass

        if "pr" in content_types:
            prs = repo.get_pulls(state="open")
            pr_list = list(prs[:5])
            if pr_list:
                all_content.append("Recent Pull Requests:")
                for pr in pr_list:
                    all_content.append(f"- PR #{pr.number}: {pr.title}")
                    if pr.body:
                        body_preview = pr.body[:200].replace("\n", " ")
                        all_content.append(f"  {body_preview}")
                all_content.append("")

        if "issue" in content_types:
            issues = repo.get_issues(state="open")
            issue_list = [i for i in list(issues[:10]) if not i.pull_request][:5]
            if issue_list:
                all_content.append("Recent Issues:")
                for issue in issue_list:
                    all_content.append(f"- Issue #{issue.number}: {issue.title}")
                    if issue.body:
                        body_preview = issue.body[:200].replace("\n", " ")
                        all_content.append(f"  {body_preview}")
                all_content.append("")

        if not all_content:
            raise ValueError(f"No content could be loaded from repository: {repo_url}")

        content = "\n".join(all_content)
        return LoaderResult(
            content=content,
            metadata={
                "source": repo_url,
                "repo": repo_name,
                "content_types": content_types,
            },
            doc_id=self.generate_doc_id(source_ref=repo_url, content=content),
        )
