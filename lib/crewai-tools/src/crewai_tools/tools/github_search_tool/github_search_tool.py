from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedGithubSearchToolSchema(BaseModel):
    """Input for GithubSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the github repo's content",
    )


class GithubSearchToolSchema(FixedGithubSearchToolSchema):
    """Input for GithubSearchTool."""

    github_repo: str = Field(..., description="Mandatory github you want to search")
    content_types: list[str] = Field(
        ...,
        description="Mandatory content types you want to be included search, options: [code, repo, pr, issue]",
    )


class GithubSearchTool(RagTool):
    name: str = "Search a github repo's content"
    description: str = "A tool that can be used to semantic search a query from a github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
    summarize: bool = False
    gh_token: str
    args_schema: type[BaseModel] = GithubSearchToolSchema
    content_types: list[str] = Field(
        default_factory=lambda: ["code", "repo", "pr", "issue"],
        description="Content types you want to be included search, options: [code, repo, pr, issue]",
    )

    def __init__(
        self,
        github_repo: str | None = None,
        content_types: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if github_repo and content_types:
            self.add(repo=github_repo, content_types=content_types)
            self.description = f"A tool that can be used to semantic search a query the {github_repo} github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
            self.args_schema = FixedGithubSearchToolSchema
            self._generate_description()

    def add(
        self,
        repo: str,
        content_types: list[str] | None = None,
    ) -> None:
        content_types = content_types or self.content_types
        super().add(
            f"https://github.com/{repo}",
            data_type=DataType.GITHUB,
            metadata={"content_types": content_types, "gh_token": self.gh_token},
        )

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        github_repo: str | None = None,
        content_types: list[str] | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if github_repo:
            self.add(
                repo=github_repo,
                content_types=content_types,
            )
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
