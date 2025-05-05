from typing import List, Optional, Type

from embedchain.loaders.github import GithubLoader
from pydantic import BaseModel, Field, PrivateAttr

from ..rag.rag_tool import RagTool


class FixedGithubSearchToolSchema(BaseModel):
    """Input for GithubSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the github repo's content",
    )


class GithubSearchToolSchema(FixedGithubSearchToolSchema):
    """Input for GithubSearchTool."""

    github_repo: str = Field(..., description="Mandatory github you want to search")
    content_types: List[str] = Field(
        ...,
        description="Mandatory content types you want to be included search, options: [code, repo, pr, issue]",
    )


class GithubSearchTool(RagTool):
    name: str = "Search a github repo's content"
    description: str = (
        "A tool that can be used to semantic search a query from a github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
    )
    summarize: bool = False
    gh_token: str
    args_schema: Type[BaseModel] = GithubSearchToolSchema
    content_types: List[str] = Field(
        default_factory=lambda: ["code", "repo", "pr", "issue"],
        description="Content types you want to be included search, options: [code, repo, pr, issue]",
    )
    _loader: GithubLoader | None = PrivateAttr(default=None)

    def __init__(
        self,
        github_repo: Optional[str] = None,
        content_types: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._loader = GithubLoader(config={"token": self.gh_token})

        if github_repo and content_types:
            self.add(repo=github_repo, content_types=content_types)
            self.description = f"A tool that can be used to semantic search a query the {github_repo} github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
            self.args_schema = FixedGithubSearchToolSchema
            self._generate_description()

    def add(
        self,
        repo: str,
        content_types: Optional[List[str]] = None,
    ) -> None:
        content_types = content_types or self.content_types

        super().add(
            f"repo:{repo} type:{','.join(content_types)}",
            data_type="github",
            loader=self._loader,
        )

    def _run(
        self,
        search_query: str,
        github_repo: Optional[str] = None,
        content_types: Optional[List[str]] = None,
    ) -> str:
        if github_repo:
            self.add(
                repo=github_repo,
                content_types=content_types,
            )
        return super()._run(query=search_query)
