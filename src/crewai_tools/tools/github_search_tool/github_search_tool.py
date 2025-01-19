from typing import Any, List, Optional, Type

from embedchain.loaders.github import GithubLoader
from pydantic import BaseModel, Field

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
    description: str = "A tool that can be used to semantic search a query from a github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
    summarize: bool = False
    gh_token: str
    args_schema: Type[BaseModel] = GithubSearchToolSchema
    content_types: List[str]

    def __init__(self, github_repo: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if github_repo is not None:
            kwargs["data_type"] = "github"
            kwargs["loader"] = GithubLoader(config={"token": self.gh_token})

            self.add(repo=github_repo)
            self.description = f"A tool that can be used to semantic search a query the {github_repo} github repo's content. This is not the GitHub API, but instead a tool that can provide semantic search capabilities."
            self.args_schema = FixedGithubSearchToolSchema
            self._generate_description()

    def add(
        self,
        repo: str,
        content_types: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        content_types = content_types or self.content_types

        super().add(f"repo:{repo} type:{','.join(content_types)}", **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "github_repo" in kwargs:
            self.add(
                repo=kwargs["github_repo"], content_types=kwargs.get("content_types")
            )

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
