from typing import Optional, Type, List, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.loaders.github import GithubLoader

from ..rag.rag_tool import RagTool


class FixedGithubSearchToolSchema(BaseModel):
	"""Input for GithubSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the github repo's content")

class GithubSearchToolSchema(FixedGithubSearchToolSchema):
	"""Input for GithubSearchTool."""
	github_repo: str = Field(..., description="Mandatory github you want to search")
	content_types: List[str] = Field(..., description="Mandatory content types you want to be inlcuded search, options: [code, repo, pr, issue]")

class GithubSearchTool(RagTool):
	name: str = "Search a github repo's content"
	description: str = "A tool that can be used to semantic search a query from a github repo's content."
	summarize: bool = False
	gh_token: str = None
	args_schema: Type[BaseModel] = GithubSearchToolSchema
	github_repo: Optional[str] = None
	content_types: List[str]

	def __init__(self, github_repo: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if github_repo is not None:
			self.github_repo = github_repo
			self.description = f"A tool that can be used to semantic search a query the {github_repo} github repo's content."
			self.args_schema = FixedGithubSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		github_repo = kwargs.get('github_repo', self.github_repo)
		loader = GithubLoader(config={"token": self.gh_token})
		app = App()
		app.add(f"repo:{github_repo} type:{','.join(self.content_types)}", data_type="github", loader=loader)
		self.app = app
		return super()._run(query=search_query)