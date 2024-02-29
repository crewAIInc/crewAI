from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.loaders.directory_loader import DirectoryLoader

from ..rag.rag_tool import RagTool


class FixedDirectorySearchToolSchema(BaseModel):
	"""Input for DirectorySearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the directory's content")

class DirectorySearchToolSchema(FixedDirectorySearchToolSchema):
	"""Input for DirectorySearchTool."""
	directory: str = Field(..., description="Mandatory directory you want to search")

class DirectorySearchTool(RagTool):
	name: str = "Search a directory's content"
	description: str = "A tool that can be used to semantic search a query from a directory's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = DirectorySearchToolSchema
	directory: Optional[str] = None

	def __init__(self, directory: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if directory is not None:
			self.directory = directory
			self.description = f"A tool that can be used to semantic search a query the {directory} directory's content."
			self.args_schema = FixedDirectorySearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		directory = kwargs.get('directory', self.directory)
		loader = DirectoryLoader(config=dict(recursive=True))
		self.app = App()
		self.app.add(directory, loader=loader)
		return super()._run(query=search_query)