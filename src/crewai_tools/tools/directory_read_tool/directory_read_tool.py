import os
from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field
from ..base_tool import BaseTool

class FixedDirectoryReadToolSchema(BaseModel):
	"""Input for DirectoryReadTool."""
	pass

class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
	"""Input for DirectoryReadTool."""
	directory: str = Field(..., description="Mandatory directory to list content")

class DirectoryReadTool(BaseTool):
	name: str = "List files in directory"
	description: str = "A tool that can be used to recursively list a directory's content."
	args_schema: Type[BaseModel] = DirectoryReadToolSchema
	directory: Optional[str] = None

	def __init__(self, directory: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if directory is not None:
			self.directory = directory
			self.description = f"A tool that can be used to list {directory}'s content."
			self.args_schema = FixedDirectoryReadToolSchema

	def _run(
		self,
		**kwargs: Any,
	) -> Any:
		directory = kwargs.get('directory', self.directory)
		return [(os.path.join(root, file).replace(directory, "").lstrip(os.path.sep)) for root, dirs, files in os.walk(directory) for file in files]

