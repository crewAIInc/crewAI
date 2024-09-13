from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from ..base_tool import BaseTool


class FixedFileReadToolSchema(BaseModel):
    """Input for FileReadTool."""

    pass


class FileReadToolSchema(FixedFileReadToolSchema):
    """Input for FileReadTool."""

    file_path: str = Field(..., description="Mandatory file full path to read the file")


class FileReadTool(BaseTool):
    name: str = "Read a file's content"
    description: str = "A tool that can be used to read a file's content."
    args_schema: Type[BaseModel] = FileReadToolSchema
    file_path: Optional[str] = None

    def __init__(self, file_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if file_path is not None:
            self.file_path = file_path
            self.description = f"A tool that can be used to read {file_path}'s content."
            self.args_schema = FixedFileReadToolSchema
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        try:
            file_path = kwargs.get("file_path", self.file_path)
            with open(file_path, "r") as file:
                return file.read()
        except Exception as e:
            return f"Fail to read the file {file_path}. Error: {e}"
