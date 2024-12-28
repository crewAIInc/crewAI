from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileReadToolSchema(BaseModel):
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
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        file_path = kwargs.get("file_path", self.file_path)
        if file_path is None:
            return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."
        
        try:
            with open(file_path, "r") as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: File not found at path: {file_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {file_path}"
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {str(e)}"
