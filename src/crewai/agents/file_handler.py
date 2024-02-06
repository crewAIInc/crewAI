from typing import List, Dict
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
from pydantic_core import PydanticCustomError


class FileHandler:
    def __init__(self):
        self.file_paths = []
        self._file_content_cache = {}
        self.working_directory = TemporaryDirectory()
        self.toolkit = FileManagementToolkit(root_dir=str(self.working_directory.name))
        self.read_tool = next(tool for tool in self.toolkit.get_tools() if tool.name == 'read_file')

    def add_file_paths(self, paths: List[str]) -> None:
        if not all(isinstance(path, str) for path in paths):
            raise PydanticCustomError("invalid_file_paths", "All paths must be strings.", {})
        self.file_paths.extend(paths)

    def load_file_content(self) -> Dict[str, str]:
        for path in self.file_paths:
            if path not in self._file_content_cache:
                try:
                    content = self.read_tool.run({"file_path": path})
                    self._file_content_cache[path] = content
                except Exception as e:
                    raise PydanticCustomError("file_read_error", f"Error reading file {path}: {e}", {})
        return self._file_content_cache
