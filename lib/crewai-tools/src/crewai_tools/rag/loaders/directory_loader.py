import os
from pathlib import Path

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class DirectoryLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load and process all files from a directory recursively.

        Args:
            source_content: Directory path or URL to a directory listing
            **kwargs: Additional options:
                - recursive: bool (default True) - Whether to search recursively
                - include_extensions: list - Only include files with these extensions
                - exclude_extensions: list - Exclude files with these extensions
                - max_files: int - Maximum number of files to process
        """
        source_ref = source_content.source_ref

        if source_content.is_url():
            raise ValueError(
                "URL directory loading is not supported. Please provide a local directory path."
            )

        if not os.path.exists(source_ref):
            raise FileNotFoundError(f"Directory does not exist: {source_ref}")

        if not os.path.isdir(source_ref):
            raise ValueError(f"Path is not a directory: {source_ref}")

        return self._process_directory(source_ref, kwargs)

    def _process_directory(self, dir_path: str, kwargs: dict) -> LoaderResult:
        recursive: bool = kwargs.get("recursive", True)
        include_extensions: list[str] | None = kwargs.get("include_extensions", None)
        exclude_extensions: list[str] | None = kwargs.get("exclude_extensions", None)
        max_files: int | None = kwargs.get("max_files", None)

        files = self._find_files(
            dir_path, recursive, include_extensions, exclude_extensions
        )

        if max_files is not None and len(files) > max_files:
            files = files[:max_files]

        all_contents = []
        processed_files = []
        errors = []

        for file_path in files:
            try:
                result = self._process_single_file(file_path)
                if result:
                    all_contents.append(f"=== File: {file_path} ===\n{result.content}")
                    processed_files.append(
                        {
                            "path": file_path,
                            "metadata": result.metadata,
                            "source": result.source,
                        }
                    )
            except Exception as e:  # noqa: PERF203
                error_msg = f"Error processing {file_path}: {e!s}"
                errors.append(error_msg)
                all_contents.append(f"=== File: {file_path} (ERROR) ===\n{error_msg}")

        combined_content = "\n\n".join(all_contents)

        metadata = {
            "format": "directory",
            "directory_path": dir_path,
            "total_files": len(files),
            "processed_files": len(processed_files),
            "errors": len(errors),
            "file_details": processed_files,
            "error_details": errors,
        }

        return LoaderResult(
            content=combined_content,
            source=dir_path,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=dir_path, content=combined_content),
        )

    def _find_files(
        self,
        dir_path: str,
        recursive: bool,
        include_ext: list[str] | None = None,
        exclude_ext: list[str] | None = None,
    ) -> list[str]:
        """Find all files in directory matching criteria."""
        files = []

        if recursive:
            for root, dirs, filenames in os.walk(dir_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for filename in filenames:
                    if self._should_include_file(filename, include_ext, exclude_ext):
                        files.append(os.path.join(root, filename))  # noqa: PERF401
        else:
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path) and self._should_include_file(
                        item, include_ext, exclude_ext
                    ):
                        files.append(item_path)
            except PermissionError:
                pass

        return sorted(files)

    @staticmethod
    def _should_include_file(
        filename: str,
        include_ext: list[str] | None = None,
        exclude_ext: list[str] | None = None,
    ) -> bool:
        """Determine if a file should be included based on criteria."""
        if filename.startswith("."):
            return False

        _, ext = os.path.splitext(filename.lower())

        if include_ext:
            if ext not in [
                e.lower() if e.startswith(".") else f".{e.lower()}" for e in include_ext
            ]:
                return False

        if exclude_ext:
            if ext in [
                e.lower() if e.startswith(".") else f".{e.lower()}" for e in exclude_ext
            ]:
                return False

        return True

    @staticmethod
    def _process_single_file(file_path: str) -> LoaderResult:
        from crewai_tools.rag.data_types import DataTypes

        data_type = DataTypes.from_content(Path(file_path))

        loader = data_type.get_loader()

        result = loader.load(SourceContent(file_path))

        if result.metadata is None:
            result.metadata = {}

        result.metadata.update(
            {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "data_type": str(data_type),
                "loader_type": loader.__class__.__name__,
            }
        )

        return result
