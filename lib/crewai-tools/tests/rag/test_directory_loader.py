import os
import tempfile

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.directory_loader import DirectoryLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.fixture
def temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestDirectoryLoader:
    def _create_file(self, directory, filename, content="test content"):
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_non_recursive(self, temp_directory):
        self._create_file(temp_directory, "file1.txt")
        self._create_file(temp_directory, "file2.txt")
        subdir = os.path.join(temp_directory, "subdir")
        os.makedirs(subdir)
        self._create_file(subdir, "file3.txt")

        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory), recursive=False)

        assert isinstance(result, LoaderResult)
        assert "file1.txt" in result.content
        assert "file2.txt" in result.content
        assert "file3.txt" not in result.content
        assert result.metadata["total_files"] == 2

    def test_load_recursive(self, temp_directory):
        self._create_file(temp_directory, "file1.txt")
        nested = os.path.join(temp_directory, "subdir", "nested")
        os.makedirs(nested)
        self._create_file(os.path.join(temp_directory, "subdir"), "file2.txt")
        self._create_file(nested, "file3.txt")

        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory), recursive=True)

        assert all(f"file{i}.txt" in result.content for i in range(1, 4))

    def test_include_and_exclude_extensions(self, temp_directory):
        self._create_file(temp_directory, "a.txt")
        self._create_file(temp_directory, "b.py")
        self._create_file(temp_directory, "c.md")

        loader = DirectoryLoader()
        result = loader.load(
            SourceContent(temp_directory), include_extensions=[".txt", ".py"]
        )
        assert "a.txt" in result.content
        assert "b.py" in result.content
        assert "c.md" not in result.content

        result2 = loader.load(
            SourceContent(temp_directory), exclude_extensions=[".py", ".md"]
        )
        assert "a.txt" in result2.content
        assert "b.py" not in result2.content
        assert "c.md" not in result2.content

    def test_max_files_limit(self, temp_directory):
        for i in range(5):
            self._create_file(temp_directory, f"file{i}.txt")

        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory), max_files=3)

        assert result.metadata["total_files"] == 3
        assert all(f"file{i}.txt" in result.content for i in range(3))

    def test_hidden_files_and_dirs_excluded(self, temp_directory):
        self._create_file(temp_directory, "visible.txt", "visible")
        self._create_file(temp_directory, ".hidden.txt", "hidden")

        hidden_dir = os.path.join(temp_directory, ".hidden")
        os.makedirs(hidden_dir)
        self._create_file(hidden_dir, "inside_hidden.txt")

        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory), recursive=True)

        assert "visible.txt" in result.content
        assert ".hidden.txt" not in result.content
        assert "inside_hidden.txt" not in result.content

    def test_directory_does_not_exist(self):
        loader = DirectoryLoader()
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            loader.load(SourceContent("/path/does/not/exist"))

    def test_path_is_not_a_directory(self):
        with tempfile.NamedTemporaryFile() as f:
            loader = DirectoryLoader()
            with pytest.raises(ValueError, match="Path is not a directory"):
                loader.load(SourceContent(f.name))

    def test_url_not_supported(self):
        loader = DirectoryLoader()
        with pytest.raises(ValueError, match="URL directory loading is not supported"):
            loader.load(SourceContent("https://example.com"))

    def test_processing_error_handling(self, temp_directory):
        self._create_file(temp_directory, "valid.txt")
        self._create_file(temp_directory, "error.txt")

        loader = DirectoryLoader()
        original_method = loader._process_single_file

        def mock(file_path):
            if "error" in file_path:
                raise ValueError("Mock error")
            return original_method(file_path)

        loader._process_single_file = mock
        result = loader.load(SourceContent(temp_directory))

        assert "valid.txt" in result.content
        assert "error.txt (ERROR)" in result.content
        assert result.metadata["errors"] == 1
        assert len(result.metadata["error_details"]) == 1

    def test_metadata_structure(self, temp_directory):
        self._create_file(temp_directory, "test.txt", "Sample")

        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory))
        metadata = result.metadata

        expected_keys = {
            "format",
            "directory_path",
            "total_files",
            "processed_files",
            "errors",
            "file_details",
            "error_details",
        }

        assert expected_keys.issubset(metadata)
        assert all(
            k in metadata["file_details"][0] for k in ("path", "metadata", "source")
        )

    def test_empty_directory(self, temp_directory):
        loader = DirectoryLoader()
        result = loader.load(SourceContent(temp_directory))

        assert result.content == ""
        assert result.metadata["total_files"] == 0
        assert result.metadata["processed_files"] == 0
