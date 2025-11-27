"""Tests for RagTool.add() method with various data_type values."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


@pytest.fixture
def mock_rag_client() -> MagicMock:
    """Create a mock RAG client for testing."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[])
    return mock_client


@pytest.fixture
def rag_tool(mock_rag_client: MagicMock) -> RagTool:
    """Create a RagTool instance with mocked client."""
    with (
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.get_rag_client",
            return_value=mock_rag_client,
        ),
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.create_client",
            return_value=mock_rag_client,
        ),
    ):
        return RagTool()


class TestDataTypeFileAlias:
    """Tests for data_type='file' alias."""

    def test_file_alias_with_existing_file(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test that data_type='file' works with existing files."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content for file alias.")

            rag_tool.add(path=str(test_file), data_type="file")

            assert mock_rag_client.add_documents.called

    def test_file_alias_with_nonexistent_file_raises_error(
        self, rag_tool: RagTool
    ) -> None:
        """Test that data_type='file' raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent/path/to/file.pdf", data_type="file")

    def test_file_alias_with_path_keyword(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test that path keyword argument works with data_type='file'."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "document.txt"
            test_file.write_text("Content via path keyword.")

            rag_tool.add(data_type="file", path=str(test_file))

            assert mock_rag_client.add_documents.called

    def test_file_alias_with_file_path_keyword(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test that file_path keyword argument works with data_type='file'."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "document.txt"
            test_file.write_text("Content via file_path keyword.")

            rag_tool.add(data_type="file", file_path=str(test_file))

            assert mock_rag_client.add_documents.called


class TestDataTypeStringValues:
    """Tests for data_type as string values matching DataType enum."""

    def test_pdf_file_string(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type='pdf_file' with existing PDF file."""
        with TemporaryDirectory() as tmpdir:
            # Create a minimal valid PDF file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n"
                b"<<\n/Root 1 0 R\n>>\n%%EOF"
            )

            # Mock the PDF loader to avoid actual PDF parsing
            with patch(
                "crewai_tools.adapters.crewai_rag_adapter.DataType.get_loader"
            ) as mock_loader:
                mock_loader_instance = MagicMock()
                mock_loader_instance.load.return_value = MagicMock(
                    content="PDF content", metadata={}, doc_id="test-id"
                )
                mock_loader.return_value = mock_loader_instance

                rag_tool.add(path=str(test_file), data_type="pdf_file")

                assert mock_rag_client.add_documents.called

    def test_text_file_string(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type='text_file' with existing text file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Plain text content.")

            rag_tool.add(path=str(test_file), data_type="text_file")

            assert mock_rag_client.add_documents.called

    def test_csv_string(self, rag_tool: RagTool, mock_rag_client: MagicMock) -> None:
        """Test data_type='csv' with existing CSV file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.csv"
            test_file.write_text("name,value\nfoo,1\nbar,2")

            rag_tool.add(path=str(test_file), data_type="csv")

            assert mock_rag_client.add_documents.called

    def test_json_string(self, rag_tool: RagTool, mock_rag_client: MagicMock) -> None:
        """Test data_type='json' with existing JSON file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text('{"key": "value", "items": [1, 2, 3]}')

            rag_tool.add(path=str(test_file), data_type="json")

            assert mock_rag_client.add_documents.called

    def test_xml_string(self, rag_tool: RagTool, mock_rag_client: MagicMock) -> None:
        """Test data_type='xml' with existing XML file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.xml"
            test_file.write_text('<?xml version="1.0"?><root><item>value</item></root>')

            rag_tool.add(path=str(test_file), data_type="xml")

            assert mock_rag_client.add_documents.called

    def test_mdx_string(self, rag_tool: RagTool, mock_rag_client: MagicMock) -> None:
        """Test data_type='mdx' with existing MDX file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mdx"
            test_file.write_text("# Heading\n\nSome markdown content.")

            rag_tool.add(path=str(test_file), data_type="mdx")

            assert mock_rag_client.add_documents.called

    def test_text_string(self, rag_tool: RagTool, mock_rag_client: MagicMock) -> None:
        """Test data_type='text' with raw text content."""
        rag_tool.add("This is raw text content.", data_type="text")

        assert mock_rag_client.add_documents.called

    def test_directory_string(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type='directory' with existing directory."""
        with TemporaryDirectory() as tmpdir:
            # Create some files in the directory
            (Path(tmpdir) / "file1.txt").write_text("Content 1")
            (Path(tmpdir) / "file2.txt").write_text("Content 2")

            rag_tool.add(path=tmpdir, data_type="directory")

            assert mock_rag_client.add_documents.called


class TestDataTypeEnumValues:
    """Tests for data_type as DataType enum values."""

    def test_datatype_file_enum_with_existing_file(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type=DataType.FILE with existing file (auto-detect)."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("File enum auto-detect content.")

            rag_tool.add(str(test_file), data_type=DataType.FILE)

            assert mock_rag_client.add_documents.called

    def test_datatype_file_enum_with_nonexistent_file_raises_error(
        self, rag_tool: RagTool
    ) -> None:
        """Test data_type=DataType.FILE raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add("nonexistent/file.pdf", data_type=DataType.FILE)

    def test_datatype_pdf_file_enum(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type=DataType.PDF_FILE with existing file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_bytes(
                b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n"
                b"<<\n/Root 1 0 R\n>>\n%%EOF"
            )

            with patch(
                "crewai_tools.adapters.crewai_rag_adapter.DataType.get_loader"
            ) as mock_loader:
                mock_loader_instance = MagicMock()
                mock_loader_instance.load.return_value = MagicMock(
                    content="PDF content", metadata={}, doc_id="test-id"
                )
                mock_loader.return_value = mock_loader_instance

                rag_tool.add(str(test_file), data_type=DataType.PDF_FILE)

                assert mock_rag_client.add_documents.called

    def test_datatype_text_file_enum(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type=DataType.TEXT_FILE with existing file."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Text file content.")

            rag_tool.add(str(test_file), data_type=DataType.TEXT_FILE)

            assert mock_rag_client.add_documents.called

    def test_datatype_text_enum(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type=DataType.TEXT with raw text."""
        rag_tool.add("Raw text using enum.", data_type=DataType.TEXT)

        assert mock_rag_client.add_documents.called

    def test_datatype_directory_enum(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test data_type=DataType.DIRECTORY with existing directory."""
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_text("Directory file content.")

            rag_tool.add(tmpdir, data_type=DataType.DIRECTORY)

            assert mock_rag_client.add_documents.called


class TestInvalidDataType:
    """Tests for invalid data_type values."""

    def test_invalid_string_data_type_raises_error(self, rag_tool: RagTool) -> None:
        """Test that invalid string data_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid data_type"):
            rag_tool.add("some content", data_type="invalid_type")

    def test_invalid_data_type_error_message_contains_valid_values(
        self, rag_tool: RagTool
    ) -> None:
        """Test that error message lists valid data_type values."""
        with pytest.raises(ValueError) as exc_info:
            rag_tool.add("some content", data_type="not_a_type")

        error_message = str(exc_info.value)
        assert "file" in error_message
        assert "pdf_file" in error_message
        assert "text_file" in error_message


class TestFileExistenceValidation:
    """Tests for file existence validation."""

    def test_pdf_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent PDF file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.pdf", data_type="pdf_file")

    def test_text_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent text file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.txt", data_type="text_file")

    def test_csv_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.csv", data_type="csv")

    def test_json_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.json", data_type="json")

    def test_xml_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent XML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.xml", data_type="xml")

    def test_docx_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent DOCX file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.docx", data_type="docx")

    def test_mdx_file_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent MDX file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(path="nonexistent.mdx", data_type="mdx")

    def test_directory_not_found_raises_error(self, rag_tool: RagTool) -> None:
        """Test that non-existent directory raises ValueError."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            rag_tool.add(path="nonexistent/directory", data_type="directory")


class TestKeywordArgumentVariants:
    """Tests for different keyword argument combinations."""

    def test_positional_argument_with_data_type(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test positional argument with data_type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Positional arg content.")

            rag_tool.add(str(test_file), data_type="text_file")

            assert mock_rag_client.add_documents.called

    def test_path_keyword_with_data_type(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test path keyword argument with data_type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Path keyword content.")

            rag_tool.add(path=str(test_file), data_type="text_file")

            assert mock_rag_client.add_documents.called

    def test_file_path_keyword_with_data_type(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test file_path keyword argument with data_type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("File path keyword content.")

            rag_tool.add(file_path=str(test_file), data_type="text_file")

            assert mock_rag_client.add_documents.called

    def test_directory_path_keyword(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test directory_path keyword argument."""
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_text("Directory content.")

            rag_tool.add(directory_path=tmpdir)

            assert mock_rag_client.add_documents.called


class TestAutoDetection:
    """Tests for auto-detection of data type from content."""

    def test_auto_detect_nonexistent_file_raises_error(self, rag_tool: RagTool) -> None:
        """Test that auto-detection raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add("path/to/document.pdf")

    def test_auto_detect_txt_file(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test auto-detection of .txt file type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "auto.txt"
            test_file.write_text("Auto-detected text file.")

            # No data_type specified - should auto-detect
            rag_tool.add(str(test_file))

            assert mock_rag_client.add_documents.called

    def test_auto_detect_csv_file(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test auto-detection of .csv file type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "auto.csv"
            test_file.write_text("col1,col2\nval1,val2")

            rag_tool.add(str(test_file))

            assert mock_rag_client.add_documents.called

    def test_auto_detect_json_file(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test auto-detection of .json file type."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "auto.json"
            test_file.write_text('{"auto": "detected"}')

            rag_tool.add(str(test_file))

            assert mock_rag_client.add_documents.called

    def test_auto_detect_directory(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test auto-detection of directory type."""
        with TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.txt").write_text("Auto-detected directory.")

            rag_tool.add(tmpdir)

            assert mock_rag_client.add_documents.called

    def test_auto_detect_raw_text(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test auto-detection of raw text (non-file content)."""
        rag_tool.add("Just some raw text content")

        assert mock_rag_client.add_documents.called


class TestMetadataHandling:
    """Tests for metadata handling with data_type."""

    def test_metadata_passed_to_documents(
        self, rag_tool: RagTool, mock_rag_client: MagicMock
    ) -> None:
        """Test that metadata is properly passed to documents."""
        with TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Content with metadata.")

            rag_tool.add(
                path=str(test_file),
                data_type="text_file",
                metadata={"custom_key": "custom_value"},
            )

            assert mock_rag_client.add_documents.called
            call_args = mock_rag_client.add_documents.call_args
            documents = call_args.kwargs.get("documents", call_args.args[0] if call_args.args else [])

            # Check that at least one document has the custom metadata
            assert any(
                doc.get("metadata", {}).get("custom_key") == "custom_value"
                for doc in documents
            )