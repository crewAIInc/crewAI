"""Test Knowledge Source metadata functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import _coerce_to_records


class TestCoerceToRecords:
    """Test the _coerce_to_records function."""

    def test_coerce_string_list(self):
        """Test coercing a list of strings."""
        documents = ["chunk1", "chunk2", "chunk3"]
        result = _coerce_to_records(documents)
        
        assert len(result) == 3
        assert result[0]["content"] == "chunk1"
        assert result[1]["content"] == "chunk2"
        assert result[2]["content"] == "chunk3"
        assert "metadata" not in result[0]

    def test_coerce_dict_with_metadata(self):
        """Test coercing dictionaries with metadata."""
        documents = [
            {
                "content": "chunk1",
                "metadata": {
                    "filepath": "/path/to/file.txt",
                    "chunk_index": 0,
                    "source_type": "text_file",
                }
            },
            {
                "content": "chunk2",
                "metadata": {
                    "filepath": "/path/to/file.txt",
                    "chunk_index": 1,
                    "source_type": "text_file",
                }
            }
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 2
        assert result[0]["content"] == "chunk1"
        assert result[0]["metadata"]["filepath"] == "/path/to/file.txt"
        assert result[0]["metadata"]["chunk_index"] == 0
        assert result[0]["metadata"]["source_type"] == "text_file"
        assert result[1]["content"] == "chunk2"
        assert result[1]["metadata"]["chunk_index"] == 1

    def test_coerce_mixed_formats(self):
        """Test coercing mixed string and dict formats."""
        documents = [
            "plain string chunk",
            {
                "content": "dict chunk",
                "metadata": {"source_type": "test"}
            }
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 2
        assert result[0]["content"] == "plain string chunk"
        assert "metadata" not in result[0]
        assert result[1]["content"] == "dict chunk"
        assert result[1]["metadata"]["source_type"] == "test"

    def test_coerce_empty_content_skipped(self):
        """Test that empty content is skipped."""
        documents = [
            {"content": "valid chunk"},
            {"content": None},
            {"content": ""},
            {"content": "another valid chunk"}
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 2
        assert result[0]["content"] == "valid chunk"
        assert result[1]["content"] == "another valid chunk"

    def test_coerce_missing_content_skipped(self):
        """Test that dicts without content key are skipped."""
        documents = [
            {"content": "valid chunk"},
            {"metadata": {"some": "data"}},
            {"content": "another valid chunk"}
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 2
        assert result[0]["content"] == "valid chunk"
        assert result[1]["content"] == "another valid chunk"

    def test_coerce_with_doc_id(self):
        """Test coercing documents with doc_id."""
        documents = [
            {
                "content": "chunk with id",
                "doc_id": "doc123",
                "metadata": {"source_type": "test"}
            }
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 1
        assert result[0]["content"] == "chunk with id"
        assert result[0]["doc_id"] == "doc123"
        assert result[0]["metadata"]["source_type"] == "test"

    def test_coerce_metadata_type_conversion(self):
        """Test that metadata values are properly converted to allowed types."""
        documents = [
            {
                "content": "test chunk",
                "metadata": {
                    "string_val": "text",
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True,
                    "none_val": None,
                    "other_val": {"nested": "dict"}
                }
            }
        ]
        result = _coerce_to_records(documents)
        
        assert len(result) == 1
        metadata = result[0]["metadata"]
        assert metadata["string_val"] == "text"
        assert metadata["int_val"] == 42
        assert metadata["float_val"] == 3.14
        assert metadata["bool_val"] is True
        assert metadata["none_val"] == ""
        assert isinstance(metadata["other_val"], str)


class TestTextFileKnowledgeSourceMetadata:
    """Test TextFileKnowledgeSource metadata functionality."""

    def test_text_file_chunks_have_metadata(self, tmpdir):
        """Test that text file chunks include metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        content = "This is a test file. " * 100
        file_path = Path(tmpdir.join("test.txt"))
        with open(file_path, "w") as f:
            f.write(content)

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = TextFileKnowledgeSource(
                file_paths=[file_path],
                storage=KnowledgeStorage(),
                chunk_size=100,
                chunk_overlap=10
            )
            source.add()

            assert len(source.chunks) > 0
            
            for i, chunk in enumerate(source.chunks):
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert chunk["metadata"]["filepath"] == str(file_path)
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "text_file"

            mock_save.assert_called_once()
            saved_chunks = mock_save.call_args[0][0]
            assert len(saved_chunks) == len(source.chunks)


class TestPDFKnowledgeSourceMetadata:
    """Test PDFKnowledgeSource metadata functionality."""

    @patch('crewai.knowledge.source.pdf_knowledge_source.PDFKnowledgeSource._import_pdfplumber')
    def test_pdf_chunks_have_metadata(self, mock_import, tmpdir):
        """Test that PDF chunks include metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content. " * 50
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        mock_import.return_value = mock_pdfplumber

        file_path = Path(tmpdir.join("test.pdf"))
        file_path.touch()

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = PDFKnowledgeSource(
                file_paths=[file_path],
                storage=KnowledgeStorage(),
                chunk_size=100,
                chunk_overlap=10
            )
            source.add()

            assert len(source.chunks) > 0
            
            for i, chunk in enumerate(source.chunks):
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert chunk["metadata"]["filepath"] == str(file_path)
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "pdf"


class TestCSVKnowledgeSourceMetadata:
    """Test CSVKnowledgeSource metadata functionality."""

    def test_csv_chunks_have_metadata(self, tmpdir):
        """Test that CSV chunks include metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        csv_content = "Name,Age,City\nJohn,30,NYC\nJane,25,LA\n" * 20
        file_path = Path(tmpdir.join("test.csv"))
        with open(file_path, "w") as f:
            f.write(csv_content)

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = CSVKnowledgeSource(
                file_paths=[file_path],
                storage=KnowledgeStorage(),
                chunk_size=100,
                chunk_overlap=10
            )
            source.add()

            assert len(source.chunks) > 0
            
            for i, chunk in enumerate(source.chunks):
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert chunk["metadata"]["filepath"] == str(file_path)
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "csv"


class TestJSONKnowledgeSourceMetadata:
    """Test JSONKnowledgeSource metadata functionality."""

    def test_json_chunks_have_metadata(self, tmpdir):
        """Test that JSON chunks include metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        json_content = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
        file_path = Path(tmpdir.join("test.json"))
        with open(file_path, "w") as f:
            f.write(json_content)

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = JSONKnowledgeSource(
                file_paths=[file_path],
                storage=KnowledgeStorage(),
                chunk_size=50,
                chunk_overlap=5
            )
            source.add()

            assert len(source.chunks) > 0
            
            for i, chunk in enumerate(source.chunks):
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert chunk["metadata"]["filepath"] == str(file_path)
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "json"


class TestStringKnowledgeSourceMetadata:
    """Test StringKnowledgeSource metadata functionality."""

    def test_string_chunks_have_metadata(self):
        """Test that string chunks include metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        content = "This is a test string. " * 50

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = StringKnowledgeSource(
                content=content,
                storage=KnowledgeStorage(),
                chunk_size=100,
                chunk_overlap=10
            )
            source.add()

            assert len(source.chunks) > 0
            
            for i, chunk in enumerate(source.chunks):
                assert isinstance(chunk, dict)
                assert "content" in chunk
                assert "metadata" in chunk
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "string"
                assert "filepath" not in chunk["metadata"]


class TestMultipleFilesMetadata:
    """Test metadata for multiple files."""

    def test_multiple_text_files_have_distinct_metadata(self, tmpdir):
        """Test that multiple files have distinct metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        file1 = Path(tmpdir.join("file1.txt"))
        file2 = Path(tmpdir.join("file2.txt"))
        
        with open(file1, "w") as f:
            f.write("Content from file 1. " * 50)
        with open(file2, "w") as f:
            f.write("Content from file 2. " * 50)

        with patch.object(KnowledgeStorage, 'save') as mock_save:
            source = TextFileKnowledgeSource(
                file_paths=[file1, file2],
                storage=KnowledgeStorage(),
                chunk_size=100,
                chunk_overlap=10
            )
            source.add()

            file1_chunks = [c for c in source.chunks if c["metadata"]["filepath"] == str(file1)]
            file2_chunks = [c for c in source.chunks if c["metadata"]["filepath"] == str(file2)]

            assert len(file1_chunks) > 0
            assert len(file2_chunks) > 0
            
            for i, chunk in enumerate(file1_chunks):
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "text_file"
            
            for i, chunk in enumerate(file2_chunks):
                assert chunk["metadata"]["chunk_index"] == i
                assert chunk["metadata"]["source_type"] == "text_file"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_storage_accepts_string_list(self):
        """Test that storage still accepts plain string lists."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        with patch('crewai.knowledge.storage.knowledge_storage.get_rag_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            storage = KnowledgeStorage()
            documents = ["chunk1", "chunk2", "chunk3"]
            storage.save(documents)
            
            mock_client_instance.add_documents.assert_called_once()
            saved_docs = mock_client_instance.add_documents.call_args[1]["documents"]
            assert len(saved_docs) == 3
            assert all("content" in doc for doc in saved_docs)

    def test_storage_accepts_dict_list(self):
        """Test that storage accepts dict lists with metadata."""
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        with patch('crewai.knowledge.storage.knowledge_storage.get_rag_client') as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            storage = KnowledgeStorage()
            documents = [
                {
                    "content": "chunk1",
                    "metadata": {"filepath": "/path/to/file.txt", "chunk_index": 0}
                },
                {
                    "content": "chunk2",
                    "metadata": {"filepath": "/path/to/file.txt", "chunk_index": 1}
                }
            ]
            storage.save(documents)
            
            mock_client_instance.add_documents.assert_called_once()
            saved_docs = mock_client_instance.add_documents.call_args[1]["documents"]
            assert len(saved_docs) == 2
            assert all("content" in doc for doc in saved_docs)
            assert all("metadata" in doc for doc in saved_docs)


class TestCrewDoclingSourceMetadata:
    """Test CrewDoclingSource metadata with conversion failures."""

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip") or pytest.importorskip("docling", reason="docling not available") is None,
        reason="docling not available"
    )
    def test_docling_filepath_metadata_with_conversion_failure(self, tmp_path):
        """Test that filepath metadata is correct even when some files fail conversion."""
        try:
            from pathlib import Path
            from unittest.mock import MagicMock, Mock
            from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
            from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
            
            file1 = tmp_path / "file1.txt"
            file2 = tmp_path / "file2.txt"
            file3 = tmp_path / "file3.txt"
            
            file1.write_text("Content from file 1")
            file2.write_text("Content from file 2")
            file3.write_text("Content from file 3")
            
            mock_doc1 = MagicMock()
            mock_doc3 = MagicMock()
            
            mock_result1 = MagicMock()
            mock_result1.document = mock_doc1
            mock_result1.input.file = file1
            
            mock_result3 = MagicMock()
            mock_result3.document = mock_doc3
            mock_result3.input.file = file3
            
            with patch("crewai.knowledge.source.crew_docling_source.DocumentConverter") as mock_converter_class:
                mock_converter = MagicMock()
                mock_converter_class.return_value = mock_converter
                mock_converter.convert_all.return_value = iter([mock_result1, mock_result3])
                mock_converter.allowed_formats = []
                
                with patch.object(KnowledgeStorage, 'save') as mock_save:
                    with patch("crewai.knowledge.source.crew_docling_source.CrewDoclingSource._chunk_doc") as mock_chunk:
                        mock_chunk.side_effect = [
                            iter(["Chunk 1 from file1", "Chunk 2 from file1"]),
                            iter(["Chunk 1 from file3", "Chunk 2 from file3"])
                        ]
                        
                        storage = KnowledgeStorage()
                        source = CrewDoclingSource(
                            file_paths=[file1, file2, file3],
                            storage=storage
                        )
                        
                        source.add()
                        
                        assert len(source.chunks) == 4
                        
                        assert source.chunks[0]["metadata"]["filepath"] == str(file1)
                        assert source.chunks[0]["metadata"]["source_type"] == "docling"
                        assert source.chunks[0]["metadata"]["chunk_index"] == 0
                        
                        assert source.chunks[1]["metadata"]["filepath"] == str(file1)
                        assert source.chunks[1]["metadata"]["chunk_index"] == 1
                        
                        assert source.chunks[2]["metadata"]["filepath"] == str(file3)
                        assert source.chunks[2]["metadata"]["source_type"] == "docling"
                        assert source.chunks[2]["metadata"]["chunk_index"] == 0
                        
                        assert source.chunks[3]["metadata"]["filepath"] == str(file3)
                        assert source.chunks[3]["metadata"]["chunk_index"] == 1
                        
                        for chunk in source.chunks:
                            assert chunk["metadata"]["filepath"] != str(file2)
        except ImportError:
            pytest.skip("docling not available")
