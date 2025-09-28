import hashlib
import os
import tempfile

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.text_loader import TextFileLoader, TextLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


def write_temp_file(content, suffix=".txt", encoding="utf-8"):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding=encoding
    ) as f:
        f.write(content)
        return f.name


def cleanup_temp_file(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class TestTextFileLoader:
    def test_basic_text_file(self):
        content = "This is test content\nWith multiple lines\nAnd more text"
        path = write_temp_file(content)
        try:
            result = TextFileLoader().load(SourceContent(path))
            assert isinstance(result, LoaderResult)
            assert result.content == content
            assert result.source == path
            assert result.doc_id
            assert result.metadata in (None, {})
        finally:
            cleanup_temp_file(path)

    def test_empty_file(self):
        path = write_temp_file("")
        try:
            result = TextFileLoader().load(SourceContent(path))
            assert result.content == ""
        finally:
            cleanup_temp_file(path)

    def test_unicode_content(self):
        content = "Hello 世界 🌍 émojis 🎉 åäö"
        path = write_temp_file(content)
        try:
            result = TextFileLoader().load(SourceContent(path))
            assert content in result.content
        finally:
            cleanup_temp_file(path)

    def test_large_file(self):
        content = "\n".join(f"Line {i}" for i in range(100))
        path = write_temp_file(content)
        try:
            result = TextFileLoader().load(SourceContent(path))
            assert "Line 0" in result.content
            assert "Line 99" in result.content
            assert result.content.count("\n") == 99
        finally:
            cleanup_temp_file(path)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            TextFileLoader().load(SourceContent("/nonexistent/path.txt"))

    def test_permission_denied(self):
        path = write_temp_file("Some content")
        os.chmod(path, 0o000)
        try:
            with pytest.raises(PermissionError):
                TextFileLoader().load(SourceContent(path))
        finally:
            os.chmod(path, 0o644)
            cleanup_temp_file(path)

    def test_doc_id_consistency(self):
        content = "Consistent content"
        path = write_temp_file(content)
        try:
            loader = TextFileLoader()
            result1 = loader.load(SourceContent(path))
            result2 = loader.load(SourceContent(path))
            expected_id = hashlib.sha256((path + content).encode("utf-8")).hexdigest()
            assert result1.doc_id == result2.doc_id == expected_id
        finally:
            cleanup_temp_file(path)

    def test_various_extensions(self):
        content = "Same content"
        for ext in [".txt", ".md", ".log", ".json"]:
            path = write_temp_file(content, suffix=ext)
            try:
                result = TextFileLoader().load(SourceContent(path))
                assert result.content == content
            finally:
                cleanup_temp_file(path)


class TestTextLoader:
    def test_basic_text(self):
        content = "Raw text"
        result = TextLoader().load(SourceContent(content))
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert result.content == content
        assert result.source == expected_hash
        assert result.doc_id == expected_hash

    def test_multiline_text(self):
        content = "Line 1\nLine 2\nLine 3"
        result = TextLoader().load(SourceContent(content))
        assert "Line 2" in result.content

    def test_empty_text(self):
        result = TextLoader().load(SourceContent(""))
        assert result.content == ""
        assert result.source == hashlib.sha256("".encode("utf-8")).hexdigest()

    def test_unicode_text(self):
        content = "世界 🌍 émojis 🎉 åäö"
        result = TextLoader().load(SourceContent(content))
        assert content in result.content

    def test_special_characters(self):
        content = "!@#$$%^&*()_+-=~`{}[]\\|;:'\",.<>/?"
        result = TextLoader().load(SourceContent(content))
        assert result.content == content

    def test_doc_id_uniqueness(self):
        result1 = TextLoader().load(SourceContent("A"))
        result2 = TextLoader().load(SourceContent("B"))
        assert result1.doc_id != result2.doc_id

    def test_whitespace_text(self):
        content = "   \n\t   "
        result = TextLoader().load(SourceContent(content))
        assert result.content == content

    def test_long_text(self):
        content = "A" * 10000
        result = TextLoader().load(SourceContent(content))
        assert len(result.content) == 10000


class TestTextLoadersIntegration:
    def test_consistency_between_loaders(self):
        content = "Consistent content"
        text_result = TextLoader().load(SourceContent(content))
        file_path = write_temp_file(content)
        try:
            file_result = TextFileLoader().load(SourceContent(file_path))

            assert text_result.content == file_result.content
            assert text_result.source != file_result.source
            assert text_result.doc_id != file_result.doc_id
        finally:
            cleanup_temp_file(file_path)
