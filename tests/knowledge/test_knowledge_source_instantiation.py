import pytest
from pathlib import Path
from unittest.mock import patch

from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

@patch('crewai.knowledge.source.base_file_knowledge_source.BaseFileKnowledgeSource.validate_content')
@patch('crewai.knowledge.source.pdf_knowledge_source.PDFKnowledgeSource.load_content')
def test_pdf_knowledge_source_instantiation(mock_load_content, mock_validate_content, tmp_path):
    """Test that PDFKnowledgeSource can be instantiated without errors."""
    mock_load_content.return_value = {}
    
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()  # Create the file
    
    pdf_source = PDFKnowledgeSource(file_paths=[pdf_path])
    assert isinstance(pdf_source, PDFKnowledgeSource)
