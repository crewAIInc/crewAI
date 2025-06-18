"""Test optional dependency handling for crewAI lite version."""

import pytest
from unittest.mock import patch, Mock


class TestOptionalDependencies:
    """Test that optional dependencies are handled gracefully."""

    def test_chromadb_import_error_memory(self):
        """Test that memory functionality raises helpful error without chromadb."""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('crewai.memory.storage.rag_storage.CHROMADB_AVAILABLE', False):
                from crewai.memory.storage.rag_storage import RAGStorage
                
                with pytest.raises(ImportError) as exc_info:
                    RAGStorage("test")
                
                assert "ChromaDB is required" in str(exc_info.value)
                assert "crewai[memory]" in str(exc_info.value)

    def test_chromadb_import_error_knowledge(self):
        """Test that knowledge functionality raises helpful error without chromadb."""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('crewai.knowledge.storage.knowledge_storage.CHROMADB_AVAILABLE', False):
                from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
                
                with pytest.raises(ImportError) as exc_info:
                    KnowledgeStorage()
                
                assert "ChromaDB is required" in str(exc_info.value)
                assert "crewai[knowledge]" in str(exc_info.value)

    def test_pdfplumber_import_error(self):
        """Test that PDF knowledge source raises helpful error without pdfplumber."""
        with patch.dict('sys.modules', {'pdfplumber': None}):
            with patch('crewai.knowledge.source.pdf_knowledge_source.PDFPLUMBER_AVAILABLE', False):
                from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
                
                from pathlib import Path
                knowledge_dir = Path("knowledge")
                knowledge_dir.mkdir(exist_ok=True)
                test_file = knowledge_dir / "test.pdf"
                test_file.touch()
                
                try:
                    with pytest.raises(ImportError) as exc_info:
                        PDFKnowledgeSource(file_paths=["test.pdf"])
                    
                    assert "pdfplumber is required" in str(exc_info.value)
                    assert "crewai[knowledge]" in str(exc_info.value)
                finally:
                    if test_file.exists():
                        test_file.unlink()
                    if knowledge_dir.exists() and not any(knowledge_dir.iterdir()):
                        knowledge_dir.rmdir()

    def test_pyvis_import_error(self):
        """Test that flow visualization raises helpful error without pyvis."""
        with patch.dict('sys.modules', {'pyvis': None}):
            with patch('crewai.flow.flow_visualizer.PYVIS_AVAILABLE', False):
                from crewai.flow.flow_visualizer import plot_flow
                
                mock_flow = Mock()
                
                with pytest.raises(ImportError) as exc_info:
                    plot_flow(mock_flow)
                
                assert "Pyvis is required" in str(exc_info.value)
                assert "crewai[visualization]" in str(exc_info.value)

    def test_auth0_import_error(self):
        """Test that authentication raises helpful error without auth0."""
        with patch.dict('sys.modules', {'auth0': None}):
            with patch('crewai.cli.authentication.utils.AUTH0_AVAILABLE', False):
                from crewai.cli.authentication.utils import validate_token
                
                with pytest.raises(ImportError) as exc_info:
                    validate_token("fake_token")
                
                assert "Auth0 is required" in str(exc_info.value)
                assert "crewai[auth]" in str(exc_info.value)

    def test_aisuite_import_error(self):
        """Test that AISuite LLM raises helpful error without aisuite."""
        with patch.dict('sys.modules', {'aisuite': None}):
            with patch('crewai.llms.third_party.ai_suite.AISUITE_AVAILABLE', False):
                from crewai.llms.third_party.ai_suite import AISuiteLLM
                
                with pytest.raises(ImportError) as exc_info:
                    AISuiteLLM("test-model")
                
                assert "AISuite is required" in str(exc_info.value)
                assert "crewai[llm-integrations]" in str(exc_info.value)

    def test_opentelemetry_graceful_degradation(self):
        """Test that telemetry degrades gracefully without opentelemetry."""
        with patch.dict('sys.modules', {'opentelemetry': None}):
            with patch('crewai.telemetry.telemetry.OPENTELEMETRY_AVAILABLE', False):
                from crewai.telemetry.telemetry import Telemetry
                
                telemetry = Telemetry()
                
                assert not telemetry.ready
                assert telemetry._is_telemetry_disabled()
                assert not telemetry._should_execute_telemetry()

    def test_embedding_configurator_import_error(self):
        """Test that embedding configurator raises helpful error without chromadb."""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('crewai.utilities.embedding_configurator.CHROMADB_AVAILABLE', False):
                from crewai.utilities.embedding_configurator import EmbeddingConfigurator
                
                configurator = EmbeddingConfigurator()
                
                with pytest.raises(ImportError) as exc_info:
                    configurator.configure_embedder(None)
                
                assert "ChromaDB is required" in str(exc_info.value)
                assert "crewai[memory]" in str(exc_info.value) or "crewai[knowledge]" in str(exc_info.value)

    def test_docling_import_error(self):
        """Test that docling knowledge source raises helpful error without docling."""
        with patch.dict('sys.modules', {'docling': None}):
            with patch('crewai.knowledge.source.crew_docling_source.DOCLING_AVAILABLE', False):
                from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
                
                with pytest.raises(ImportError) as exc_info:
                    CrewDoclingSource()
                
                assert "docling package is required" in str(exc_info.value)
                assert "uv add docling" in str(exc_info.value)
