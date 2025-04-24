import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestEmbeddingConfiguratorImports:
    """Test that ChromaDB is not imported at module level."""
    
    def test_no_chromadb_import_at_module_level(self):
        """Test that chromadb is not imported when the module is imported."""
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('crewai.utilities.embedding_configurator'):
                del sys.modules[module_name]
        
        mock_chromadb = MagicMock()
        
        chromadb_imported = [False]
        
        def mock_import(name, *args, **kwargs):
            if name == 'chromadb':
                chromadb_imported[0] = True
                return mock_chromadb
            return importlib.__import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            from crewai.utilities import embedding_configurator
            
            assert not chromadb_imported[0], "chromadb was imported at module level"
    
    def test_chromadb_import_in_configure_embedder(self):
        """Test that chromadb is imported when configure_embedder is called."""
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('crewai.utilities.embedding_configurator'):
                del sys.modules[module_name]
        
        from crewai.utilities.embedding_configurator import EmbeddingConfigurator
        
        mock_chromadb = MagicMock()
        
        def mock_import(name, *args, **kwargs):
            if name == 'chromadb':
                raise ImportError("Mock import error for chromadb")
            return importlib.__import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match="Mock import error for chromadb"):
                EmbeddingConfigurator().configure_embedder()
