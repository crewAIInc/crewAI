import unittest
from unittest.mock import MagicMock, patch


class TestEmbeddingConfigurator(unittest.TestCase):
    @patch('crewai.utilities.embedding_configurator.CHROMADB_AVAILABLE', False)
    def test_embedding_configurator_with_chromadb_unavailable(self):
        from crewai.utilities.embedding_configurator import EmbeddingConfigurator
        
        # Create an instance of EmbeddingConfigurator
        configurator = EmbeddingConfigurator()
        
        # Verify that embedding_functions is empty
        self.assertEqual(configurator.embedding_functions, {})
        
        # Verify that configure_embedder returns None
        self.assertIsNone(configurator.configure_embedder())
        
    @patch('crewai.utilities.embedding_configurator.CHROMADB_AVAILABLE', True)
    def test_embedding_configurator_with_chromadb_available(self):
        from crewai.utilities.embedding_configurator import EmbeddingConfigurator
        
        # Create an instance of EmbeddingConfigurator
        configurator = EmbeddingConfigurator()
        
        # Verify that embedding_functions is not empty
        self.assertNotEqual(configurator.embedding_functions, {})
        
        # Mock the _create_default_embedding_function method
        configurator._create_default_embedding_function = MagicMock(return_value="mock_embedding_function")
        
        # Verify that configure_embedder returns the mock embedding function
        self.assertEqual(configurator.configure_embedder(), "mock_embedding_function")
